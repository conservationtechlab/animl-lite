"""
General utils

"""
import ctypes
import functools
import os
import sys
import warnings

import cv2
import numpy as np
import onnxruntime as ort


@functools.cache
def cuda_device_count():
    """
    Return the number of CUDA GPUs ONNX Runtime can actually run on here, 0 if none.

    Called by get_onnx_device() and cuda_usable(). Callers should use those
    rather than ort.get_available_providers() directly, because that function
    lists every provider onnxruntime-gpu was *built* with. It reports CUDA even
    on machines with no GPU, no driver, or no CUDA/cuDNN libraries. In those
    cases ORT logs a red "Failed to load libonnxruntime_providers_cuda" error
    on every session and quietly falls back to CPU.

    Kept as a separate cached function rather than inlined in get_onnx_device
    because it loads shared libraries and initialises the CUDA driver. That
    should happen once per process, not on every model load.

    Steps:
      1. preload_dlls() loads the CUDA 12 / cuDNN 9 libraries from the
         nvidia-*-cu12 pip packages, if installed. ORT's CUDA provider has no
         RUNPATH to site-packages, so it can't find them by itself.
      2. Load ORT's CUDA provider library. If any of its CUDA/cuDNN dependencies
         are missing, this raises OSError.
      3. Ask the NVIDIA driver how many GPUs it sees. This catches machines
         that have the libraries but no driver or no GPU. The count follows
         CUDA_VISIBLE_DEVICES, as ORT's device_id does.
    """
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        return 0  # CPU-only onnxruntime build (e.g. macOS)

    try:
        ort.preload_dlls()
    except Exception:
        pass  # fall back to system libs / LD_LIBRARY_PATH

    win = sys.platform == 'win32'
    capi = os.path.join(os.path.dirname(ort.__file__), 'capi')
    provider = 'onnxruntime_providers_cuda.dll' if win else 'libonnxruntime_providers_cuda.so'
    try:
        ctypes.CDLL(os.path.join(capi, provider))
        driver = ctypes.WinDLL('nvcuda.dll') if win else ctypes.CDLL('libcuda.so.1')
        count = ctypes.c_int(0)
        if driver.cuInit(0) != 0 or driver.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return 0
        return count.value
    except OSError:
        return 0


def cuda_usable():
    """
    Return True only if ONNX Runtime can actually run on a CUDA GPU here.

    Called by check_onnx_cuda(). See cuda_device_count() for how this is checked.
    """
    return cuda_device_count() > 0


def _cuda_index(user_set):
    """Device index from 'cuda' / 'cuda:N' / 'CUDAExecutionProvider', None for anything else"""
    if user_set in ('cuda', 'CUDAExecutionProvider'):
        return 0
    if isinstance(user_set, str) and user_set.startswith('cuda:') and user_set[5:].isdigit():
        return int(user_set[5:])
    return None


def get_onnx_device(user_set=None, quiet=False):
    """
    Get gpu if available
    """
    n_gpus = cuda_device_count()
    device_number = _cuda_index(user_set)
    if n_gpus > 0:
        # user selects cpu
        if user_set in ['cpu', 'CPUExecutionProvider']:
            providers = ['CPUExecutionProvider']
            if not quiet:
                print('CUDA is available but set to cpu by user.')
        # user selects cuda device and is available
        elif device_number is not None:
            # the selected GPU doesn't exist, but others do: use the first rather than the CPU
            if device_number >= n_gpus:
                warnings.warn(f'CUDA device {device_number} was selected, but only {n_gpus} CUDA '
                              f'device(s) are available (cuda:0 to cuda:{n_gpus - 1}). Using cuda:0.',
                              RuntimeWarning, stacklevel=2)
                device_number = 0
            providers = [('CUDAExecutionProvider', {'device_id': device_number}), 'CPUExecutionProvider']
            if not quiet:
                print(f'Attempting to use CUDA device: cuda:{device_number}')
        # no user input
        elif user_set is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']     
            if not quiet:
                print('Using available CUDA device.')
        # unknown user input
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if not quiet:
                print('User-specified device unknown, using available CUDA device.')

    # cuda not available
    else:
        # CoreML/MPS not compatible at this time
        providers = ['CPUExecutionProvider']
        # warn user if they specified cuda but it is not available
        if device_number is not None:
            if not quiet:
                print('Warning: CUDA device specified but not available, using CPU instead.')
        else:
            if not quiet:
                print('Using CPU device.')
       
    return providers


def create_onnx_session(model_path, providers):
    """
    Create an ONNX Runtime session, warning if CUDA was requested but not used

    cuda_usable() checks that the CUDA libraries load and a GPU is present, but
    the CUDA provider can still fail when the session is created (driver too old
    for the CUDA runtime, an unsupported GPU, a device index that doesn't exist,
    no free GPU memory). ONNX Runtime then runs on CPU, only logging an error,
    so models run many times slower with no sign of why in a GUI or notebook.

    The warning is not affected by quiet: it reports a problem, not progress.

    Args:
        model_path (str | Path): path to the .onnx file
        providers (list): execution providers, as returned by get_onnx_device

    Returns:
        onnxruntime.InferenceSession
    """
    session = ort.InferenceSession(str(model_path), providers=providers)
    requested = [p[0] if isinstance(p, tuple) else p for p in providers]
    if 'CUDAExecutionProvider' in requested and 'CUDAExecutionProvider' not in session.get_providers():
        warnings.warn(f'CUDA was requested for {model_path} but could not be used, running on CPU. '
                      'Check that the NVIDIA driver, CUDA and cuDNN versions match onnxruntime-gpu.',
                      RuntimeWarning, stacklevel=3)
    return session


# ==============================================================================
# FRAME SELECTION
# ==============================================================================

def _laplacian_variance(image):
    """Calculate Laplacian variance for sharpness"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Scale float [0,1] to uint8 [0,255]
    if gray.dtype == np.float32 or gray.dtype == np.float64:
        gray = (gray * 255).astype(np.uint8)
    
    # Compute Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    return laplacian.var()

# ==============================================================================
# COORDINATE CONVERSION
# ==============================================================================

def _xywh2xyxy(bbox):
    """
    Converts bounding boxes from xywh to xyxy format.

    Args:
        bbox (list): Bounding box coordinates in the format [x_min, y_min, width, height].

    Returns:
        list: Normalized bounding box coordinates in the format [x_min, y_min, width, height].
    """
    y = np.copy(bbox)
    y[2] = y[0] + y[2]  # bottom right x
    y[3] = y[1] + y[3]  # bottom right y
    return y

# THIS ONE
def _xyxy2xywh(bbox):
    """
    Converts bounding boxes from xywh to xyxy format.

    Args:
        bbox (list): Bounding box coordinates in the format [x_min, y_min, width, height].
                     x_min,y_min are the top left corner.

    Returns:
        list: Normalized bounding box coordinates in the format [x_min, y_min, width, height].
    """
    y = np.copy(bbox)
    y[2] = y[2] - y[0]  # width
    y[3] = y[3] - y[1]  # height
    return y


def _xywh_to_xywhc(bbox):
    """
    Converts bounding boxes from xywh to xywhc format.

    Args:
        bbox (list): Bounding box coordinates in the format [x_min, y_min, width, height].
                     x_min,y_min are the top left corner.
    Returns:
        list: Normalized bounding box coordinates in the format [x_center, y_center, width, height].
    """
    y = np.copy(bbox)
    y[0] = y[0] + y[2] / 2  # x center
    y[1] = y[1] + y[3] / 2  # y center
    return y


def _xywh_to_absxyxy(bbox, width, height):
    """
    Converts bounding box from [x_min, y_min, width, height] to [x1, y1, x2, y2] format.
    Used for converting annotation bounding boxes to absolute pixel coordinates for
    visualization and evaluation. (plot_box)

    Args:
        bbox (list): Bounding box in the format [x_min, y_min, width, height].
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        list: Bounding box in the format [x1, y1, x2, y2].
    """
    x_min, y_min, w, h = bbox
    x1 = x_min
    y1 = y_min
    x2 = x_min + w
    y2 = y_min + h

    return [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]


def _normalize_boxes(bbox, image_sizes):
    """
    Converts absolute bounding box coordinates to relative coordinates.

    Args:
        bbox (list): Absolute bounding box coordinates.
        img_size (tuple): Image size in the format (width, height).

    Returns:
        list: Normalized bounding box coordinates.
    """
    img_height, img_width  = image_sizes
    y = np.copy(bbox)   
    y[[0,2]] = np.clip(y[[0,2]] / img_width, 0, 1)
    y[[1,3]] = np.clip(y[[1,3]] / img_height, 0, 1)
    return y


def _scale_letterbox(bbox, resized_shape, original_shape):
    """
    Converts bounding box coordinates from a resized, letterboxed image space
    back to the original image's coordinate space. Assumes input coordinates
    are in normalized [x_corner, y_corner, width, height] format.

    Args:
        bbox (np.ndarray): A numpy array or tensor of bounding
                                             boxes, shape (n, 4), in
                                             (x_corner, y_corner, width, height) format.
                                             Coordinates are in pixels relative
                                             to the resized/padded image.
        resized_shape (tuple): The (height, width) of the resized and
                               letterboxed image.
        original_shape (tuple): The (height, width) of the original image.

    Returns:
        np.ndarray: A numpy array of bounding boxes, shape (n, 4), with
                    coordinates in normalized (x_corner, y_corner, width, height)
                    format.
    """
    # Convert input xywh (top-left corner) to xyxy
    xyxy_coords = _xywh2xyxy(bbox)

    # Calculate the scaling ratio and padding
    ratio = min(resized_shape[0] / original_shape[0], resized_shape[1] / original_shape[1])
    new_unpad_shape = (int(round(original_shape[0] * ratio)), int(round(original_shape[1] * ratio)))
    dw = (resized_shape[1] - new_unpad_shape[1]) / 2  # x-padding
    dh = (resized_shape[0] - new_unpad_shape[0]) / 2  # y-padding

    # Remove padding from coordinates
    xyxy_coords[[0, 2]] -= (dw / resized_shape[1])
    xyxy_coords[[1, 3]] -= (dh /resized_shape[0])

    # Scale to original image size
    xyxy_coords[[0, 2]] = xyxy_coords[[0, 2]] *  resized_shape[1]/new_unpad_shape[1]
    xyxy_coords[[1, 3]] = xyxy_coords[[1, 3]] *  resized_shape[0]/new_unpad_shape[0]

    # Clip coordinates to be within the original image dimensions
    xyxy_coords[[0, 2]] = np.clip(xyxy_coords[[0, 2]], 0, 1)  
    xyxy_coords[[1, 3]] = np.clip(xyxy_coords[[1, 3]], 0, 1) 

    # Convert final xyxy to xywh (top-left corner)
    xywh_coords = _xyxy2xywh(xyxy_coords)

    return xywh_coords


# ==============================================================================
# MODEL OUTPUT
# ==============================================================================

def softmax(x):
    '''
    Helper function to softmax
    '''
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)


def _nms(boxes, scores, iou_thres):
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        r = order[1:]
        w = np.maximum(0, np.minimum(x2[i], x2[r]) - np.maximum(x1[i], x1[r]))
        h = np.maximum(0, np.minimum(y2[i], y2[r]) - np.maximum(y1[i], y1[r]))
        inter = w * h
        iou = inter / (areas[i] + areas[r] - inter + 1e-9)
        order = r[iou <= iou_thres]
    return np.array(keep, dtype=np.int64)

 
def non_max_suppression(raw, conf_thres=0.1, iou_thres=0.45, max_det=300,
                        max_nms=30000, agnostic=False):
    out = []
    for pred in raw:
        # prefilter on confidence threshold
        pred = pred[pred[:, 4] > conf_thres]
        if not len(pred):
            out.append(np.zeros((0, 6), np.float32))
            continue
        # class conf = obj * cls
        scores = pred[:, 5:] * pred[:, 4:5]                  
        cls = scores.argmax(1)
        conf = scores[np.arange(len(pred)), cls]
        keep = conf > conf_thres
        pred, cls, conf = pred[keep], cls[keep], conf[keep]
        # empty after threshold
        if not len(pred):
            out.append(np.zeros((0, 6), np.float32))
            continue
        # convert xywh(center) -> xyxy
        cx, cy, w, h = pred[:, :4].T                        
        boxes = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], 1)
        if len(boxes) > max_nms:
            top = conf.argsort()[::-1][:max_nms]
            boxes, cls, conf = boxes[top], cls[top], conf[top]
        # per-class NMS
        offset = 0 if agnostic else cls[:, None] * 7680     
        k = _nms(boxes + offset, conf, iou_thres)[:max_det]
        out.append(np.concatenate([boxes[k], conf[k, None], cls[k, None]], 1).astype(np.float32))
    return out
 
