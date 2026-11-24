"""
General utils

"""
import ctypes
import functools
import os
import sys
from dataclasses import dataclass, fields

import cv2
import numpy as np
import onnxruntime as ort


@functools.cache
def cuda_usable():
    """
    Return True only if ONNX Runtime can actually run on a CUDA GPU here.

    Called by get_onnx_device() and check_onnx_cuda(). Callers should use those
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
         that have the libraries but no driver or no GPU.
    """
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        return False  # CPU-only onnxruntime build (e.g. macOS)

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
            return False
        return count.value > 0
    except OSError:
        return False


def softmax(x):
    '''
    Helper function to softmax
    '''
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)


def get_onnx_device(user_set=None, quiet=False):
    """
    Get gpu if available
    """
    if cuda_usable():
        # user selects cuda device and is available
        if user_set in ['cpu', 'CPUExecutionProvider']:
            providers = ['CPUExecutionProvider']
            if not quiet:
                print('CUDA is available but set to cpu by user.')
        # user selects cuda device and is available
        elif user_set in ['CUDAExecutionProvider', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']:
            device_number = int(user_set.split(':')[-1]) if ':' in user_set else 0
            providers = [('CUDAExecutionProvider', {'device_id': device_number}), 'CPUExecutionProvider']
            if not quiet:
                print(f'Attempting to use CUDA device: {user_set}')
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
        if user_set is not None and user_set in ['cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']:
            if not quiet:
                print('Warning: CUDA device specified but not available, using CPU instead.')
        else:
            if not quiet:
                print('Using CPU device.')
       
    return providers
 
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

ASPECT_MODES = ("wide", "tall") # aspect_mode=None means no aspect-ratio preference

@dataclass(frozen=True)
class ScoreWeights:
    """
    Weights for the frame score (see _frame_score). Each weight must be a finite number >= 0 and
    the four must sum to 1, so a frame score always stays between 0 and 1. With aspect_mode=None the
    aspect weight is dropped and the other three are rescaled to sum to 1.

    The defaults were chosen by comparing scoring options on hand-checked leopard videos
    (issue #20): confidence barely varies between a good and a bad frame, so it gets a
    small weight; box size and the sharpness of the animal itself do most of the work.

    Args:
        conf (float): detection confidence
        area (float): bounding-box area (fraction of the image)
        aspect (float): aspect-ratio preference (ignored when aspect_mode is None)
        clarity (float): sharpness of the animal (Laplacian variance of the bbox crop)
    """
    conf: float = 0.10
    area: float = 0.55
    aspect: float = 0.10
    clarity: float = 0.25

    def __post_init__(self):
        weights = [self.conf, self.area, self.aspect, self.clarity]
        #checks to see if weights are non negative int/float
        for value in weights:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <0:
                raise ValueError("Weights must be non-negative numbers")

        #checks that weights add up to 1, otherwise raise value error
        if not(0.999999 <= sum(weights) <= 1.000001):
            raise ValueError("Weights must sum to 1")    

    @classmethod
    def from_dict(cls, weights):
        """Build from a dict with keys conf, area, aspect and clarity; raises ValueError on missing or unknown keys."""
        names = {f.name for f in fields(cls)}
        missing = names - set(weights)
        unknown = set(weights) - names
        if missing or unknown:
            raise ValueError(f"score weights need exactly the keys {sorted(names)}; "
                             f"missing: {sorted(missing)}, unknown: {sorted(unknown)}")
        return cls(**weights)


DEFAULT_SCORE_WEIGHTS = ScoreWeights()


def _coerce_weights(weights):
    """
    Converts input weights to a validated ScoreWeights instance.
    
    Args:
        weights (ScoreWeights, dict or None): Frame scoring weights. If None, the default weights are used.
        If a dict, it must contain exactly the keys conf, area, aspect and clarity.
        
    Returns:
        ScoreWeights: Validated scoring weights.

    Raises:
        TypeError: If weights is not a ScoreWeights instance, dict or None.
        ValueError: If the dict contains missing or unknown keys, or the weights are invalid or do not sum to 1.    
    """
    if weights is None:
        return DEFAULT_SCORE_WEIGHTS
    if isinstance(weights, ScoreWeights):
        return weights
    if isinstance(weights, dict):
        return ScoreWeights.from_dict(weights)
    raise TypeError(f"weights must be a ScoreWeights, a dict or None, got {type(weights).__name__}")


def _normalize_aspect_mode(mode):
    """
    Validates and normalizes the preferred bounding-box aspect ratio mode.

    Args:
        mode (str or None): Preferred orientation of the bounding box.
            'wide' rewards wider boxes, 'tall' rewards taller boxes, and
            None indicates that aspect ratio should be ignored.
            The string 'none' is also accepted for backwards compatibility.

    Returns:
        str or None: The normalized aspect mode ('wide', 'tall' or None).

    Raises:
        ValueError: If mode is not None, 'none', 'wide' or 'tall'.
    """
    if mode is None:
        return None

    if isinstance(mode, str):
        mode = mode.lower()

    if mode == "none":
        return None

    if mode not in ASPECT_MODES:
        raise ValueError(f"aspect_mode must be None or one of {ASPECT_MODES}, got '{mode}'")

    return mode


def _crop_clarity(image, bbox):
    """
    Sharpness of the animal only: Laplacian variance of the bounding-box crop, scaled to [0, 1].

    Args:
        image (np.ndarray): H x W x C image the bbox refers to
        bbox (array-like): normalized (x_corner, y_corner, width, height) relative to `image`

    Returns:
        float: min(laplacian_variance / 500, 1.0); 0.0 if the crop is empty
    """
    height, width = image.shape[:2]
    x0 = int(np.clip(round(bbox[0] * width), 0, width))
    y0 = int(np.clip(round(bbox[1] * height), 0, height))
    x1 = int(np.clip(round((bbox[0] + bbox[2]) * width), 0, width))
    y1 = int(np.clip(round((bbox[1] + bbox[3]) * height), 0, height))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0
    return min(float(_laplacian_variance(image[y0:y1, x0:x1])) / 500, 1.0)


def _aspect_score(bbox_w, bbox_h, image_width, image_height, mode="wide"):
    """
    Rewards boxes with the preferred orientation, measured in PIXELS (stored boxes are
    normalized, which distorts the ratio on a non-square image).

    mode 'wide': width / height  (e.g. a leopard seen from the side)
    mode 'tall': height / width  (e.g. a giraffe, or a bear standing up)
    mode None:   0.0 (aspect ratio is ignored)
    A ratio of 2:1 or more in the chosen direction gets the full score of 1.0.
    """
    mode = _normalize_aspect_mode(mode)
    if mode is None:
        return 0.0
    w_px = bbox_w * image_width
    h_px = bbox_h * image_height
    if w_px <= 0 or h_px <= 0:
        return 0.0
    ratio = w_px / h_px if mode == "wide" else h_px / w_px
    return min(ratio / 2.0, 1.0)


def _frame_score(conf, bbox_w, bbox_h, clarity, image_width, image_height,
                 aspect_mode=None, weights=None):
    """
    Weighted score used to pick the best frames of a video.

    Args:
        conf (float): detection confidence
        bbox_w, bbox_h (float): normalized box size relative to the original image
        clarity (float): crop clarity in [0, 1] (see _crop_clarity)
        image_width, image_height (int): original image size in pixels
        aspect_mode (str): None (default, species-neutral), 'wide' or 'tall' (see _aspect_score)
        weights (ScoreWeights or dict): defaults to DEFAULT_SCORE_WEIGHTS; a dict must have the keys
            conf, area, aspect and clarity

    With aspect_mode=None the aspect weight is dropped and the other weights are
    rescaled to sum to 1, so scores keep the same 0-1 range.
    """
    aspect_mode = _normalize_aspect_mode(aspect_mode)
    w = _coerce_weights(weights)
    if aspect_mode is None:
        total = w.conf + w.area + w.clarity
        if total <= 0:
            raise ValueError("conf, area and clarity weights must not all be zero when aspect_mode is None")
        return (w.conf * conf + w.area * bbox_w * bbox_h + w.clarity * clarity) / total
    return (w.conf * conf
            + w.area * bbox_w * bbox_h
            + w.aspect * _aspect_score(bbox_w, bbox_h, image_width, image_height, aspect_mode)
            + w.clarity * clarity)

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
    Converts bounding boxes from xyxy to xywh format.

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
 
