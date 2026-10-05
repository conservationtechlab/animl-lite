"""
General utils

"""
import cv2
import numpy as np
import onnxruntime as ort


MEGADETECTORv5_SIZE = 1280
SDZWA_CLASSIFIER_SIZE = 299

MODEL_TYPES = {"megadetector", "yolo", "miewid", "classifier"}


def softmax(x):
    '''
    Helper function to softmax
    '''
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)


import onnxruntime as ort


def get_onnx_device(user_set=None, quiet=False):
    """
    Get the best available ONNX Runtime execution providers.

    user_set: None (auto), 'cpu', 'cuda', 'cuda:N', or 'mps'/'coreml' (macOS).
    Returns a list of providers in priority order, always ending with CPU.
    """
    def log(msg):
        if not quiet:
            print(msg)

    available = ort.get_available_providers()
    has_cuda = 'CUDAExecutionProvider' in available
    has_coreml = 'CoreMLExecutionProvider' in available
    cpu = ['CPUExecutionProvider']

    choice = user_set.lower().strip() if isinstance(user_set, str) else user_set

    # --- explicit CPU ---
    if choice in ('cpu', 'cpuexecutionprovider'):
        if has_cuda or has_coreml:
            log('GPU is available but CPU was set by user.')
        return cpu

    # --- explicit CUDA ---
    if choice in ('cuda', 'cudaexecutionprovider') or (isinstance(choice, str) and choice.startswith('cuda:')):
        if has_cuda:
            device_id = 0
            if ':' in choice:
                try:
                    device_id = int(choice.split(':')[-1])
                except ValueError:
                    log(f'Invalid CUDA device "{user_set}", using device 0.')
            log(f'Attempting to use CUDA device: {device_id}')
            return [('CUDAExecutionProvider', {'device_id': device_id}), *cpu]
        log('Warning: CUDA device specified but not available, using CPU instead.')
        return cpu

    # --- explicit Apple (MPS is the PyTorch name; ONNX Runtime uses CoreML) ---
    if choice in ('mps', 'coreml', 'coremlexecutionprovider'):
        if has_coreml:
            log('Attempting to use CoreML.')
            return ['CoreMLExecutionProvider', *cpu]
        log('Warning: CoreML specified but not available, using CPU instead.')
        return cpu

    # --- unknown user input ---
    if choice is not None:
        log(f'User-specified device "{user_set}" unknown, selecting automatically.')

    # --- automatic selection: CUDA > CoreML > CPU ---
    if has_cuda:
        log('Using available CUDA device.')
        return ['CUDAExecutionProvider', *cpu]
    if has_coreml:
        log('Using available CoreML device.')
        return ['CoreMLExecutionProvider', *cpu]

    log('No GPU available, using CPU.')
    return cpu


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
