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


def get_onnx_device(user_set=None, quiet=False):
    """
    Get gpu if available
    """
    providers = ort.get_available_providers()
    
    if 'CUDAExecutionProvider' in providers:
        # user selects cuda device and is available
        if user_set in ['cpu', 'CPUExecutionProvider']:
            if not quiet:
                print('CUDA is available but set to cpu by user.')
                providers = ['CPUExecutionProvider']
        # user selects cuda device and is available
        elif user_set in ['CUDAExecutionProvider', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']:
            device_number = int(user_set.split(':')[-1]) if ':' in user_set else 0
            providers = [('CUDAExecutionProvider', {'device_id': device_number}), 'CPUExecutionProvider']
            if not quiet:
                print(f'Attempting to use CUDA device: {user_set}')
        # no user input
        elif user_set is None:
            if not quiet:
                print('Using available CUDA device.')
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']     
        # unknown user input
        else:
            if not quiet:
                print('User-specified device unknown, using available CUDA device.')
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # cuda not available
    else:
        if user_set is not None and user_set in ['cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']:
            if not quiet:
                print('Warning: CUDA device specified but not available, using CPU instead.')
        # CoreML/MPS not compatible at this time
        providers = ['CPUExecutionProvider']
       
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


# Weights for the frame score (must sum to 1). Chosen by comparing scoring options on hand-checked
# leopard videos (issue #20): confidence barely varies between a good and a bad frame, so it gets a
# small weight; box size and the sharpness of the animal itself do most of the work.
SCORE_WEIGHTS = {'conf': 0.10, 'area': 0.55, 'aspect': 0.10, 'clarity': 0.25}
ASPECT_MODES = ("wide", "tall", "none")


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
    mode 'none': 0.0 (aspect ratio is ignored)
    A ratio of 2:1 or more in the chosen direction gets the full score of 1.0.
    """
    if mode not in ASPECT_MODES:
        raise ValueError(f"aspect_mode must be one of {ASPECT_MODES}, got '{mode}'")
    if mode == "none":
        return 0.0
    w_px = bbox_w * image_width
    h_px = bbox_h * image_height
    if w_px <= 0 or h_px <= 0:
        return 0.0
    ratio = w_px / h_px if mode == "wide" else h_px / w_px
    return min(ratio / 2.0, 1.0)


def _frame_score(conf, bbox_w, bbox_h, clarity, image_width, image_height,
                 aspect_mode="wide", weights=None):
    """
    Weighted score used to pick the best frames of a video.

    Args:
        conf (float): detection confidence
        bbox_w, bbox_h (float): normalized box size relative to the original image
        clarity (float): crop clarity in [0, 1] (see _crop_clarity)
        image_width, image_height (int): original image size in pixels
        aspect_mode (str): 'wide', 'tall' or 'none' (see _aspect_score)
        weights (dict): keys conf, area, aspect, clarity; defaults to SCORE_WEIGHTS
    """
    w = SCORE_WEIGHTS if weights is None else weights
    return (w['conf'] * conf
            + w['area'] * bbox_w * bbox_h
            + w['aspect'] * _aspect_score(bbox_w, bbox_h, image_width, image_height, aspect_mode)
            + w['clarity'] * clarity)

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
