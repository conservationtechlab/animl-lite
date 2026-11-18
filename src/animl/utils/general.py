"""
General utils

"""
import math
from dataclasses import dataclass, fields

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


# ==============================================================================
# FRAME SCORING
# Code by Nikita Sharma
# Assisted by Claude
#
# These functions give each detection a "frame score" between 0 and 1, used to
# pick the best frames of a video. A good frame shows the animal large, sharp,
# confidently detected and (optionally) in a preferred shape, e.g. side-on.
# The score is a weighted sum of four parts:
#     conf    - how sure the detector is that this is an animal
#     area    - how much of the image the bounding box covers
#     aspect  - how well the box's shape matches the preferred shape (optional)
#     clarity - how sharp the animal itself is (not the whole frame)
# ==============================================================================

# How close the weights must add up to 1.0. Floating-point math is not exact
# (1/3 + 1/3 + 1/3 can come out as 0.9999999999), so tiny rounding differences
# like that still count as "equal to 1".
WEIGHT_SUM_TOLERANCE = 1e-6

# A Laplacian variance at or above this value counts as perfectly sharp
# (clarity 1.0). Lower values are scaled down linearly toward 0.
LAPLACIAN_VARIANCE_FOR_FULL_CLARITY = 500

# A box this many times wider than tall (or taller than wide, in 'tall' mode)
# gets the full aspect score of 1.0.
ASPECT_RATIO_FOR_FULL_SCORE = 2.0

# A crop must be at least this many pixels wide and tall to measure sharpness.
MIN_CROP_SIZE_PX = 2


@dataclass(frozen=True)
class ScoreWeights:
    """
    How much each part counts toward the frame score (see _frame_score).

    Each weight is a share of the total: conf=0.10 means confidence makes up
    10% of the score. Every weight must be a finite number >= 0 and the four
    must sum to 1, so a frame score always stays between 0 and 1. With
    aspect_mode=None the aspect weight is dropped and the other three are
    rescaled so they still sum to 1.

    The class is frozen (read-only after creation) so the shared
    DEFAULT_SCORE_WEIGHTS can never be changed by accident.

    The defaults were chosen by comparing scoring options on hand-checked
    leopard videos (issue #20): confidence barely varies between a good and a
    bad frame, so it gets a small weight; box size and the sharpness of the
    animal itself do most of the work.

    Args:
        conf (float): weight for detection confidence
        area (float): weight for bounding-box area (fraction of the image)
        aspect (float): weight for the aspect-ratio preference (ignored when aspect_mode is None)
        clarity (float): weight for sharpness of the animal (Laplacian variance of the bbox crop)

    Raises:
        ValueError: if a weight is not a finite number >= 0, or the weights do not sum to 1
    """
    conf: float = 0.10
    area: float = 0.55
    aspect: float = 0.10
    clarity: float = 0.25

    def __post_init__(self):
        """
        Validate the weights as soon as a ScoreWeights is created.

        Python's dataclass calls this method automatically after __init__, so
        it must keep the name __post_init__. The checks themselves live in the
        two helper methods below. If either check fails, a ValueError is raised
        and the object is never created, so every ScoreWeights holds valid weights.
        """
        self._check_each_weight_is_a_finite_non_negative_number()
        self._check_weights_sum_to_one()

    def _check_each_weight_is_a_finite_non_negative_number(self):
        """
        Reject any weight that is not a real number >= 0.

        A weight is rejected if it is:
          - True or False (Python treats bools as ints, so True would otherwise pass as 1)
          - not a number at all (e.g. the string "0.5" or None)
          - infinite or NaN ("not a number")
          - negative

        Raises:
            ValueError: naming the first bad weight and its value
        """
        for weight_field in fields(self):
            weight_name = weight_field.name
            weight_value = getattr(self, weight_name)

            is_bool = isinstance(weight_value, bool)
            is_number = isinstance(weight_value, (int, float))

            if is_bool or not is_number:
                raise ValueError(f"score weight '{weight_name}' must be a number, got {weight_value!r}")
            if not math.isfinite(weight_value):
                raise ValueError(f"score weight '{weight_name}' must be finite (not infinity or NaN), "
                                 f"got {weight_value!r}")
            if weight_value < 0:
                raise ValueError(f"score weight '{weight_name}' must be >= 0, got {weight_value!r}")

    def _check_weights_sum_to_one(self):
        """
        Make sure the four weights add up to 1, allowing for tiny rounding errors.

        If they summed to more than 1 the frame score could go above 1; if they
        summed to less than 1 even a perfect frame could not reach 1.

        Raises:
            ValueError: if the total is not 1 (within WEIGHT_SUM_TOLERANCE)
        """
        total_of_all_weights = self.conf + self.area + self.aspect + self.clarity
        if not math.isclose(total_of_all_weights, 1.0, abs_tol=WEIGHT_SUM_TOLERANCE):
            raise ValueError("score weights must sum to 1 so scores stay between 0 and 1, "
                             f"got {total_of_all_weights:g}")

    @classmethod
    def from_dict(cls, weights):
        """
        Build a ScoreWeights from a dict, e.g. {'conf': 0.1, 'area': 0.55, 'aspect': 0.1, 'clarity': 0.25}.

        The dict must have exactly the keys conf, area, aspect and clarity. A
        missing or misspelled key raises an error instead of being silently
        ignored or filled in with a default.

        Args:
            weights (dict): weight name -> weight value

        Returns:
            ScoreWeights: the validated weights

        Raises:
            ValueError: on missing or unknown keys, or invalid weight values
        """
        expected_keys = {weight_field.name for weight_field in fields(cls)}
        given_keys = set(weights)
        missing_keys = expected_keys - given_keys
        unknown_keys = given_keys - expected_keys
        if missing_keys or unknown_keys:
            raise ValueError(f"score weights need exactly the keys {sorted(expected_keys)}; "
                             f"missing: {sorted(missing_keys)}, unknown: {sorted(unknown_keys)}")
        return cls(**weights)


# Shared default weights, used whenever the caller does not pass their own.
DEFAULT_SCORE_WEIGHTS = ScoreWeights()


def _coerce_weights(weights):
    """
    Turn whatever the caller passed as weights into a ScoreWeights.

    Args:
        weights (ScoreWeights, dict or None):
            None       -> DEFAULT_SCORE_WEIGHTS
            ScoreWeights -> returned unchanged
            dict       -> validated with ScoreWeights.from_dict

    Returns:
        ScoreWeights

    Raises:
        TypeError: if weights is any other type
    """
    if weights is None:
        return DEFAULT_SCORE_WEIGHTS
    if isinstance(weights, ScoreWeights):
        return weights
    if isinstance(weights, dict):
        return ScoreWeights.from_dict(weights)
    raise TypeError(f"weights must be a ScoreWeights, a dict or None, got {type(weights).__name__}")


# Accepted aspect-ratio preferences. aspect_mode=None means no preference.
ASPECT_MODES = ("wide", "tall")


def _normalize_aspect_mode(mode):
    """
    Check an aspect_mode value and return it in its standard form.

    Args:
        mode (str or None): 'wide', 'tall', None, or the string 'none'
            (accepted for backwards compatibility and treated like None)

    Returns:
        str or None: 'wide', 'tall' or None (no aspect-ratio preference)

    Raises:
        ValueError: for any other value
    """
    if mode is None or (isinstance(mode, str) and mode.lower() == "none"):
        return None
    if mode not in ASPECT_MODES:
        raise ValueError(f"aspect_mode must be None or one of {ASPECT_MODES}, got '{mode}'")
    return mode


def _crop_clarity(image, bbox):
    """
    Measure how sharp the animal is, using only the pixels inside its bounding box.

    Measuring the crop instead of the whole frame means a sharp background
    cannot make a blurry animal look sharp (and vice versa).

    Args:
        image (np.ndarray): H x W x C image the bbox refers to
        bbox (array-like): normalized (x_corner, y_corner, width, height) relative to `image`,
            each value between 0 and 1

    Returns:
        float: clarity between 0 and 1, i.e.
            min(laplacian_variance / LAPLACIAN_VARIANCE_FOR_FULL_CLARITY, 1.0);
            0.0 if the crop is too small to measure
    """
    image_height_px, image_width_px = image.shape[:2]
    box_x, box_y, box_width, box_height = bbox[0], bbox[1], bbox[2], bbox[3]

    # Convert the normalized box to pixel coordinates, kept inside the image.
    left_px = int(np.clip(round(box_x * image_width_px), 0, image_width_px))
    top_px = int(np.clip(round(box_y * image_height_px), 0, image_height_px))
    right_px = int(np.clip(round((box_x + box_width) * image_width_px), 0, image_width_px))
    bottom_px = int(np.clip(round((box_y + box_height) * image_height_px), 0, image_height_px))

    if right_px - left_px < MIN_CROP_SIZE_PX or bottom_px - top_px < MIN_CROP_SIZE_PX:
        return 0.0

    animal_crop = image[top_px:bottom_px, left_px:right_px]
    sharpness = float(_laplacian_variance(animal_crop))
    return min(sharpness / LAPLACIAN_VARIANCE_FOR_FULL_CLARITY, 1.0)


def _aspect_score(bbox_w, bbox_h, image_width, image_height, mode="wide"):
    """
    Reward boxes whose shape matches the preferred orientation.

    The ratio is measured in PIXELS, because stored boxes are normalized
    (0-1 of the image), which distorts the ratio on a non-square image.

        mode 'wide': width / height  (e.g. a leopard seen from the side)
        mode 'tall': height / width  (e.g. a giraffe, or a bear standing up)
        mode None:   always 0.0 (aspect ratio is ignored)

    A ratio of ASPECT_RATIO_FOR_FULL_SCORE (2:1) or more in the chosen
    direction gets the full score of 1.0; smaller ratios scale down linearly.

    Args:
        bbox_w, bbox_h (float): normalized box width and height
        image_width, image_height (int): original image size in pixels
        mode (str or None): 'wide', 'tall' or None

    Returns:
        float: aspect score between 0 and 1
    """
    mode = _normalize_aspect_mode(mode)
    if mode is None:
        return 0.0

    box_width_px = bbox_w * image_width
    box_height_px = bbox_h * image_height
    if box_width_px <= 0 or box_height_px <= 0:
        return 0.0

    if mode == "wide":
        shape_ratio = box_width_px / box_height_px
    else:
        shape_ratio = box_height_px / box_width_px
    return min(shape_ratio / ASPECT_RATIO_FOR_FULL_SCORE, 1.0)


def _frame_score(conf, bbox_w, bbox_h, clarity, image_width, image_height,
                 aspect_mode=None, weights=None):
    """
    Combine confidence, box size, shape and sharpness into one score used to pick the best frames of a video.

    Args:
        conf (float): detection confidence
        bbox_w, bbox_h (float): normalized box size relative to the original image
        clarity (float): crop clarity in [0, 1] (see _crop_clarity)
        image_width, image_height (int): original image size in pixels
        aspect_mode (str): None (default, species-neutral), 'wide' or 'tall' (see _aspect_score)
        weights (ScoreWeights or dict): defaults to DEFAULT_SCORE_WEIGHTS; a dict must have the keys
            conf, area, aspect and clarity

    Returns:
        float: frame score between 0 and 1 (higher is a better frame)

    With aspect_mode=None the aspect weight is dropped and the other weights are
    rescaled to sum to 1, so scores keep the same 0-1 range.
    """
    aspect_mode = _normalize_aspect_mode(aspect_mode)
    score_weights = _coerce_weights(weights)
    box_area = bbox_w * bbox_h

    if aspect_mode is None:
        # No shape preference: leave out the aspect part and divide by the
        # remaining weights so the score can still reach 1.
        weights_without_aspect = score_weights.conf + score_weights.area + score_weights.clarity
        if weights_without_aspect <= 0:
            raise ValueError("conf, area and clarity weights must not all be zero when aspect_mode is None")
        weighted_sum = (score_weights.conf * conf
                        + score_weights.area * box_area
                        + score_weights.clarity * clarity)
        return weighted_sum / weights_without_aspect

    shape_score = _aspect_score(bbox_w, bbox_h, image_width, image_height, aspect_mode)
    return (score_weights.conf * conf
            + score_weights.area * box_area
            + score_weights.aspect * shape_score
            + score_weights.clarity * clarity)

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
