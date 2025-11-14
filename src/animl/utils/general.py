"""
General utils

"""
import cv2
import numpy as np
import onnxruntime as ort


def softmax(x):
    '''
    Helper function to softmax
    '''
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)


def tensor_to_onnx(tensor, channel_last=False):
    '''
    Helper function for onnx, shifts dims to BxHxWxC
    '''
    if channel_last:
        tensor = tensor.permute(0, 2, 3, 1)  # reorder BxCxHxW to BxHxWxC

    tensor = tensor.numpy()

    return tensor


def get_device(quiet=False):
    """
    Get gpu if available
    """
    if "CUDAExecutionProvider" in ort.get_available_providers():
        if not quiet:
            print("CUDA is available. Using GPU for inference.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        if not quiet:
            print("CUDA is not available (onnxruntime CPU-only).")
        return ["CPUExecutionProvider"]

 
# ==============================================================================
# COORDINATE CONVERSION
# ==============================================================================
def xyxyc2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywhc2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxyc2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def xywh2xyxy(bbox):
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
def xyxy2xywh(bbox):
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


def convert_minxywh_to_absxyxy(bbox, width, height):
    """
    Converts bounding box from [x_min, y_min, width, height] to [x1, y1, x2, y2] format.

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



def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def normalize_boxes(bbox, image_sizes):
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


# ==============================================================================
# Augmentations
# ==============================================================================

def letterbox(im: np.ndarray, new_shape = (640, 640),
              color = (114, 114, 114),
              auto: bool = True,
              scaleFill: bool = False,
              scaleup: bool = True,
              stride: int = 32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def scale_letterbox(bbox, resized_shape, original_shape):
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
    xyxy_coords = xywh2xyxy(bbox)

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
    xywh_coords = xyxy2xywh(xyxy_coords)

    return xywh_coords
