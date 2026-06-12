"""
Object Detection Module

Functions for loading MegaDetector, as well as custom YOLO models
parse_detections() converts json output into a dataframe

"""
from shutil import copyfile
from typing import Optional
import time
from animl.utils.visualization import MD_LABELS
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import onnxruntime as ort

from animl import file_management
from animl.generator import manifest_dataloader
from animl.utils.general import _normalize_boxes, _xyxy2xywh, _scale_letterbox, get_onnx_device
from animl.utils.visualization import MD_LABELS


def load_detector(model_path: str, model_type: str = "megadetector", device: Optional[str] = None):
    """
    Load Detector model from filepath.

    Args:
        model_path (str): path to model file
        model_type (str): type of model expected ["megadetector", "yolo", "miewid"]
        device (str): specify to run on cpu or gpu

    Returns:
        object: loaded model object
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    providers = get_onnx_device(user_set=device)
    model = ort.InferenceSession(model_path, providers=providers)
    model.model_type = model_type.lower()
    return model


def detect(detector,
           image_file_names,
           resize_width: int,
           resize_height: int,
           letterbox: bool = True,
           category_map: Optional[dict] = MD_LABELS,
           confidence_threshold: float = 0.1,
           file_col: str = 'filepath',
           checkpoint_path: Optional[str] = None,
           checkpoint_frequency: int = -1) -> list[dict]:
    """
    Runs Detector model on a batches of image files.

    Args:
        detector (object): preloaded detector model
        image_file_names (mult): list of image filenames, a single image filename, or manifest
                                    containing a list of images.
        resize_width (int): width to resize images to
        resize_height (int): height to resize images to
        letterbox (bool): if True, resize and pad image to keep aspect ratio, else resize without padding
        category_map (dict): mapping of category IDs to human-readable labels
        confidence_threshold (float): only detections above this threshold are returned
        file_col (str): column name containing file paths
        device (str): specify to run on cpu or gpu
        checkpoint_path (str): path to checkpoint file
        checkpoint_frequency (int): write results to checkpoint file every N images

    Returns:
        list: list of dicts, each dict represents detections on one image
    """
    # convert map keys to int if they are string (ie from reticulate)
    category_map = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in category_map.items()}

    # Single image filepath
    if isinstance(image_file_names, str):
        # convert img path to tensor
        batch_from_dataloader = manifest_dataloader(pd.DataFrame([[image_file_names, 0]], columns=['filepath', 'frame']),
                                                    crop=False,
                                                    normalize=True,
                                                    letterbox=letterbox,
                                                    resize_width=resize_width,
                                                    resize_height=resize_height)

        input_name = detector.get_inputs()[0].name
        outputs = detector.run(None, {input_name: batch_from_dataloader[0]})[0]
        results = _convert_detections(outputs,
                                      batch_from_dataloader,
                                      letterbox,
                                      confidence_threshold=confidence_threshold,
                                      category_map=category_map,
                                      model_type=detector.model_type)
        return results

    # Full manifest, select file_col
    elif isinstance(image_file_names, pd.DataFrame):
        if file_col not in image_file_names.columns:
            raise ValueError(f"file_col {file_col} not found in manifest columns")
        # no frame column, assume all images and set to 0
        if 'frame' not in image_file_names.columns:
            print("Warning: 'frame' column not found in manifest columns. Defaulting to 0 assuming images.")
            image_file_names['frame'] = 0
        # create a list of image paths
        manifest = image_file_names[[file_col, 'frame']]

    # load checkpoint
    if checkpoint_path and file_management.check_file(checkpoint_path, output_type="Megadetector raw output"):
        results = file_management.load_json(checkpoint_path).get('images')
        already_processed = set([r['filepath'] for r in results])
        manifest = image_file_names[~image_file_names[file_col].isin(already_processed)][[file_col, 'frame']].reset_index(drop=True)
        if manifest.empty:
            print("All images have already been processed. Exiting.")
            return results
    else:
        results = []
        image_file_names = set(image_file_names)

    count = 0

    # create dataloader
    dataloader = manifest_dataloader(manifest,
                                     crop=False,
                                     normalize=True,
                                     letterbox=letterbox,
                                     resize_width=resize_width,
                                     resize_height=resize_height)

    start_time = time.time()
    for _, batch in tqdm(enumerate(dataloader), total=len(manifest)):
        count += 1

        # ONNX Runtime inference
        input_name = detector.get_inputs()[0].name
        outputs = detector.run(None, {input_name: batch[0]})[0]
        outputs = _convert_detections(outputs,
                                      batch,
                                      letterbox,
                                      confidence_threshold=confidence_threshold,
                                      category_map=category_map,
                                      model_type=detector.model_type)
        # Process outputs to match expected format
        results.extend(outputs)

        # Write a checkpoint if necessary
        if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
            print('Writing a new checkpoint after having processed {} images since last restart'.format(count))
            _save_detection_checkpoint(checkpoint_path, results)

    print(f"\nFinished detection. Total images processed: {len(results)} at {round(len(results)/(time.time() - start_time), 1)} img/s.")
    if checkpoint_path:
        _save_detection_checkpoint(checkpoint_path, results)

    return results


def _convert_detections(predictions: list,
                        batch_from_dataloader: list,
                        letterbox: bool,
                        confidence_threshold: float = 0.1,
                        model_type: str = 'megadetector',
                        category_map: dict = MD_LABELS) -> pd.DataFrame:
    # Converts output into nested list with categories, conf, and bboxes in expected format for parsing function.
    # Supports YOLOv5/MDv5, YOLOv6+, and ONNX models with either relative or absolute bounding box outputs.
    # If letterbox=True, rescales bboxes back to original image size.

    # unpack batch dataloader output
    image_tensors = batch_from_dataloader[0]
    image_paths = batch_from_dataloader[1]
    image_frames = batch_from_dataloader[2]
    image_sizes = batch_from_dataloader[3]

        # if no category map provided, default to MD_LABELS
    if category_map is None:
        print("No category map provided, defaulting to MD_LABELS. ",
              "This may lead to incorrect category labels if using a custom model.")
        category_map = MD_LABELS

    results = []

    for i, pred in enumerate(predictions):

        boxes = pred[:, :4]  # Bounding box coordinates
        conf = pred[:, 4]  # Confidence scores
        category = pred[:, 5]  # Class labels as integers
        max_detection_conf = float(round(conf.max(), 4)) if len(conf) > 0 else None

        # no detections
        if len(conf) == 0:
            # category is []
            data = {'filepath': str(image_paths[i]),
                    'frame': int(image_frames[i]),
                    'max_detection_conf': max_detection_conf,
                    # for MD models, set category to 0 (empty) if no detections, for other models set to None
                    'category': 0 if model_type == "megadetector" else None,
                    'category_label': 'empty',
                    'detections': []}
            results.append(data)
        else:
            detections = []
            for j in range(len(conf)):

                # skip low-confidence detections
                if conf[j] < confidence_threshold:
                    continue

                bbox = _normalize_boxes(boxes[j], image_tensors[i].shape[1:])
                bbox = _xyxy2xywh(bbox)
                if bbox.all() == 0:
                    continue

                if letterbox:
                    bbox = _scale_letterbox(bbox, image_tensors[i].shape[1:], image_sizes[i, :])

                # increase md categories by 1
                if model_type == "megadetector":
                    category[j] += 1

                # build detection dict
                detection = {'category': int(category[j]),
                             'category_label': category_map.get(int(category[j]), "unknown"),
                             'conf': float(round(conf[j], 4)),
                             'bbox_x': float(round(bbox[0], 4)),
                             'bbox_y': float(round(bbox[1], 4)),
                             'bbox_w': float(round(bbox[2], 4)),
                             'bbox_h': float(round(bbox[3], 4))}
                detections.append(detection)

            data = {'filepath': str(image_paths[i]),
                    'frame': int(image_frames[i]),
                    'max_detection_conf': max_detection_conf,
                    'detections': detections}
            results.append(data)

    return results


def parse_detections(detections: list[dict],
                     manifest: Optional[pd.DataFrame] = None,
                     out_file: Optional[str] = None,
                     threshold: float = 0,
                     file_col: str = "filepath"):
    """
    Converts listed output from detector to DataFrame.

    Args:
        detections (list[dict]): md output dicts
        manifest (pd.DataFrame): full file manifest, if not None, merge md predictions automatically
        out_file (str): path to save dataframe
        threshold (float): parse only detections above given confidence threshold
        file_col (str): if manifest, merge results onto file_col

    Returns:
        df (pd.DataFrame): formatted md outputs, one row per detection
    """
    if manifest is not None and file_col not in manifest.columns:
        raise ValueError(f"file_col '{file_col}' not found in manifest columns")

    if manifest is not None and 'frame' not in manifest.columns:
        print("""Warning: 'frame' column not found in manifest columns. Defaulting to 0 for all rows.""")
        manifest['frame'] = 0

    # check results format
    if not isinstance(detections, list):
        raise TypeError("MD results input must be list")
    if len(detections) == 0:
        raise AssertionError("'detections' contains no detections")

    # load results from file if they have already been parsed
    if file_management.check_file(out_file, output_type="Detections"):
        return file_management.load_data(out_file)

    lst = []
    for frame in tqdm(detections):
        try:
            frame_detections = frame['detections']
        except KeyError:
            print('File error ', frame['filepath'])
            continue

        if len(frame_detections) == 0:
            data = {'filepath': frame['filepath'],
                    'frame': frame['frame'],
                    'max_detection_conf': frame['max_detection_conf'],
                    'category': frame['category'] if 'category' in frame else None,
                    'category_label': frame['category_label'] if 'category_label' in frame else 'empty',
                    'conf': None, 'bbox_x': None, 'bbox_y': None, 'bbox_w': None, 'bbox_h': None}
            lst.append(data)

        else:
            for detection in frame_detections:
                if (detection['conf'] > threshold):
                    data = {'filepath': frame['filepath'],
                            'frame': frame['frame'],
                            'max_detection_conf': frame['max_detection_conf'],
                            'category': detection['category'],
                            'category_label': detection['category_label'],
                            'conf': detection['conf'],
                            'bbox_x': np.clip(detection['bbox_x'], 0, 1),
                            'bbox_y': np.clip(detection['bbox_y'], 0, 1),
                            'bbox_w': np.clip(detection['bbox_w'], 0, 1),
                            'bbox_h': np.clip(detection['bbox_h'], 0, 1)}
                    lst.append(data)

    df = pd.DataFrame(lst)

    if manifest is not None:
        if file_col in manifest.columns:
            df = manifest.merge(df, left_on=[file_col, 'frame'], right_on=["filepath", "frame"], how='left')
        else:
            raise ValueError("Please provide a manifest with a valid file_col to merge results onto.")

    if out_file:
        file_management.save_data(df, out_file)

    return df


def _save_detection_checkpoint(checkpoint_path: str, results: dict) -> None:
    """
    Save a checkpoint of the detection results to a JSON file.

    Args:
        checkpoint_path (str): the path to the checkpoint file
        results (list): a list of detection results to save
    """
    assert checkpoint_path is not None
    # Back up any previous checkpoints, to protect against crashes while we're writing
    # the checkpoint file.
    checkpoint_tmp_path = None
    if Path(checkpoint_path).is_file():
        checkpoint_tmp_path = str(checkpoint_path) + '_tmp'
        copyfile(checkpoint_path, checkpoint_tmp_path)

    # Write the new checkpoint
    file_management.save_json({'images': results}, checkpoint_path, prompt=False)

    # Remove the backup checkpoint if it exists
    if checkpoint_tmp_path is not None:
        Path(checkpoint_tmp_path).unlink()



def get_animals(manifest: pd.DataFrame):
    """
    Pulls MD animal detections for classification

    Args:
        manifest (pd.DataFrame): DataFrame containing one row for every MD detection

    Returns:
        subset of manifest containing only animal detections
    """
    if "category_label" in manifest.columns:
        return manifest[manifest["category_label"] == "animal"].reset_index(drop=True)
    # Removes all images that MegaDetector gave no detection for
    else:
        # make sure category column is int and fill NaN with 0 (empty)
        manifest["category"] = manifest["category"].fillna(0)
        # Pulls only the animal detections
        return manifest[manifest["category"].astype(int) == 1].reset_index(drop=True)


def get_empty(manifest: pd.DataFrame):
    """
    Pulls MD non-animal detections

    Args:
        manifest (pd.DataFrame): DataFrame containing one row for every MD detection

    Returns:
        otherdf: subset of manifest containing empty, vehicle and human detections
        with added prediction and confidence columns
    """
    if "category_label" in manifest.columns:
        otherdf = manifest[manifest["category_label"] != "animal"].reset_index(drop=True)

    else:
        # Convert category column to int and fill NaN with 0 (empty) if necessary
        manifest["category"] = manifest["category"].fillna(0)
        manifest["category"] = manifest["category"].astype(int)
        manifest["category_label"] = manifest["category"].replace(MD_LABELS)
        otherdf = manifest[manifest["category"] != 1].reset_index(drop=True)

    if not otherdf.empty:
        otherdf['prediction'] = otherdf["category_label"]
        otherdf['confidence'] = otherdf['conf'].fillna(1)  # correct empty conf

    return otherdf
