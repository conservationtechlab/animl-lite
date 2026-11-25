"""
Object Detection Module

Functions for loading MegaDetector, as well as custom YOLO models
parse_detections() converts json output into a dataframe

"""
from shutil import copyfile
from typing import Optional, Union
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from animl import file_management
from animl.generator import ManifestGenerator
from animl.utils.general import (_normalize_boxes, _xyxy2xywh, _scale_letterbox, get_onnx_device,
                                 _crop_clarity, _frame_score, _normalize_aspect_mode,
                                 _coerce_weights, ScoreWeights, non_max_suppression,
                                 _laplacian_variance, create_onnx_session)
from animl.utils.visualization import MD_LABELS


def load_detector(model_path: str, 
                  model_type: str = "megadetector_v1000",
                  device: Optional[str] = None,
                  quiet: bool = True):
    """
    Load Detector model from filepath.

    Args:
        model_path (str): path to model file
        model_type (str): type of model expected ["megadetector_v1000", "megadetector_v5", "yolo", "miewid"]
        device (str): specify to run on cpu or gpu
        quiet (bool): if True, suppresses output messages

    Returns:
        object: loaded model object
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    providers = get_onnx_device(user_set=device, quiet=quiet)
    model = create_onnx_session(model_path, providers)
    model.model_type = model_type.lower()
    return model


def detect(detector,
           image_file_names,
           resize_width: int,
           resize_height: int,
           file_col: str = 'filepath',
           letterbox: bool = True,
           category_map: Optional[dict] = MD_LABELS,
           confidence_threshold: float = 0.1,
           calculate_clarity: bool = False,
           aspect_mode: Optional[str] = None,
           score_weights: Optional[ScoreWeights] = None,
           batch_size: int = 1,
           prefetch_size: int = 2,
           num_workers: int = 2,
           quiet: bool = True,
           checkpoint_path: Optional[str] = None,
           checkpoint_frequency: int = -1,) -> tuple[list[dict], list]:
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
        calculate_clarity (bool): if True, calculate the clarity of each detected animal (Laplacian variance of its
                                    bounding-box crop) and a frame score used to pick the best frames of a video
        aspect_mode (str): aspect-ratio preference in the frame score. Defaults to None so detection stays
                                    species-neutral; callers that know the target species can set 'wide'
                                    (animals wider than tall) or 'tall' (animals taller than wide)
        score_weights (ScoreWeights): weights for the frame score (a dict with keys conf, area, aspect and
                                    clarity is also accepted; defaults to animl.utils.general.DEFAULT_SCORE_WEIGHTS)
        file_col (str): column name containing file paths
        batch_size (int): number of images per batch
        prefetch_size (int): number of batches to prefetch
        num_workers (int): number of worker threads for data loading
        quiet (bool): if True, suppress the progress bar
        device (str): specify to run on cpu or gpu
        checkpoint_path (str): path to checkpoint file
        checkpoint_frequency (int): write results to checkpoint file every N images

    Returns:
        tuple of list of dicts, each dict represents detections on one image, and list of files that failed to load
    """
    # convert map keys to int if they are string (ie from reticulate)
    category_map = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in category_map.items()}

    # Single image filepath
    if isinstance(image_file_names, str):
        manifest = pd.DataFrame([[image_file_names, 0]], columns=[file_col, 'frame'])

    # list of image filepaths
    elif isinstance(image_file_names, list):
        # create a data frame from list of image paths
        manifest = pd.DataFrame(image_file_names, columns=[file_col])
        # no frame column, assume all images and set to 0
        manifest['frame'] = 0

    # full manifest, select file_col
    elif isinstance(image_file_names, pd.DataFrame):
        if file_col not in image_file_names.columns:
            raise ValueError(f"file_col {file_col} not found in manifest columns")
        # no frame column, assume all images and set to 0
        if 'frame' not in image_file_names.columns:
            print("Warning: 'frame' column not found in manifest columns. Defaulting to 0 assuming images.")
            image_file_names['frame'] = 0
        # create a list of image paths
        manifest = image_file_names[[file_col, 'frame']]

    # single row pd.Series, select file_col
    elif isinstance(image_file_names, pd.Series):
        if file_col not in image_file_names.index:
            raise ValueError(f"file_col {file_col} not found in Series index")
        if 'frame' not in image_file_names.index:
            print("Warning: 'frame' column not found in Series index. Defaulting to 0 assuming images.")
            image_file_names['frame'] = 0
        # create a data frame from image_file_names
        manifest = pd.DataFrame(image_file_names).T
    # column from pd.DataFrame, expected input
    else:
        raise ValueError('image_file_names is not a recognized object')

    # load checkpoint
    if checkpoint_path and file_management.check_file(checkpoint_path, output_type="Megadetector raw output"):
        checkpoint = file_management.load_json(checkpoint_path)
        # checkpoints currently use 'images'; accept 'detections' for forward compatibility
        results = checkpoint.get('detections', checkpoint.get('images', []))
        already_processed = set([r['filepath'] for r in results])
        manifest = manifest[~manifest[file_col].isin(already_processed)][[file_col, 'frame']].reset_index(drop=True)
        if manifest.empty:
            print("All images have already been processed. Exiting.")
            return results, []
    else:
        results = []

    count = 0

    # create dataloader
    dataloader = ManifestGenerator(manifest,
                                   resize_width=resize_width,
                                   resize_height=resize_height,
                                   file_col=file_col,
                                   crop=False,
                                   normalize=True,
                                   letterbox=letterbox,
                                   batch_size=batch_size,
                                   prefetch_size=prefetch_size,
                                   num_workers=num_workers,
                                   quiet=quiet)

    start_time = time.time()
    failed_files = []
    for batch, failed in dataloader:
        count += 1
        # generator reports manifest rows; detect reports their files
        failed_files.extend(dataloader.manifest.iloc[failed][file_col].tolist())

        # handle bad batches (eg. empty images, corrupted files)
        if batch is None:
            continue

        # ONNX Runtime inference
        input_name = detector.get_inputs()[0].name
        outputs = detector.run(None, {input_name: batch[0]})[0]
        outputs = _convert_detections(outputs,
                                      batch,
                                      letterbox,
                                      confidence_threshold=confidence_threshold,
                                      category_map=category_map,
                                      model_type=detector.model_type,
                                      calculate_clarity=calculate_clarity,
                                      aspect_mode=aspect_mode,
                                      score_weights=score_weights)
        # Process outputs to match expected format
        results.extend(outputs)

        # Write a checkpoint if necessary
        if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
            print('Writing a new checkpoint after having processed {} images since last restart'.format(count))
            _save_detection_checkpoint(checkpoint_path, results)

    if not quiet:
        print(f"\nFinished detection. Total images processed: {len(results)} at {round(len(results)/(time.time() - start_time), 1)} img/s.")
    if checkpoint_path:
        _save_detection_checkpoint(checkpoint_path, results)

    return results, failed_files


def _convert_detections(predictions: list,
                        batch_from_dataloader: list,
                        letterbox: bool,
                        confidence_threshold: float = 0.1,
                        model_type: str = 'megadetector_v1000',
                        category_map: dict = MD_LABELS,
                        calculate_clarity=False,
                        aspect_mode: Optional[str] = None,
                        score_weights: Optional[ScoreWeights] = None) -> pd.DataFrame:
    # Converts output into nested list with categories, conf, and bboxes in expected format for parsing function.
    # Supports YOLOv5/MDv5, YOLOv6+, and ONNX models with either relative or absolute bounding box outputs.
    # If letterbox=True, rescales bboxes back to original image size.

    MD_MODELS = ["megadetector", "megadetector_v1000", "megadetector_v5", "megadetector_v5a", "megadetector_v5b"]

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

    aspect_mode = _normalize_aspect_mode(aspect_mode)
    score_weights = _coerce_weights(score_weights)

    results = []

    # Run non-max suppression for MegaDetector v5 models to filter overlapping detections
    if model_type in ["megadetector_v5", "megadetector_v5a", "megadetector_v5b"]:
        predictions = non_max_suppression(predictions, conf_thres=confidence_threshold, iou_thres=0.45)

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
                    'category': 0 if model_type in MD_MODELS else None,
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

                tensor_bbox = bbox.copy()  # box in the model-input image, used to crop for clarity

                if letterbox:
                    bbox = _scale_letterbox(bbox, image_tensors[i].shape[1:], image_sizes[i, :])

                # increase md categories by 1
                if model_type in MD_MODELS:
                    category[j] += 1

                # build detection dict
                detection = {'category': int(category[j]),
                             'category_label': category_map.get(int(category[j]), "unknown"),
                             'conf': float(round(conf[j], 4)),
                             'bbox_x': float(round(bbox[0], 4)),
                             'bbox_y': float(round(bbox[1], 4)),
                             'bbox_w': float(round(bbox[2], 4)),
                             'bbox_h': float(round(bbox[3], 4))}

                if calculate_clarity:
                    # clarity of the animal itself (bounding-box crop), not the whole frame
                    clarity = _crop_clarity(image_tensors[i].transpose(1, 2, 0), tensor_bbox)
                    detection['clarity'] = clarity
                    # image_sizes[i] is the original (height, width); needed to measure aspect in pixels
                    img_h, img_w = int(image_sizes[i][0]), int(image_sizes[i][1])
                    detection['score'] = _frame_score(detection['conf'], detection['bbox_w'], detection['bbox_h'],
                                                      clarity, img_w, img_h,
                                                      aspect_mode=aspect_mode, weights=score_weights)

                detections.append(detection)

            data = {'filepath': str(image_paths[i]),
                    'frame': int(image_frames[i]),
                    'max_detection_conf': max_detection_conf,
                    'detections': detections}
            results.append(data)

    return results


def parse_detections(results: Union[list, tuple],
                     manifest: Optional[pd.DataFrame] = None,
                     out_file: Optional[str] = None,
                     threshold: float = 0,
                     file_col: str = "filepath",
                     score: bool = False,
                     quiet: bool = True) -> pd.DataFrame:
    """
    Converts listed output from detector to DataFrame.

    Args:
        results (Union[list, tuple]): md output dicts or tuple of (md output dicts, failed files)
        manifest (pd.DataFrame): full file manifest, if not None, merge md predictions automatically
        out_file (str): path to save dataframe
        threshold (float): parse only detections above given confidence threshold
        file_col (str): if manifest, merge results onto file_col
        score (bool): if True, calculate a score for each detection based on confidence, bbox size, and clarity

    Returns:
        df (pd.DataFrame): formatted md outputs, one row per detection
    """
    if manifest is not None and file_col not in manifest.columns:
        raise ValueError(f"file_col '{file_col}' not found in manifest columns")

    if manifest is not None and 'frame' not in manifest.columns:
        print("""Warning: 'frame' column not found in manifest columns. Defaulting to 0 for all rows.""")
        manifest['frame'] = 0

    # unpack results
    if isinstance(results, (tuple, list)) and len(results) == 2 and isinstance(results[0], list):
        detections, failed_files = results
    elif isinstance(results, dict):
        # detection checkpoint files use the 'images' key
        detections = results.get('detections', results.get('images', []))
        failed_files = results.get('failed_files', [])
    else:
        detections, failed_files = results, []

    if len(failed_files) > 0:
        print(f"Warning: {len(failed_files)} files failed to load during detection and will be excluded from results.")
        if out_file is not None:
            with (Path(out_file).parent / "detection_failed_files.txt").open("w") as f:
                for item in failed_files:
                    f.write(f"{item}\n")

    # check results format
    if not isinstance(detections, list):
        raise TypeError("MD results input must be list")

    # load results from file if they have already been parsed
    if file_management.check_file(out_file, output_type="Detections"):
        return file_management.load_data(out_file)

    lst = []
    for frame in tqdm(detections, disable=quiet):
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
            if score:
                data['score'] = None
                data['clarity'] = None
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

                    if score:
                        data['score'] = detection.get('score', None)
                        data['clarity'] = detection.get('clarity', None)
                    lst.append(data)

    # columns are set so an empty result (no images, e.g. all unreadable) can still be merged and filtered
    columns = ['filepath', 'frame', 'max_detection_conf', 'category', 'category_label',
               'conf', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'] + (['score', 'clarity'] if score else [])
    df = pd.DataFrame(lst, columns=columns)

    if manifest is not None:
        if file_col in manifest.columns:
            df = manifest.merge(df, left_on=[file_col, 'frame'], right_on=["filepath", "frame"], how='left')
        else:
            raise ValueError("Please provide a manifest with a valid file_col to merge results onto.")

    if out_file:
        file_management.save_data(df, out_file)

    return df


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
    # TODO(next major): write under 'detections'; 'images' kept so older versions can resume
    file_management.save_json({'images': results}, checkpoint_path, prompt=False)

    # Remove the backup checkpoint if it exists
    if checkpoint_tmp_path is not None:
        Path(checkpoint_tmp_path).unlink()
