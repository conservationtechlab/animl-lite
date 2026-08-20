"""
Automated Pipeline Functions

@ Kyra Swanson 2023
"""
import pandas as pd

from animl import (classification, detection, export, file_management, video_processing)
from animl.utils import visualization
from animl.utils.general import MEGADETECTORv5_SIZE, SDZWA_CLASSIFIER_SIZE, get_onnx_device


def from_paths(image_dir: str,
               detector_file: str,
               classifier_file: str,
               class_label: str = "class",
               sort: bool = False,
               visualize: bool = False,
               sequence: bool = False,
               detect_only: bool = False) -> pd.DataFrame:
    """
    This function is the main method to invoke all the sub functions
    to create a working directory for the image directory.

    Args:
        image_dir (str): directory path containing the images or videos.
        detector_file (str): file path of the MegaDetector model.
        classifier_file (str): file path of the classifier model.
        class_label: column in the class list that contains the label wanted
        batch_size (int): batch size for inference
        sort (bool): toggle option to create symlinks
        visualize (bool): if True, run visualization
        sequence (bool): if True, run sequence_classification

    Returns:
        pandas.DataFrame: Concatenated dataframe of animal and empty detections
    """
    # Create a working directory, build the file manifest from img_dir
    print("Searching directory...")
    working_dir = file_management.WorkingDirectory(image_dir)
    files = file_management.build_file_manifest(image_dir,
                                                out_file=working_dir.filemanifest,
                                                exif=True)
    print(f"Found {len(files)} files.")

    # Obtain frames from videos
    all_frames = video_processing.extract_frames(files, frames=5, out_file=working_dir.imageframes)

    print("Running images and video frames through detector...")
    device = get_onnx_device(user_set=None)

    if (file_management.check_file(working_dir.detections, output_type="Detections")):
        detections = file_management.load_data(working_dir.detections)
    else:
        detector = detection.load_detector(detector_file, "megadetector", device=device)

        md_results = detection.detect(detector,
                                      all_frames,
                                      resize_height=MEGADETECTORv5_SIZE,
                                      resize_width=MEGADETECTORv5_SIZE,
                                      checkpoint_path=working_dir.mdraw,
                                      checkpoint_frequency=1000)
        # Convert MD JSON to pandas dataframe, merge with manifest
        print("Converting MD JSON to dataframe and merging with manifest...")
        detections = detection.parse_detections(md_results, manifest=all_frames, out_file=working_dir.detections)

    # Detection only option - skip classification and export results directly from detections
    if detect_only:
        print("Detection only flag set, skipping classification.")
        manifest = detections

        # Sort
        if sort:
            print("Sorting...")
            working_dir.activate_linkdir()
            manifest = export.export_folders(manifest, working_dir.linkdir, label_col='category')
        # Plot boxes
        if visualize:
            working_dir.activate_visdir()
            visualization.plot_all_bounding_boxes(manifest, working_dir.visdir,
                                                  file_col='filepath', classifier_label_col=None)

    else:
        # Extract animal detections from the rest
        animals = detection.get_animals(detections)
        empty = detection.get_empty(detections)

        # Use the classifier model to predict the species of animal detections
        print("Predicting species of animal detections...")
        classifier, class_list = classification.load_classifier(classifier_file)
        predictions_output = classification.classify(classifier,
                                                     animals,
                                                     resize_height=SDZWA_CLASSIFIER_SIZE,
                                                     resize_width=SDZWA_CLASSIFIER_SIZE,
                                                     out_file=working_dir.predictions)
        if sequence:
            print("Classifying sequences...")
            manifest = classification.sequence_classification(animals,
                                                              empty,
                                                              predictions_output,
                                                              class_list[class_label],
                                                              station_col='station',
                                                              empty_class="",
                                                              sort_columns=["station", "datetime", "frame"],
                                                              maxdiff=60)
        else:
            print("Classifying individual frames...")
            manifest = classification.single_classification(animals,
                                                            empty,
                                                            predictions_output,
                                                            class_list[class_label],
                                                            best=True)

        if sort:
            print("Sorting...")
            working_dir.activate_linkdir()
            manifest = export.export_folders(manifest, working_dir.linkdir, label_col='prediction')

        # Plot boxes
        if visualize:
            working_dir.activate_visdir()
            visualization.plot_all_bounding_boxes(manifest, working_dir.visdir,
                                                  file_col='filepath', classifier_label_col='prediction')

    # Save final results to csv
    file_management.save_data(manifest, working_dir.results)
    print("Final Results in " + str(working_dir.results))

    return manifest


def from_config(config: str):
    """
    This function is the main method to invoke all the sub functions
    to create a working directory for the image directory.

    Args:
        config (str): path containing config file for inference

    Returns:
        pandas.DataFrame: Concatenated dataframe of animal and empty detections
    """
    print(f'Using config "{config}"')
    cfg = file_management.load_yaml(config)

    # get image dir and cuda defaults
    image_dir = cfg['image_dir']
    device = cfg.get('device', 'cpu')

    print("Searching directory...")
    # Create a working directory, default to image_dir
    working_dir = file_management.WorkingDirectory(cfg.get('working_dir', image_dir))

    files = file_management.build_file_manifest(image_dir,
                                                exif=cfg.get('exif', True),
                                                out_file=working_dir.filemanifest,
                                                data_timezone=cfg.get('data_timezone', None),
                                                station_depth=cfg.get('station_depth', None),
                                                camera_depth=cfg.get('camera_depth', None))
    print(f"Found {len(files)} files.")

    # split out videos
    all_frames = video_processing.extract_frames(files,
                                                 frames=cfg.get('frames', 5),
                                                 fps=cfg.get('fps', None),
                                                 out_file=working_dir.imageframes)

    print("Running images and video frames through detector...")
    device = get_onnx_device(user_set=None)

    if (file_management.check_file(working_dir.detections, output_type="Detections")):
        detections = file_management.load_data(working_dir.detections)
    else:
        detector = detection.load_detector(cfg['detector_file'], model_type=cfg.get('detector_type', 'megadetector'), device=device)
        categories = cfg.get('detector_class_list', None)
        if categories is None:
            category_map = visualization.MD_LABELS
        else:
            categories = file_management.load_data(categories)
            category_map = file_management.class_list_to_dict(categories,
                                                              id_col=cfg.get('detector_class_key_col', 'id'),
                                                              class_col=cfg.get('detector_class_value_col', 'class'))

        md_results = detection.detect(detector,
                                      all_frames,
                                      resize_height=cfg.get('detection_resize_height', MEGADETECTORv5_SIZE),
                                      resize_width=cfg.get('detection_resize_width', MEGADETECTORv5_SIZE),
                                      letterbox=cfg.get('letterbox', True),
                                      category_map=category_map,
                                      file_col=cfg.get('detection_file_col', 'filepath'),
                                      checkpoint_path=working_dir.mdraw,
                                      checkpoint_frequency=cfg.get('checkpoint_frequency', 1000))
        # Convert MD JSON to pandas dataframe, merge with manifest
        print("Converting MD JSON to dataframe and merging with manifest...")
        detections = detection.parse_detections(md_results, manifest=all_frames, out_file=working_dir.detections)

    # Detection only option - skip classification and export results directly from detections
    if cfg.get('detect_only', False):
        print("Detection only flag set, skipping classification.")
        manifest = detections

        # Sort
        if cfg.get('sort', True):
            print("Sorting...")
            working_dir.activate_linkdir()
            manifest = export.export_folders(manifest,
                                             working_dir.linkdir,
                                             label_col='category',
                                             copy=cfg.get('copy', False))
        # Plot boxes
        if cfg.get('visualize', False):
            working_dir.activate_visdir()
            visualization.plot_all_bounding_boxes(manifest,
                                                  working_dir.visdir,
                                                  file_col='filepath',
                                                  classifier_label_col=None)

    else:
        # Extract animal detections from the rest
        animals = detection.get_animals(detections)
        empty = detection.get_empty(detections)

        # Use the classifier model to predict the species of animal detections
        print("Predicting species...")
        # Load classifier
        classifier, class_list = classification.load_classifier(cfg['classifier_file'], cfg.get('class_list', None))

        predictions_output = classification.classify(classifier,
                                                     animals,
                                                     resize_height=cfg.get('classification_resize_height',
                                                                           SDZWA_CLASSIFIER_SIZE),
                                                     resize_width=cfg.get('classification_resize_width',
                                                                          SDZWA_CLASSIFIER_SIZE),
                                                     file_col=cfg.get('classification_file_col', 'filepath'),
                                                     out_file=working_dir.predictions)

        # Convert predictions to labels
        if 'station' in animals.columns and cfg.get('sequence', False):
            manifest = classification.sequence_classification(animals,
                                                              empty,
                                                              predictions_output,
                                                              class_list[cfg.get('class_label_col', 'class')],
                                                              station_col='station',
                                                              empty_class=cfg['empty_class'],
                                                              sort_columns=["station", "datetime", "frame"],
                                                              file_col=cfg.get('classification_file_col', 'frame'),
                                                              maxdiff=60)
        else:
            manifest = classification.single_classification(animals,
                                                            empty,
                                                            predictions_output,
                                                            class_list[cfg.get('class_label_col', 'class')],
                                                            file_col=cfg.get('classification_file_col', 'filepath'),
                                                            best=cfg.get('best_only', True))

        if cfg.get('sort', True):
            print("Sorting...")
            working_dir.activate_linkdir()
            manifest = export.export_folders(manifest, working_dir.linkdir, copy=cfg.get('copy', False))

        # Plot boxes
        if cfg.get('visualize', False):
            working_dir.activate_visdir()
            visualization.plot_all_bounding_boxes(manifest,
                                                  working_dir.visdir,
                                                  file_col='filepath',
                                                  classifier_label_col='prediction')

    file_management.save_data(manifest, working_dir.results)
    print("Final Results in " + str(working_dir.results))

    return manifest
