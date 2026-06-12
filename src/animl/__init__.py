__version__ = '3.3.0'

from animl import classification
from animl import detection
from animl import export
from animl import file_management
from animl import generator
from animl import pipeline
from animl import reid
from animl import utils
from animl import video_processing

from animl.classification import (classify, load_class_list, load_classifier,
                                  sequence_classification,
                                  single_classification,)
from animl.detection import (detect, get_animals, get_empty, load_detector,
                             parse_detections,)
from animl.export import (export_camptrapdp, export_camtrapR, export_coco,
                          export_folders, export_megadetector,
                          export_timelapse, export_yolo, remove_link,
                          update_labels_from_folders,)
from animl.file_management import (IMAGE_EXTENSIONS, VALID_EXTENSIONS,
                                   VIDEO_EXTENSIONS, WorkingDirectory,
                                   active_times, build_file_manifest,
                                   check_file, class_list_to_dict, load_data,
                                   load_json, load_yaml, save_data, save_json,
                                   save_yaml, sequence_calculation,)
from animl.generator import (ManifestGenerator, manifest_dataloader,)
from animl.pipeline import (from_config, from_paths,)
from animl.reid import (compute_batched_distance_matrix,
                        compute_distance_matrix, cosine_distance, distance,
                        euclidean_squared_distance, extract_miew_embeddings,
                        inference, load_miew, remove_diagonal,)
from animl.utils import (MD_COLORS, MD_LABELS, MEGADETECTORv5_SIZE,
                         MODEL_TYPES, SDZWA_CLASSIFIER_SIZE, animlr,
                         check_exiftool, check_onnx_cuda, general,
                         get_onnx_device, get_version, plot_all_bounding_boxes,
                         plot_box, plot_from_file, visualization,)
from animl.video_processing import (extract_frames, get_frame_as_image,)

__all__ = ['IMAGE_EXTENSIONS', 'MD_COLORS', 'MD_LABELS', 'MEGADETECTORv5_SIZE',
           'MODEL_TYPES', 'ManifestGenerator', 'SDZWA_CLASSIFIER_SIZE',
           'VALID_EXTENSIONS', 'VIDEO_EXTENSIONS', 'WorkingDirectory',
           'active_times', 'animlr', 'build_file_manifest', 'check_exiftool',
           'check_file', 'check_onnx_cuda', 'class_list_to_dict',
           'classification', 'classify', 'compute_batched_distance_matrix',
           'compute_distance_matrix', 'cosine_distance', 'detect', 'detection',
           'distance', 'euclidean_squared_distance', 'export',
           'export_camptrapdp', 'export_camtrapR', 'export_coco',
           'export_folders', 'export_megadetector', 'export_timelapse',
           'export_yolo', 'extract_frames', 'extract_miew_embeddings',
           'file_management', 'from_config', 'from_paths', 'general',
           'generator', 'get_animals', 'get_empty', 'get_frame_as_image',
           'get_onnx_device', 'get_version', 'inference', 'load_class_list',
           'load_classifier', 'load_data', 'load_detector', 'load_json',
           'load_miew', 'load_yaml', 'manifest_dataloader', 'parse_detections',
           'pipeline', 'plot_all_bounding_boxes', 'plot_box', 'plot_from_file',
           'reid', 'remove_diagonal', 'remove_link', 'save_data', 'save_json',
           'save_yaml', 'sequence_calculation', 'sequence_classification',
           'single_classification', 'update_labels_from_folders', 'utils',
           'video_processing', 'visualization']