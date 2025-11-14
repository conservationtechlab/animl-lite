__version__ = '3.1.0'
from animl import classification
from animl import detection
from animl import export
from animl import file_management
from animl import generator
from animl import pipeline
from animl import reid
from animl import split
from animl import utils
from animl import video_processing

from animl.classification import (SDZWA_CLASSIFIER_SIZE, classify,
                                  load_class_list, load_classifier,
                                  sequence_classification,
                                  single_classification,)
from animl.detection import (MEGADETECTORv5_SIZE, convert_onnx_detections,
                             detect, load_detector, parse_detections,)
from animl.export import (export_coco, export_folders, export_megadetector,
                          export_timelapse, remove_link,
                          update_labels_from_folders,)
from animl.file_management import (IMAGE_EXTENSIONS, VALID_EXTENSIONS,
                                   VIDEO_EXTENSIONS, WorkingDirectory,
                                   active_times, build_file_manifest,
                                   check_file, load_data, load_json, save_data,
                                   save_detection_checkpoint, save_json,)
from animl.generator import (Letterbox, ManifestGenerator, Normalize,
                             manifest_dataloader, pil_to_numpy_array,)
from animl.pipeline import (from_config, from_paths,)
from animl.reid import (compute_batched_distance_matrix,
                        compute_distance_matrix, cosine_distance, distance,
                        euclidean_squared_distance, extract_miew_embeddings,
                        inference, load_miew, remove_diagonal,)
from animl.split import (get_animals, get_empty, train_val_test,)
from animl.utils import (MD_COLORS, MD_LABELS, clip_coords,
                         convert_minxywh_to_absxyxy, general, get_device,
                         letterbox, normalize_boxes, plot_all_bounding_boxes,
                         plot_box, plot_from_file, scale_letterbox, softmax,
                         tensor_to_onnx, visualization, xyn2xy, xywh2xyxy,
                         xywhc2xyxy, xywhn2xyxy, xyxy2xywh, xyxyc2xywh,
                         xyxyc2xywhn,)
from animl.video_processing import (count_frames, extract_frames,
                                    get_frame_as_image,)

__all__ = ['IMAGE_EXTENSIONS', 'Letterbox', 'MD_COLORS', 'MD_LABELS',
           'MEGADETECTORv5_SIZE', 'ManifestGenerator', 'Normalize',
           'SDZWA_CLASSIFIER_SIZE', 'VALID_EXTENSIONS', 'VIDEO_EXTENSIONS',
           'WorkingDirectory', 'active_times', 'build_file_manifest',
           'check_file', 'classification', 'classify', 'clip_coords',
           'compute_batched_distance_matrix', 'compute_distance_matrix',
           'convert_minxywh_to_absxyxy', 'convert_onnx_detections',
           'cosine_distance', 'count_frames', 'detect', 'detection',
           'distance', 'euclidean_squared_distance', 'export', 'export_coco',
           'export_folders', 'export_megadetector', 'export_timelapse',
           'extract_frames', 'extract_miew_embeddings', 'file_management',
           'from_config', 'from_paths', 'general', 'generator', 'get_animals',
           'get_device', 'get_empty', 'get_frame_as_image', 'inference',
           'letterbox', 'load_class_list', 'load_classifier', 'load_data',
           'load_detector', 'load_json', 'load_miew', 'manifest_dataloader',
           'normalize_boxes', 'parse_detections', 'pil_to_numpy_array',
           'pipeline', 'plot_all_bounding_boxes', 'plot_box', 'plot_from_file',
           'reid', 'remove_diagonal', 'remove_link', 'save_data',
           'save_detection_checkpoint', 'save_json', 'scale_letterbox',
           'sequence_classification', 'single_classification', 'softmax',
           'split', 'tensor_to_onnx', 'train_val_test',
           'update_labels_from_folders', 'utils', 'video_processing',
           'visualization', 'xyn2xy', 'xywh2xyxy', 'xywhc2xyxy', 'xywhn2xyxy',
           'xyxy2xywh', 'xyxyc2xywh', 'xyxyc2xywhn']
