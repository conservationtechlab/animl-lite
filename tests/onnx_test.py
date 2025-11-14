"""
Test pipeline using ONNX models.

"""
import unittest
import time
import shutil
from pathlib import Path
import yaml

import animl


@unittest.skip
def onnx_test():
    start_time = time.time()

    workingdir = Path.cwd() / 'examples' / 'Southwest' / 'Animl-Directory'
    shutil.rmtree(workingdir, ignore_errors=True)

    config = Path.cwd() / 'examples' / 'animl.yml'

    results = animl.from_config(config)

    # export_coco
    animl.export_timelapse(results, out_dir=workingdir, only_animl=True)

    # export.remove_link
    # export.update_labels_from_folders
    print(f"Test completed in {time.time() - start_time:.2f} seconds")


@unittest.skip
def onnx_gpu_test():
    # test onnx model on gpu
    start_time = time.time()

    workingdir = animl.WorkingDirectory(Path.cwd() / 'examples' / 'Southwest')

    config = Path.cwd() / 'examples' / 'animl.yml'
    cfg = yaml.safe_load(open(config, 'r'))

    allframes = animl.load_data(workingdir.imageframes)

    model_cpu = animl.load_detector(cfg['detector_file'])

    results = animl.detect_test(model_cpu, allframes, resize_width=960, resize_height=960)
    detections = animl.parse_detections(results, manifest=allframes)

    print(detections)

    print(f"ONNX GPU Test completed in {time.time() - start_time:.2f} seconds")


#onnx_test()
onnx_gpu_test()
