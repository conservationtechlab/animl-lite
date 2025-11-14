'''
    Main script.

    Runs full animl workflow on a given directory.
    User must provide MegaDetector, Classifier, and Class list files,
    otherwise will pull MDv5 and the CTL Southwest v2 models by default.

    Usage example
    > python -m animl /home/usr/animl-py/examples/Southwest/

    OR

    > python -m animl /image/dir megadetector.pt classifier.h5 class_file.csv

    Paths to model files must be edited to local machine.

    @ Kyra Swanson, 2023
'''
import unittest
import time
import shutil
from pathlib import Path

import animl


@unittest.skip
def main_test():
    start_time = time.time()

    image_dir = Path.cwd() / 'examples' / 'Southwest'
    workingdir = Path.cwd() / 'examples' / 'Southwest' / 'Animl-Directory'
    shutil.rmtree(workingdir, ignore_errors=True)

    megadetector = Path.cwd() / 'models/md_v1000.0.0-sorrel.onnx'
    classifier_file = Path.cwd() / 'models/sdzwa_southwest_v3.onnx'

    animl.from_paths(image_dir, megadetector, classifier_file,
                     sort=True, visualize=True, sequence=False)

    print(f"Pipeline took {time.time() - start_time:.2f} seconds")


main_test()
