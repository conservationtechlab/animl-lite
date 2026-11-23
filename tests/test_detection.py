from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from animl import detection


class FakeDetector:
    """ONNX-like MegaDetector returning fixed boxes in letterboxed 960x960 pixels"""
    model_type = "megadetector"

    def __init__(self, boxes):
        self.boxes = np.array(boxes, dtype=np.float32)

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def run(self, _names, feed):
        return [np.repeat(self.boxes[np.newaxis], feed["images"].shape[0], axis=0)]


def _image(tmp_path, name="img.jpg", size=(640, 480)):
    path = tmp_path / name
    Image.new("RGB", size, (90, 90, 90)).save(path)
    return str(path)


def _detect(detector, files, **kwargs):
    kwargs = {"num_workers": 0, "quiet": True, **kwargs}
    return detection.detect(detector, files, 960, 960, **kwargs)


class TestInputs:
    BOX = [[300, 300, 700, 700, 0.8, 0]]

    def test_single_path(self, tmp_path):
        """ManifestGenerator isn't subscriptable; a single path raised a TypeError"""
        path = _image(tmp_path)
        (result,) = _detect(FakeDetector(self.BOX), path)
        assert result["filepath"] == path
        assert len(result["detections"]) == 1

    def test_list_of_paths(self, tmp_path):
        paths = [_image(tmp_path, "a.jpg"), _image(tmp_path, "b.jpg")]
        results = _detect(FakeDetector(self.BOX), paths)
        assert [r["filepath"] for r in results] == paths

    def test_custom_file_col(self, tmp_path):
        """file_col wasn't passed to ManifestGenerator, which raised for any column but 'filepath'"""
        path = _image(tmp_path)
        (result,) = _detect(FakeDetector(self.BOX), pd.DataFrame({"path": [path], "frame": [0]}), file_col="path")
        assert result["filepath"] == path
