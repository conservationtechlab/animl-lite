from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
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


def _detect(detector, tmp_path, size=(640, 480)):
    path = tmp_path / "img.jpg"
    Image.new("RGB", size, (90, 90, 90)).save(path)
    manifest = pd.DataFrame({"filepath": [str(path)], "frame": [0]})
    output = detection.detect(detector, manifest, 960, 960, num_workers=0, quiet=True)
    results = output[0] if isinstance(output, tuple) else output
    return results[0]


def test_box_at_the_left_edge_is_kept(tmp_path):
    """Boxes were dropped when any coordinate was 0, i.e. touching the left or top edge"""
    # the 640x480 image fills x 0..960, y 120..840 of the letterboxed input
    detector = FakeDetector([[0, 300, 400, 700, 0.9, 0], [300, 300, 700, 700, 0.8, 0]])
    result = _detect(detector, tmp_path)
    assert [d["conf"] for d in result["detections"]] == pytest.approx([0.9, 0.8])
    assert result["detections"][0]["bbox_x"] == 0


def test_box_at_the_top_edge_is_kept(tmp_path):
    # the 480x640 image fills x 120..840, y 0..960 of the letterboxed input
    detector = FakeDetector([[300, 0, 700, 500, 0.9, 0]])
    result = _detect(detector, tmp_path, size=(480, 640))
    assert len(result["detections"]) == 1
    assert result["detections"][0]["bbox_y"] == 0
