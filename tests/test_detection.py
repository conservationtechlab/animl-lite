from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from animl import detection


class TestParseDetections:
    def test_no_images(self):
        """An empty list (e.g. every image unreadable) used to raise an AssertionError"""
        df = detection.parse_detections([])
        assert df.empty
        assert {"filepath", "frame", "category_label", "conf", "bbox_x"}.issubset(df.columns)
        assert detection.get_animals(df).empty

    def test_no_images_merged_onto_manifest(self):
        manifest = pd.DataFrame({"filepath": ["a.jpg", "b.jpg"], "frame": [0, 0]})
        df = detection.parse_detections([], manifest=manifest)
        assert df["filepath"].tolist() == ["a.jpg", "b.jpg"]
        assert df["conf"].isna().all()

    def test_score_columns(self):
        assert {"score", "clarity"}.issubset(detection.parse_detections([], score=True).columns)

    def test_detections_unchanged(self):
        frames = [{"filepath": "a.jpg", "frame": 0, "max_detection_conf": 0.9,
                   "detections": [{"category": 1, "category_label": "animal", "conf": 0.9,
                                   "bbox_x": 0.1, "bbox_y": 0.2, "bbox_w": 0.3, "bbox_h": 0.4}]},
                  {"filepath": "b.jpg", "frame": 0, "max_detection_conf": None, "category": 0,
                   "category_label": "empty", "detections": []}]
        df = detection.parse_detections(frames)
        assert df[["filepath", "category_label", "bbox_w"]].values.tolist()[0] == ["a.jpg", "animal", 0.3]
        assert df["category_label"].tolist() == ["animal", "empty"]


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
    kwargs = {"num_workers": 0, "use_progress_bar": False, **kwargs}
    return detection.detect(detector, files, 960, 960, **kwargs)


class TestBoxes:
    def test_box_at_the_left_edge_is_kept(self, tmp_path):
        """Boxes were dropped when any coordinate was 0, i.e. touching the left or top edge"""
        # the 640x480 image fills x 0..960, y 120..840 of the letterboxed input
        detector = FakeDetector([[0, 300, 400, 700, 0.9, 0], [300, 300, 700, 700, 0.8, 0]])
        (result,) = _detect(detector, pd.DataFrame({"filepath": [_image(tmp_path)], "frame": [0]}))
        assert [d["conf"] for d in result["detections"]] == pytest.approx([0.9, 0.8])
        assert result["detections"][0]["bbox_x"] == 0

    def test_empty_box_is_skipped(self, tmp_path):
        detector = FakeDetector([[300, 300, 300, 700, 0.9, 0], [300, 300, 700, 700, 0.8, 0]])
        (result,) = _detect(detector, pd.DataFrame({"filepath": [_image(tmp_path)], "frame": [0]}))
        assert [d["conf"] for d in result["detections"]] == pytest.approx([0.8])


class TestInputs:
    BOX = [[300, 300, 700, 700, 0.8, 0]]

    def test_single_path(self, tmp_path):
        """ManifestGenerator isn't subscriptable; a single path raised a TypeError"""
        path = _image(tmp_path)
        (result,) = _detect(FakeDetector(self.BOX), path)
        assert result["filepath"] == path
        assert len(result["detections"]) == 1

    def test_custom_file_col(self, tmp_path):
        path = _image(tmp_path)
        (result,) = _detect(FakeDetector(self.BOX), pd.DataFrame({"path": [path], "frame": [0]}), file_col="path")
        assert result["filepath"] == path

    def test_checkpoint_resumes_a_list(self, tmp_path, monkeypatch):
        """Resuming filtered the input with DataFrame operations, which failed for a list"""
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")   # load the checkpoint
        first, second = _image(tmp_path, "a.jpg"), _image(tmp_path, "b.jpg")
        checkpoint = str(tmp_path / "checkpoint.json")
        _detect(FakeDetector(self.BOX), [first], checkpoint_path=checkpoint)
        detector = FakeDetector(self.BOX)
        calls = []
        run = detector.run
        detector.run = lambda names, feed: calls.append(len(feed["images"])) or run(names, feed)
        results = _detect(detector, [first, second], checkpoint_path=checkpoint)
        assert sorted(r["filepath"] for r in results) == [first, second]
        assert calls == [1]   # only the new image is detected
