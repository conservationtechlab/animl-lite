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
        (result,), _ = _detect(FakeDetector(self.BOX), path)
        assert result["filepath"] == path
        assert len(result["detections"]) == 1

    def test_list_of_paths(self, tmp_path):
        paths = [_image(tmp_path, "a.jpg"), _image(tmp_path, "b.jpg")]
        results, _ = _detect(FakeDetector(self.BOX), paths)
        assert [r["filepath"] for r in results] == paths

    def test_custom_file_col(self, tmp_path):
        """file_col wasn't passed to ManifestGenerator, which raised for any column but 'filepath'"""
        path = _image(tmp_path)
        (result,), _ = _detect(FakeDetector(self.BOX), pd.DataFrame({"path": [path], "frame": [0]}), file_col="path")
        assert result["filepath"] == path


class TestFailedFiles:
    BOX = [[300, 300, 700, 700, 0.8, 0]]

    def test_detect_returns_failed_files(self, tmp_path):
        path = _image(tmp_path)
        missing = str(tmp_path / "missing.jpg")
        results, failed_files = detection.detect(FakeDetector(self.BOX), [path, missing], 960, 960,
                                                 num_workers=0, quiet=True)
        assert [r["filepath"] for r in results] == [path]
        assert failed_files == [missing]

    def test_parse_detections_accepts_detect_output_and_saves_failed_files(self, tmp_path):
        path = _image(tmp_path)
        missing = str(tmp_path / "missing.jpg")
        out_file = tmp_path / "detections.csv"
        output = detection.detect(FakeDetector(self.BOX), [path, missing], 960, 960, num_workers=0, quiet=True)
        df = detection.parse_detections(output, out_file=str(out_file))
        assert df["filepath"].tolist() == [path]
        assert (tmp_path / "detection_failed_files.txt").read_text().splitlines() == [missing]

    def test_parse_detections_accepts_checkpoint_dict(self, tmp_path):
        path = _image(tmp_path)
        results, _ = detection.detect(FakeDetector(self.BOX), path, 960, 960, num_workers=0, quiet=True)
        assert detection.parse_detections({"images": results})["filepath"].tolist() == [path]

    def test_resume_accepts_detections_key(self, tmp_path, monkeypatch):
        import json
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")  # load the checkpoint
        path = _image(tmp_path)
        checkpoint = tmp_path / "checkpoint.json"
        results, _ = detection.detect(FakeDetector(self.BOX), path, 960, 960, num_workers=0, quiet=True)
        checkpoint.write_text(json.dumps({"detections": results}))
        resumed, failed_files = detection.detect(FakeDetector(self.BOX), [path], 960, 960, num_workers=0,
                                                 quiet=True, checkpoint_path=str(checkpoint))
        assert [r["filepath"] for r in resumed] == [path]
        assert failed_files == []


class TestParseDetectionsEmpty:
    def test_no_images(self):
        """An empty list (e.g. every image unreadable) used to raise an AssertionError"""
        df = detection.parse_detections([])
        assert df.empty
        assert {"filepath", "frame", "category_label", "conf", "bbox_x"}.issubset(df.columns)
        assert detection.get_animals(df).empty
        assert detection.get_empty(df).empty

    def test_no_images_merged_onto_manifest(self):
        manifest = pd.DataFrame({"filepath": ["a.jpg", "b.jpg"], "frame": [0, 0]})
        df = detection.parse_detections(([], ["a.jpg", "b.jpg"]), manifest=manifest)
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
