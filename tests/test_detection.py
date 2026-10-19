from __future__ import annotations

import pandas as pd

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
