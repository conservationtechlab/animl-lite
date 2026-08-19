from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from animl import pipeline


class FakeWorkingDir:
    def __init__(self, root: Path):
        self.root = root
        self.basedir = root / "Animl-Directory"
        self.basedir.mkdir(exist_ok=True)
        self.linkdir = self.basedir / "Sorted"
        self.visdir = self.basedir / "Plots"
        self.filemanifest = self.basedir / "FileManifest.csv"
        self.imageframes = self.basedir / "ImageFrames.csv"
        self.results = self.basedir / "Results.csv"
        self.predictions = self.basedir / "Predictions.csv"
        self.detections = self.basedir / "Detections.csv"
        self.mdraw = self.basedir / "MD_Raw.json"

    def activate_linkdir(self):
        self.linkdir.mkdir(exist_ok=True)

    def activate_visdir(self):
        self.visdir.mkdir(exist_ok=True)


def _base_frames_df(tmp_path: Path):
    f = tmp_path / "img.jpg"
    f.write_bytes(b"x")
    return pd.DataFrame(
        {
            "filepath": [str(f)],
            "filename": [f.name],
            "extension": [".jpg"],
            "datetime": ["2026-01-01 00:00:00"],
            "station": ["s1"],
            "frame": [0],
        }
    )


def _base_detections_df(frames_df: pd.DataFrame):
    return frames_df.assign(
        max_detection_conf=0.9,
        category=1,
        category_label="animal",
        conf=0.9,
        bbox_x=0.1,
        bbox_y=0.2,
        bbox_w=0.3,
        bbox_h=0.4,
    )


def test_from_paths_detect_only_flow(monkeypatch, tmp_path: Path):
    calls = {"save": 0, "classify": 0}
    wd = FakeWorkingDir(tmp_path)
    frames = _base_frames_df(tmp_path)
    detections = _base_detections_df(frames)

    monkeypatch.setattr(pipeline.file_management, "WorkingDirectory", lambda _p: wd)
    monkeypatch.setattr(pipeline.file_management, "build_file_manifest", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline.video_processing, "extract_frames", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline, "get_onnx_device", lambda **_k: ["CPUExecutionProvider"])
    monkeypatch.setattr(pipeline.file_management, "check_file", lambda *_a, **_k: False)
    monkeypatch.setattr(pipeline.detection, "load_detector", lambda *_a, **_k: object())
    monkeypatch.setattr(pipeline.detection, "detect", lambda *_a, **_k: [{"filepath": frames.iloc[0]["filepath"], "detections": []}])
    monkeypatch.setattr(pipeline.detection, "parse_detections", lambda *_a, **_k: detections)
    monkeypatch.setattr(pipeline.classification, "classify", lambda *_a, **_k: calls.__setitem__("classify", calls["classify"] + 1))
    monkeypatch.setattr(pipeline.file_management, "save_data", lambda *_a, **_k: calls.__setitem__("save", calls["save"] + 1))

    out = pipeline.from_paths("/tmp/images", "det.onnx", "cls.onnx", detect_only=True)
    assert len(out) == 1
    assert calls["classify"] == 0
    assert calls["save"] == 1


def test_from_paths_classification_single(monkeypatch, tmp_path: Path):
    wd = FakeWorkingDir(tmp_path)
    frames = _base_frames_df(tmp_path)
    detections = _base_detections_df(frames)
    animals = detections.copy()
    empty = pd.DataFrame(columns=detections.columns)

    monkeypatch.setattr(pipeline.file_management, "WorkingDirectory", lambda _p: wd)
    monkeypatch.setattr(pipeline.file_management, "build_file_manifest", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline.video_processing, "extract_frames", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline, "get_onnx_device", lambda **_k: ["CPUExecutionProvider"])
    monkeypatch.setattr(pipeline.file_management, "check_file", lambda *_a, **_k: False)
    monkeypatch.setattr(pipeline.detection, "load_detector", lambda *_a, **_k: object())
    monkeypatch.setattr(pipeline.detection, "detect", lambda *_a, **_k: [{"filepath": frames.iloc[0]["filepath"], "detections": []}])
    monkeypatch.setattr(pipeline.detection, "parse_detections", lambda *_a, **_k: detections)
    monkeypatch.setattr(pipeline.detection, "get_animals", lambda *_a, **_k: animals)
    monkeypatch.setattr(pipeline.detection, "get_empty", lambda *_a, **_k: empty)
    monkeypatch.setattr(pipeline.classification, "load_classifier", lambda *_a, **_k: (object(), pd.DataFrame({"class": ["jaguar"]})))
    monkeypatch.setattr(pipeline.classification, "classify", lambda *_a, **_k: np.array([[0.8]]))
    monkeypatch.setattr(
        pipeline.classification,
        "single_classification",
        lambda *_a, **_k: animals.assign(prediction="jaguar", confidence=0.72),
    )
    monkeypatch.setattr(pipeline.file_management, "save_data", lambda *_a, **_k: None)

    out = pipeline.from_paths("/tmp/images", "det.onnx", "cls.onnx", sequence=False)
    assert "prediction" in out.columns
    assert out.loc[0, "prediction"] == "jaguar"


def test_from_paths_classification_sequence(monkeypatch, tmp_path: Path):
    wd = FakeWorkingDir(tmp_path)
    frames = _base_frames_df(tmp_path)
    detections = _base_detections_df(frames)

    monkeypatch.setattr(pipeline.file_management, "WorkingDirectory", lambda _p: wd)
    monkeypatch.setattr(pipeline.file_management, "build_file_manifest", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline.video_processing, "extract_frames", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline, "get_onnx_device", lambda **_k: ["CPUExecutionProvider"])
    monkeypatch.setattr(pipeline.file_management, "check_file", lambda *_a, **_k: False)
    monkeypatch.setattr(pipeline.detection, "load_detector", lambda *_a, **_k: object())
    monkeypatch.setattr(pipeline.detection, "detect", lambda *_a, **_k: [{"filepath": frames.iloc[0]["filepath"], "detections": []}])
    monkeypatch.setattr(pipeline.detection, "parse_detections", lambda *_a, **_k: detections)
    monkeypatch.setattr(pipeline.detection, "get_animals", lambda *_a, **_k: detections)
    monkeypatch.setattr(pipeline.detection, "get_empty", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(pipeline.classification, "load_classifier", lambda *_a, **_k: (object(), pd.DataFrame({"class": ["jaguar"]})))
    monkeypatch.setattr(pipeline.classification, "classify", lambda *_a, **_k: np.array([[0.8]]))
    monkeypatch.setattr(
        pipeline.classification,
        "sequence_classification",
        lambda *_a, **_k: detections.assign(prediction="jaguar", confidence=0.72, sequence=0),
    )
    monkeypatch.setattr(pipeline.file_management, "save_data", lambda *_a, **_k: None)

    out = pipeline.from_paths("/tmp/images", "det.onnx", "cls.onnx", sequence=True)
    assert "sequence" in out.columns


def test_from_config_detect_only_with_sort_and_visualize(monkeypatch, tmp_path: Path):
    wd = FakeWorkingDir(tmp_path)
    frames = _base_frames_df(tmp_path)
    detections = _base_detections_df(frames)
    cfg = {
        "image_dir": "/tmp/images",
        "detector_file": "det.onnx",
        "detect_only": True,
        "sort": True,
        "visualize": True,
        "copy": False,
    }
    calls = {"sorted": 0, "viz": 0}

    monkeypatch.setattr(pipeline.file_management, "load_yaml", lambda *_a, **_k: cfg)
    monkeypatch.setattr(pipeline.file_management, "WorkingDirectory", lambda _p: wd)
    monkeypatch.setattr(pipeline.file_management, "build_file_manifest", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline.video_processing, "extract_frames", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline, "get_onnx_device", lambda **_k: ["CPUExecutionProvider"])
    monkeypatch.setattr(pipeline.file_management, "check_file", lambda *_a, **_k: False)
    monkeypatch.setattr(pipeline.detection, "load_detector", lambda *_a, **_k: object())
    monkeypatch.setattr(pipeline.detection, "detect", lambda *_a, **_k: [{"filepath": frames.iloc[0]["filepath"], "detections": []}])
    monkeypatch.setattr(pipeline.detection, "parse_detections", lambda *_a, **_k: detections)
    monkeypatch.setattr(
        pipeline.export,
        "export_folders",
        lambda manifest, *_a, **_k: calls.__setitem__("sorted", calls["sorted"] + 1) or manifest,
    )
    monkeypatch.setattr(
        pipeline.visualization,
        "plot_all_bounding_boxes",
        lambda *_a, **_k: calls.__setitem__("viz", calls["viz"] + 1),
    )
    monkeypatch.setattr(pipeline.file_management, "save_data", lambda *_a, **_k: None)

    pipeline.from_config("config.yaml")
    assert calls["sorted"] == 1
    assert calls["viz"] == 1


def test_from_config_uses_existing_detections_if_present(monkeypatch, tmp_path: Path):
    wd = FakeWorkingDir(tmp_path)
    frames = _base_frames_df(tmp_path)
    detections = _base_detections_df(frames)
    cfg = {"image_dir": "/tmp/images", "detector_file": "det.onnx", "classifier_file": "cls.onnx"}
    calls = {"load_detector": 0, "load_data": 0}

    monkeypatch.setattr(pipeline.file_management, "load_yaml", lambda *_a, **_k: cfg)
    monkeypatch.setattr(pipeline.file_management, "WorkingDirectory", lambda _p: wd)
    monkeypatch.setattr(pipeline.file_management, "build_file_manifest", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline.video_processing, "extract_frames", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline, "get_onnx_device", lambda **_k: ["CPUExecutionProvider"])
    monkeypatch.setattr(pipeline.file_management, "check_file", lambda *_a, **_k: True)
    monkeypatch.setattr(
        pipeline.file_management,
        "load_data",
        lambda *_a, **_k: calls.__setitem__("load_data", calls["load_data"] + 1) or detections,
    )
    monkeypatch.setattr(
        pipeline.detection,
        "load_detector",
        lambda *_a, **_k: calls.__setitem__("load_detector", calls["load_detector"] + 1) or object(),
    )
    monkeypatch.setattr(pipeline.detection, "get_animals", lambda *_a, **_k: detections)
    monkeypatch.setattr(pipeline.detection, "get_empty", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(pipeline.classification, "load_classifier", lambda *_a, **_k: (object(), pd.DataFrame({"class": ["jaguar"]})))
    monkeypatch.setattr(pipeline.classification, "classify", lambda *_a, **_k: np.array([[0.8]]))
    monkeypatch.setattr(pipeline.classification, "single_classification", lambda *_a, **_k: detections.assign(prediction="jaguar", confidence=0.72))
    monkeypatch.setattr(pipeline.file_management, "save_data", lambda *_a, **_k: None)

    pipeline.from_config("config.yaml")
    assert calls["load_data"] == 1
    assert calls["load_detector"] == 0


def test_from_config_uses_custom_detector_category_map(monkeypatch, tmp_path: Path):
    wd = FakeWorkingDir(tmp_path)
    frames = _base_frames_df(tmp_path)
    detections = _base_detections_df(frames)
    class_list_path = tmp_path / "detector_classes.csv"
    pd.DataFrame({"id": [1], "class": ["animal"]}).to_csv(class_list_path, index=False)
    cfg = {
        "image_dir": "/tmp/images",
        "detector_file": "det.onnx",
        "classifier_file": "cls.onnx",
        "detector_class_list": str(class_list_path),
    }
    detect_args = {}

    monkeypatch.setattr(pipeline.file_management, "load_yaml", lambda *_a, **_k: cfg)
    monkeypatch.setattr(pipeline.file_management, "WorkingDirectory", lambda _p: wd)
    monkeypatch.setattr(pipeline.file_management, "build_file_manifest", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline.video_processing, "extract_frames", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline, "get_onnx_device", lambda **_k: ["CPUExecutionProvider"])
    monkeypatch.setattr(pipeline.file_management, "check_file", lambda *_a, **_k: False)
    monkeypatch.setattr(pipeline.detection, "load_detector", lambda *_a, **_k: object())

    def _detect(_detector, _frames, **kwargs):
        detect_args.update(kwargs)
        return [{"filepath": frames.iloc[0]["filepath"], "detections": []}]

    monkeypatch.setattr(pipeline.detection, "detect", _detect)
    monkeypatch.setattr(pipeline.detection, "parse_detections", lambda *_a, **_k: detections)
    monkeypatch.setattr(pipeline.detection, "get_animals", lambda *_a, **_k: detections)
    monkeypatch.setattr(pipeline.detection, "get_empty", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(pipeline.classification, "load_classifier", lambda *_a, **_k: (object(), pd.DataFrame({"class": ["jaguar"]})))
    monkeypatch.setattr(pipeline.classification, "classify", lambda *_a, **_k: np.array([[0.8]]))
    monkeypatch.setattr(pipeline.classification, "single_classification", lambda *_a, **_k: detections.assign(prediction="jaguar", confidence=0.72))
    monkeypatch.setattr(pipeline.file_management, "save_data", lambda *_a, **_k: None)

    pipeline.from_config("config.yaml")
    assert detect_args["category_map"] == {1: "animal"}


def test_from_config_sequence_path_when_station_present(monkeypatch, tmp_path: Path):
    wd = FakeWorkingDir(tmp_path)
    frames = _base_frames_df(tmp_path)
    detections = _base_detections_df(frames)
    cfg = {
        "image_dir": "/tmp/images",
        "detector_file": "det.onnx",
        "classifier_file": "cls.onnx",
        "sequence": True,
        "empty_class": "empty",
    }
    calls = {"seq": 0, "single": 0}

    monkeypatch.setattr(pipeline.file_management, "load_yaml", lambda *_a, **_k: cfg)
    monkeypatch.setattr(pipeline.file_management, "WorkingDirectory", lambda _p: wd)
    monkeypatch.setattr(pipeline.file_management, "build_file_manifest", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline.video_processing, "extract_frames", lambda *_a, **_k: frames)
    monkeypatch.setattr(pipeline, "get_onnx_device", lambda **_k: ["CPUExecutionProvider"])
    monkeypatch.setattr(pipeline.file_management, "check_file", lambda *_a, **_k: False)
    monkeypatch.setattr(pipeline.detection, "load_detector", lambda *_a, **_k: object())
    monkeypatch.setattr(pipeline.detection, "detect", lambda *_a, **_k: [{"filepath": frames.iloc[0]["filepath"], "detections": []}])
    monkeypatch.setattr(pipeline.detection, "parse_detections", lambda *_a, **_k: detections)
    monkeypatch.setattr(pipeline.detection, "get_animals", lambda *_a, **_k: detections)
    monkeypatch.setattr(pipeline.detection, "get_empty", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(pipeline.classification, "load_classifier", lambda *_a, **_k: (object(), pd.DataFrame({"class": ["jaguar"]})))
    monkeypatch.setattr(pipeline.classification, "classify", lambda *_a, **_k: np.array([[0.8]]))
    monkeypatch.setattr(
        pipeline.classification,
        "sequence_classification",
        lambda *_a, **_k: calls.__setitem__("seq", calls["seq"] + 1) or detections.assign(prediction="jaguar", confidence=0.72),
    )
    monkeypatch.setattr(
        pipeline.classification,
        "single_classification",
        lambda *_a, **_k: calls.__setitem__("single", calls["single"] + 1) or detections.assign(prediction="jaguar", confidence=0.72),
    )
    monkeypatch.setattr(pipeline.file_management, "save_data", lambda *_a, **_k: None)

    pipeline.from_config("config.yaml")
    assert calls["seq"] == 1
    assert calls["single"] == 0
