from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def _install_optional_stubs() -> None:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        ort = types.ModuleType("onnxruntime")

        class _FakeSession:
            def __init__(self, *args, **kwargs):
                self.model_type = "mock"

            def get_inputs(self):
                return [types.SimpleNamespace(name="input")]

            def run(self, *_args, **_kwargs):
                return [np.zeros((1, 1), dtype=np.float32)]

            def get_modelmeta(self):
                return types.SimpleNamespace(custom_metadata_map={})

        ort.InferenceSession = _FakeSession
        ort.get_available_providers = lambda: ["CPUExecutionProvider"]
        sys.modules["onnxruntime"] = ort

    try:
        import exiftool  # noqa: F401
    except Exception:
        exiftool = types.ModuleType("exiftool")

        class _FakeExifToolHelper:
            version = "0.0"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get_metadata(self, _filepath):
                return [{}]

        exiftool.ExifToolHelper = _FakeExifToolHelper
        sys.modules["exiftool"] = exiftool


_install_optional_stubs()


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    (root / "station_a" / "cam_1").mkdir(parents=True)
    (root / "station_b" / "cam_2").mkdir(parents=True)

    for idx, rel in enumerate([
        Path("station_a/cam_1/a.jpg"),
        Path("station_a/cam_1/b.png"),
        Path("station_b/cam_2/c.jpeg"),
    ]):
        out = root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32 + idx, 24 + idx), color=(idx * 10, 20, 30)).save(out)

    # Supported extension placeholder; exif=False tests do not open it.
    (root / "station_b" / "cam_2" / "clip.mp4").write_bytes(b"fake-video")
    # Unsupported file
    (root / "station_b" / "cam_2" / "notes.txt").write_text("ignore me")
    return root


@pytest.fixture
def required_results_manifest(tmp_path: Path) -> pd.DataFrame:
    img = tmp_path / "img.jpg"
    Image.new("RGB", (100, 50), color=(255, 255, 255)).save(img)
    return pd.DataFrame(
        [
            {
                "filepath": str(img),
                "filename": img.name,
                "filemodifydate": "2026-01-01 00:00:00",
                "frame": 0,
                "max_detection_conf": 0.9,
                "category": 1,
                "category_label": "animal",
                "conf": 0.9,
                "bbox_x": 0.1,
                "bbox_y": 0.2,
                "bbox_w": 0.3,
                "bbox_h": 0.4,
                "prediction": "jaguar",
                "confidence": 0.8,
                "width": 100,
                "height": 50,
                "datetime": "2026-01-01 00:00:00",
                "station": "station_a",
                "extension": ".jpg",
            }
        ]
    )
