from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from animl.detection import (
    _convert_detections,
    get_animals,
    get_empty,
    load_detector,
    parse_detections,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_detections_with_animal() -> list[dict]:
    return [
        {
            "filepath": "img_a.jpg",
            "frame": 0,
            "max_detection_conf": 0.9,
            "category": 1,
            "category_label": "animal",
            "detections": [
                {
                    "category": 1,
                    "category_label": "animal",
                    "conf": 0.9,
                    "bbox_x": 0.1,
                    "bbox_y": 0.2,
                    "bbox_w": 0.3,
                    "bbox_h": 0.4,
                }
            ],
        }
    ]


@pytest.fixture
def sample_detections_empty() -> list[dict]:
    return [
        {
            "filepath": "img_b.jpg",
            "frame": 0,
            "max_detection_conf": None,
            "category": 0,
            "category_label": "empty",
            "detections": [],
        }
    ]


@pytest.fixture
def mixed_detections(
    sample_detections_with_animal, sample_detections_empty
) -> list[dict]:
    return sample_detections_with_animal + sample_detections_empty


@pytest.fixture
def real_detector_model() -> Path:
    """
    Use an existing real ONNX model file for testing.
    Point this to your actual model location.
    """
    # Option 1: Copy from a known location in your repo/test data
    real_model_path = Path(__file__).parent / "fixtures" / "md_v1000.0.0-sorrel.onnx"
    
    if real_model_path.exists():
        return real_model_path

    pytest.skip("Real detector model not found")




# ---------------------------------------------------------------------------
# load_detector
# ---------------------------------------------------------------------------

def test_load_detector_raises_for_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_detector(str(tmp_path / "nonexistent.onnx"))


def test_load_detector_returns_model_with_type(real_detector_model: Path):
    model = load_detector(str(real_detector_model), model_type="megadetector")
    assert model.model_type == "megadetector"


def test_load_detector_sets_custom_model_type(real_detector_model: Path):
    model = load_detector(str(real_detector_model), model_type="yolo")
    assert model.model_type == "yolo"


# ---------------------------------------------------------------------------
# parse_detections
# ---------------------------------------------------------------------------

def test_parse_detections_returns_dataframe(sample_detections_with_animal):
    df = parse_detections(sample_detections_with_animal)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_parse_detections_contains_expected_columns(sample_detections_with_animal):
    df = parse_detections(sample_detections_with_animal)
    for col in ("filepath", "frame", "category", "conf", "bbox_x", "bbox_y", "bbox_w", "bbox_h"):
        assert col in df.columns, f"Missing column: {col}"


def test_parse_detections_empty_detections(sample_detections_empty):
    df = parse_detections(sample_detections_empty)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["category_label"] == "empty"


def test_parse_detections_raises_on_empty_list():
    with pytest.raises(AssertionError):
        parse_detections([])


def test_parse_detections_raises_on_non_list():
    with pytest.raises(TypeError):
        parse_detections({"not": "a list"})


def test_parse_detections_threshold_filters(sample_detections_with_animal):
    df = parse_detections(sample_detections_with_animal, threshold=0.95)
    # The only detection has conf=0.9, which is below the threshold of 0.95,
    # so it should be filtered out leaving no animal rows with a non-null conf.
    assert df.empty


def test_parse_detections_merges_manifest(sample_detections_with_animal, tmp_path: Path):
    manifest = pd.DataFrame(
        {"filepath": ["img_a.jpg"], "frame": [0], "station": ["s1"]}
    )
    df = parse_detections(sample_detections_with_animal, manifest=manifest)
    assert "station" in df.columns


def test_parse_detections_raises_invalid_manifest_col(sample_detections_with_animal):
    manifest = pd.DataFrame({"other_col": ["img_a.jpg"]})
    with pytest.raises(ValueError):
        parse_detections(sample_detections_with_animal, manifest=manifest, file_col="filepath")


def test_parse_detections_saves_to_file(tmp_path: Path, sample_detections_with_animal):
    out = tmp_path / "detections.csv"
    df = parse_detections(sample_detections_with_animal, out_file=str(out))
    assert out.exists()
    assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# get_animals / get_empty
# ---------------------------------------------------------------------------

@pytest.fixture
def manifest_with_mixed_categories() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filepath": ["a.jpg", "b.jpg", "c.jpg"],
            "frame": [0, 0, 0],
            "category": [1, 0, 2],
            "conf": [0.9, 1, 0.8],
            "category_label": ["animal", "empty", "vehicle"],
        }
    )


def test_get_animals_returns_only_animals(manifest_with_mixed_categories):
    animals = get_animals(manifest_with_mixed_categories)
    assert not animals.empty
    assert all(animals["category_label"] == "animal")


def test_get_animals_returns_empty_df_when_no_animals():
    manifest = pd.DataFrame(
        {"category_label": ["empty", "vehicle"], "filepath": ["a.jpg", "b.jpg"]}
    )
    animals = get_animals(manifest)
    assert animals.empty


def test_get_animals_uses_category_when_no_label_col():
    manifest = pd.DataFrame(
        {"category": [1, 0, 1], "filepath": ["a.jpg", "b.jpg", "c.jpg"]}
    )
    animals = get_animals(manifest)
    assert len(animals) == 2


def test_get_empty_returns_non_animals(manifest_with_mixed_categories):
    others = get_empty(manifest_with_mixed_categories)
    assert not others.empty
    assert all(others["category_label"] != "animal")


def test_get_empty_returns_empty_df_when_all_animals():
    manifest = pd.DataFrame(
        {
            "category": [1, 1],
            "category_label": ["animal", "animal"],
            "filepath": ["a.jpg", "b.jpg"],
        }
    )
    others = get_empty(manifest)
    assert others.empty


# ---------------------------------------------------------------------------
# _convert_detections (internal helper)
# ---------------------------------------------------------------------------

def test_convert_detections_no_detection_produces_empty_result():
    # Simulate a batch with one image that has zero predictions above threshold.
    image_tensor = np.zeros((1, 3, 32, 32), dtype=np.float32)
    image_paths = ["img.jpg"]
    image_frames = [0]
    image_sizes = np.array([[32, 32]])
    # pred with no detections (empty rows)
    predictions = [np.zeros((0, 6), dtype=np.float32)]
    batch = (image_tensor, image_paths, image_frames, image_sizes)
    results = _convert_detections(predictions, batch, letterbox=False)
    assert len(results) == 1
    assert results[0]["detections"] == []
    assert results[0]["filepath"] == "img.jpg"
