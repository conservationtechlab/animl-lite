from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from animl.classification import (
    classify,
    load_class_list,
    load_classifier,
    single_classification,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def class_list_file(tmp_path: Path) -> Path:
    f = tmp_path / "classes.csv"
    f.write_text("class\njaguar\nocelot\npuma\n")
    return f


@pytest.fixture
def class_list_df() -> pd.DataFrame:
    return pd.DataFrame({"class": ["jaguar", "ocelot", "puma"]})


@pytest.fixture
def animals_df(tmp_path: Path) -> pd.DataFrame:
    from PIL import Image

    img = tmp_path / "animal.jpg"
    Image.new("RGB", (299, 299), color=(100, 50, 10)).save(img)
    return pd.DataFrame(
        {
            "filepath": [str(img)],
            "frame": [0],
            "conf": [0.9],
            "category": [1],
            "category_label": ["animal"],
            "max_detection_conf": [0.9],
            "bbox_x": [0.1],
            "bbox_y": [0.1],
            "bbox_w": [0.5],
            "bbox_h": [0.5],
        }
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filepath": ["empty.jpg"],
            "frame": [0],
            "conf": [None],
            "category": [0],
            "category_label": ["empty"],
            "max_detection_conf": [None],
            "prediction": ["empty"],
            "confidence": [1.0],
        }
    )


@pytest.fixture
def real_classifier_model() -> Path:
    """
    Use an existing real ONNX model file for testing.
    Point this to your actual model location.
    """
    # Option 1: Copy from a known location in your repo/test data
    real_model_path = Path(__file__).parent / "fixtures" / "sdzwa_southwest_v3.onnx"
    
    if real_model_path.exists():
        return real_model_path

    pytest.skip("Real classifier model not found")


# ---------------------------------------------------------------------------
# load_class_list
# ---------------------------------------------------------------------------

def test_load_class_list_returns_dataframe(class_list_file: Path):
    df = load_class_list(str(class_list_file))
    assert isinstance(df, pd.DataFrame)
    assert "class" in df.columns
    assert len(df) == 3


def test_load_class_list_raises_for_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_class_list(str(tmp_path / "nonexistent.csv"))


# ---------------------------------------------------------------------------
# load_classifier
# ---------------------------------------------------------------------------

def test_load_classifier_raises_for_missing_model(tmp_path: Path):
    with pytest.raises(Exception):
        load_classifier(str(tmp_path / "nonexistent.onnx"))


def test_load_classifier_returns_model_and_class_list(real_classifier_model: Path, class_list_file: Path):
    model, class_list = load_classifier(str(real_classifier_model), classes=class_list_file)
    assert model is not None
    assert model.model_type == "classifier"
    assert class_list is not None


def test_load_classifier_accepts_dataframe_classes(real_classifier_model: Path, class_list_df: pd.DataFrame):
    model, class_list = load_classifier(str(real_classifier_model), classes=class_list_df)
    assert class_list is not None


def test_load_classifier_no_classes_returns_none_class_list(real_classifier_model: Path):
    model, class_list = load_classifier(str(real_classifier_model))
    # class list is should load from onnx model
    assert class_list is not None


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def test_classify_returns_numpy_array(real_classifier_model: Path, animals_df: pd.DataFrame):
    model, _ = load_classifier(str(real_classifier_model))
    result = classify(model, animals_df, resize_width=299, resize_height=299, crop=False)
    assert isinstance(result, np.ndarray)


def test_classify_raises_on_invalid_input(real_classifier_model: Path):
    model, _ = load_classifier(str(real_classifier_model))
    with pytest.raises(AssertionError):
        classify(model, 12345)


def test_classify_accepts_string_filepath(real_classifier_model: Path, animals_df: pd.DataFrame):
    model, _ = load_classifier(str(real_classifier_model))
    filepath = animals_df["filepath"].iloc[0]
    result = classify(model, filepath, resize_width=299, resize_height=299)
    assert isinstance(result, np.ndarray)


def test_classify_accepts_list_of_filepaths(real_classifier_model: Path, animals_df: pd.DataFrame):
    model, _ = load_classifier(str(real_classifier_model))
    result = classify(model, animals_df, resize_width=299, resize_height=299)
    assert isinstance(result, np.ndarray)


def test_classify_saves_to_file(real_classifier_model: Path, tmp_path: Path, animals_df: pd.DataFrame):
    model, _ = load_classifier(str(real_classifier_model))
    out = tmp_path / "classifications.csv"
    classify(model, animals_df, resize_width=299, resize_height=299, crop=False, out_file=str(out))
    assert out.exists()


def test_classify_raises_on_missing_file_col(real_classifier_model: Path):
    model, _ = load_classifier(str(real_classifier_model))
    df = pd.DataFrame({"other_col": ["img.jpg"]})
    with pytest.raises(ValueError):
        classify(model, df, file_col="filepath")


# ---------------------------------------------------------------------------
# single_classification
# ---------------------------------------------------------------------------

def test_single_classification_adds_prediction_column(animals_df: pd.DataFrame):
    predictions_raw = np.array([[0.1, 0.8, 0.1]])
    class_list = ["jaguar", "ocelot", "puma"]
    result = single_classification(animals_df, None, predictions_raw, class_list)
    assert "prediction" in result.columns
    assert "confidence" in result.columns
    assert result["prediction"].iloc[0] == "ocelot"


def test_single_classification_includes_empty(animals_df: pd.DataFrame, empty_df: pd.DataFrame):
    predictions_raw = np.array([[0.1, 0.8, 0.1]])
    class_list = ["jaguar", "ocelot", "puma"]
    result = single_classification(animals_df, empty_df, predictions_raw, class_list)
    assert len(result) == 2
    assert "empty" in result["prediction"].values


def test_single_classification_best_returns_one_per_file(animals_df: pd.DataFrame):
    predictions_raw = np.array([[0.1, 0.8, 0.1]])
    class_list = ["jaguar", "ocelot", "puma"]
    result = single_classification(animals_df, None, predictions_raw, class_list, best=True)
    assert len(result) == 1


def test_single_classification_empty_animals_still_returns(empty_df: pd.DataFrame):
    animals = pd.DataFrame(
        columns=["filepath", "frame", "conf", "category", "category_label",
                 "max_detection_conf", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    )
    predictions_raw = np.zeros((0, 3))
    class_list = ["jaguar", "ocelot", "puma"]
    result = single_classification(animals, empty_df, predictions_raw, class_list)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
