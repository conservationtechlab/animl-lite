from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from animl.classification import classify, sequence_classification, single_classification


class _FakeClassifier:
    """Returns the same logits for every image: class 0 wins."""

    def get_inputs(self):
        return [type("Input", (), {"name": "input"})()]

    def run(self, _outputs, feeds):
        n = feeds["input"].shape[0]
        # like onnxruntime, a list of the model's outputs
        return [np.tile(np.array([[2.0, 1.0, 0.0]], dtype=np.float32), (n, 1))]


LOGITS = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float32)


class _FakeClassifierByImage:
    """Image i (red = 20 * i) gets LOGITS[i], so each row can be traced to its image."""

    def get_inputs(self):
        return [type("Input", (), {"name": "input"})()]

    def run(self, _outputs, feeds):
        rows = np.rint(feeds["input"][:, 0].mean(axis=(1, 2)) * 255 / 20).astype(int)
        return [LOGITS[rows]]


def _images(tmp_path: Path, n: int) -> pd.DataFrame:
    paths = []
    for i in range(n):
        path = tmp_path / f"img_{i}.png"
        Image.new("RGB", (16, 16), color=(20 * i, 0, 0)).save(path)
        paths.append(str(path))
    return pd.DataFrame({"filepath": paths, "frame": 0})


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_classify_softmax_over_classes_not_over_the_batch(tmp_path: Path, batch_size):
    """Probabilities were normalised across the images of a batch, and were all 1 for a batch of one"""
    expected = np.exp(LOGITS) / np.exp(LOGITS).sum(axis=1, keepdims=True)
    predictions, _ = classify(_FakeClassifierByImage(), _images(tmp_path, 3), 8, 8, crop=False, normalize=True,
                              batch_size=batch_size, num_workers=0)
    np.testing.assert_allclose(predictions, expected, rtol=1e-5)


@pytest.fixture
def detections(tmp_path: Path) -> pd.DataFrame:
    img = tmp_path / "img.jpg"
    Image.new("RGB", (20, 10), color=(0, 0, 0)).save(img)
    return pd.DataFrame({
        "filepath": [str(img), str(tmp_path / "missing.jpg"), str(img)],
        "frame": [0, 0, 0],
        "bbox_x": [0.0, 0.0, 0.5], "bbox_y": [0.0, 0.0, 0.0],
        "bbox_w": [0.5, 0.5, 0.5], "bbox_h": [1.0, 1.0, 1.0],
    })


def test_classify_returns_failed_rows(detections):
    predictions, failed_rows = classify(_FakeClassifier(), detections, 8, 8, batch_size=2, num_workers=0)
    assert failed_rows == [1]
    assert predictions.shape == (2, 3)


def test_classify_resume_from_out_file_returns_failed_rows(detections, tmp_path: Path):
    out_file = str(tmp_path / "predictions.csv")
    predictions, failed_rows = classify(_FakeClassifier(), detections, 8, 8, num_workers=0, out_file=out_file)
    assert (tmp_path / "classification_failed_files.txt").is_file()
    with patch("builtins.input", return_value="y"):
        loaded, loaded_failed_rows = classify(_FakeClassifier(), detections, 8, 8, num_workers=0, out_file=out_file)
    assert loaded_failed_rows == failed_rows
    np.testing.assert_allclose(loaded, predictions, rtol=1e-5)


def test_classify_no_detections_loaded_raises(tmp_path: Path):
    detections = pd.DataFrame({"filepath": [str(tmp_path / "missing.jpg")], "frame": [0]})
    with pytest.raises(ValueError, match="None of the detections could be loaded"):
        classify(_FakeClassifier(), detections, 8, 8, crop=False, num_workers=0)


def test_single_classification_drops_only_failed_rows():
    """A failed crop drops only its own row, not other detections in the same file."""
    animals = pd.DataFrame({
        "filepath": ["a.jpg", "a.jpg", "b.jpg"],
        "extension": [".jpg", ".jpg", ".jpg"],
        "conf": [0.9, 0.8, 0.7],
    })
    predictions = np.array([[0.9, 0.05, 0.05], [0.1, 0.1, 0.8]])  # rows 0 and 2 loaded
    result = single_classification(animals, None, (predictions, [1]), ["cat", "dog", "bird"])
    assert result["filepath"].tolist() == ["a.jpg", "b.jpg"]
    assert result["prediction"].tolist() == ["cat", "bird"]


def test_single_classification_failed_rows_from_reticulate_dict():
    """Failed rows passed back from R as floats in a dict should drop the same rows."""
    animals = pd.DataFrame({
        "filepath": ["a.jpg", "b.jpg", "c.jpg"],
        "extension": [".jpg", ".jpg", ".jpg"],
        "conf": [0.9, 0.8, 0.7],
    })
    predictions = np.array([[0.9, 0.05, 0.05], [0.1, 0.1, 0.8]])
    result = single_classification(animals, None, {"predictions": predictions, "failed_rows": [1.0]},
                                   ["cat", "dog", "bird"])
    assert result["filepath"].tolist() == ["a.jpg", "c.jpg"]


def _animals(n=3):
    return pd.DataFrame({
        "filepath": [f"{i}.jpg" for i in range(n)],
        "extension": [".jpg"] * n,
        "conf": [0.9] * n,
        "station": ["A"] * n,
        "datetime": pd.to_datetime(["2023-01-01 10:00:00"] * n),
    })


@pytest.mark.parametrize("predictions,message", [
    (np.full((2, 3), 1 / 3), "Number of predictions does not match"),
    (np.full((3, 2), 0.5), "Number of classes in predictions does not match"),
])
def test_single_classification_checks_shape(predictions, message):
    with pytest.raises(AssertionError, match=message):
        single_classification(_animals(), None, predictions, ["cat", "dog", "bird"])


def test_single_classification_count_checked_after_failed_rows_dropped():
    predictions = np.full((3, 3), 1 / 3)  # sized for every row, but row 1 failed
    with pytest.raises(AssertionError, match="Number of predictions does not match"):
        single_classification(_animals(), None, (predictions, [1]), ["cat", "dog", "bird"])


@pytest.mark.parametrize("predictions,message", [
    (np.full((2, 3), 1 / 3), "Number of predictions does not match"),
    (np.full((3, 2), 0.5), "Number of classes in predictions does not match"),
])
def test_sequence_classification_checks_shape(predictions, message):
    with pytest.raises(AssertionError, match=message):
        sequence_classification(_animals(), None, predictions, pd.Series(["cat", "dog", "bird"]), "station")
