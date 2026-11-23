from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from animl import classification
from animl.utils.general import softmax

LOGITS = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float32)


class FakeClassifier:
    """Logits chosen by each image's red value: image i (red = 20 * i) gets LOGITS[i]"""
    model_type = "classifier"

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, _names, feed):
        images = feed["input"]
        rows = np.rint(images[:, 0].mean(axis=(1, 2)) * 255 / 20).astype(int)
        return [LOGITS[rows]]


def _manifest(tmp_path: Path, n: int) -> pd.DataFrame:
    paths = []
    for i in range(n):
        path = tmp_path / f"img_{i}.png"
        Image.new("RGB", (16, 16), color=(20 * i, 0, 0)).save(path)
        paths.append(str(path))
    return pd.DataFrame({"filepath": paths, "frame": 0})


def _classify(manifest, **kwargs):
    kwargs = {"resize_width": 8, "resize_height": 8, "crop": False, "normalize": True,
              "num_workers": 0, "quiet": True, **kwargs}
    return classification.classify(FakeClassifier(), manifest, **kwargs)


def _predictions(manifest, **kwargs):
    predictions, failed_files = _classify(manifest, **kwargs)
    assert failed_files == []
    return predictions


def test_softmax_over_classes_not_over_the_batch(tmp_path):
    """Probabilities were normalised across the images of a batch"""
    probs = _predictions(_manifest(tmp_path, 3), batch_size=3)
    np.testing.assert_allclose(probs, softmax(LOGITS), rtol=1e-6)
    np.testing.assert_allclose(probs.sum(axis=1), 1, rtol=1e-6)


def test_same_result_for_any_batch_size(tmp_path):
    manifest = _manifest(tmp_path, 3)
    np.testing.assert_allclose(_predictions(manifest, batch_size=1), _predictions(manifest, batch_size=2), rtol=1e-6)


class TestFailedFiles:
    """Unloadable images are left out of their batch; results must still match their rows.
    classify returns (predictions, failed_files) as in animl-py."""

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_rows_match_detections(self, tmp_path, num_workers):
        manifest = _manifest(tmp_path, 3)
        missing = str(tmp_path / "missing.png")
        manifest = pd.concat([manifest.iloc[:1], pd.DataFrame({"filepath": [missing], "frame": [0]}),
                              manifest.iloc[1:]], ignore_index=True)
        predictions, failed_files = _classify(manifest, batch_size=2, num_workers=num_workers)
        np.testing.assert_allclose(predictions, softmax(LOGITS), rtol=1e-6)
        assert failed_files == [missing]

    def test_nothing_loaded(self, tmp_path):
        """np.vstack([]) used to raise"""
        missing = str(tmp_path / "missing.png")
        predictions, failed_files = _classify(pd.DataFrame({"filepath": [missing], "frame": [0]}))
        assert len(predictions) == 0
        assert failed_files == [missing]

    def test_whole_file_left_out_when_one_of_its_rows_fails(self):
        """e.g. one unreadable frame of a video: consumers drop the file, so the predictions must too"""
        by_row = np.array([[0.9, 0.1], [np.nan, np.nan], [0.2, 0.8]])
        filepaths = pd.Series(["video.mp4", "video.mp4", "b.jpg"])
        predictions, failed_files = classification._split_failed(by_row, filepaths)
        assert failed_files == ["video.mp4"]
        np.testing.assert_array_equal(predictions, [[0.2, 0.8]])

    def test_save_and_load_failed_files(self, tmp_path):
        out_file = str(tmp_path / "predictions.csv")
        assert classification.load_failed_files(out_file) == []
        classification.save_failed_files(out_file, ["a.jpg", "b.mp4"])
        assert classification.load_failed_files(out_file) == ["a.jpg", "b.mp4"]
        # no failures: the list from the earlier run no longer applies
        classification.save_failed_files(out_file, [])
        assert not (tmp_path / classification.FAILED_FILES_NAME).exists()

    def test_failed_files_saved_and_loaded_with_out_file(self, tmp_path, monkeypatch):
        manifest = _manifest(tmp_path, 2)
        missing = str(tmp_path / "missing.png")
        manifest = pd.concat([manifest, pd.DataFrame({"filepath": [missing], "frame": [0]})], ignore_index=True)
        out_file = tmp_path / "out" / "predictions.csv"
        out_file.parent.mkdir()
        predictions, failed_files = _classify(manifest, out_file=str(out_file))
        assert (out_file.parent / classification.FAILED_FILES_NAME).read_text() == f"{missing}\n"

        monkeypatch.setattr("builtins.input", lambda _prompt: "y")  # load the saved results
        loaded, loaded_failed = _classify(manifest, out_file=str(out_file))
        np.testing.assert_allclose(loaded, predictions, rtol=1e-6)
        assert loaded_failed == failed_files

    def test_single_classification_leaves_out_failed_files(self):
        animals = pd.DataFrame({"filepath": ["a.jpg", "missing.jpg", "c.jpg"], "conf": [1.0, 1.0, 1.0]})
        predictions = np.array([[0.9, 0.1], [0.2, 0.8]])
        result = classification.single_classification(animals, None, (predictions, ["missing.jpg"]), ["cat", "dog"])
        assert dict(zip(result["filepath"], result["prediction"])) == {"a.jpg": "cat", "c.jpg": "dog"}

    def test_single_classification_count_mismatch_raises(self):
        """predictions that don't match the detections used to be assigned to the wrong rows"""
        animals = pd.DataFrame({"filepath": ["a.jpg", "b.jpg", "c.jpg"], "conf": [1.0, 1.0, 1.0]})
        with pytest.raises(ValueError, match="does not match"):
            classification.single_classification(animals, None, np.array([[0.9, 0.1], [0.2, 0.8]]), ["cat", "dog"])

    def test_sequence_classification_leaves_out_failed_files(self):
        animals = pd.DataFrame({"filepath": ["a.jpg", "missing.jpg", "c.jpg"], "conf": [1.0, 1.0, 1.0],
                                "station": ["s1", "s1", "s2"], "frame": 0,
                                "datetime": ["2024-01-01 00:00:00", "2024-01-01 00:00:10", "2024-01-02 00:00:00"]})
        empty = pd.DataFrame({"filepath": ["e.jpg"], "conf": [1.0], "station": ["s3"], "frame": 0,
                              "datetime": ["2024-01-03 00:00:00"], "prediction": ["empty"], "confidence": [1.0]})
        predictions = np.array([[0.9, 0.1], [0.2, 0.8]])
        result = classification.sequence_classification(animals, empty, (predictions, ["missing.jpg"]),
                                                        pd.Series(["cat", "dog"]), station_col="station")
        labels = dict(zip(result["filepath"], result["prediction"]))
        assert "missing.jpg" not in labels
        assert labels["a.jpg"] == "cat" and labels["c.jpg"] == "dog"
