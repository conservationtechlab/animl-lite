from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
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
              "num_workers": 0, "use_progress_bar": False, **kwargs}
    return classification.classify(FakeClassifier(), manifest, **kwargs)


def test_softmax_over_classes_not_over_the_batch(tmp_path):
    """Probabilities were normalised across the images of a batch"""
    probs = _classify(_manifest(tmp_path, 3), batch_size=3)
    np.testing.assert_allclose(probs, softmax(LOGITS), rtol=1e-6)
    np.testing.assert_allclose(probs.sum(axis=1), 1, rtol=1e-6)


def test_same_result_for_any_batch_size(tmp_path):
    manifest = _manifest(tmp_path, 3)
    np.testing.assert_allclose(_classify(manifest, batch_size=1), _classify(manifest, batch_size=2), rtol=1e-6)
