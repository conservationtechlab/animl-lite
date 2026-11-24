from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from animl.reid import extract_reid_embeddings


class _FakeReid:
    """Embedding = (mean red value, 1), so each row can be traced to its image."""

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, _names, feed):
        images = feed["input"]
        return [np.stack([images[:, 0].mean(axis=(1, 2)), np.ones(len(images))], axis=1).astype(np.float32)]


@pytest.fixture
def images(tmp_path: Path):
    dark, bright = tmp_path / "dark.png", tmp_path / "bright.png"
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(dark)
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(bright)
    return str(dark), str(bright), str(tmp_path / "missing.png")


def _manifest(rows):
    return pd.DataFrame([{"filepath": f, "frame": 0, "bbox_x": x, "bbox_y": 0.0, "bbox_w": 1.0, "bbox_h": 1.0}
                         for f, x in rows])


@pytest.mark.parametrize("batch_size", [1, 3])
def test_embeddings_match_manifest_rows(images, batch_size):
    """Images that failed to load were left out, so every later embedding moved up a row."""
    dark, bright, missing = images
    # a crop that fails (bbox outside the image) next to a good crop of the same image:
    # a list of failed filepaths can't tell these rows apart
    manifest = _manifest([(dark, 0.0), (dark, 1.5), (missing, 0.0), (bright, 0.0)])
    emb = extract_reid_embeddings(_FakeReid(), manifest, resize=8, batch_size=batch_size, num_workers=0)
    assert emb.shape == (4, 2)
    assert np.isnan(emb[[1, 2]]).all()
    assert np.isfinite(emb[[0, 3]]).all()
    assert emb[0, 0] < emb[3, 0]


def test_nothing_loaded_raises(images):
    _, _, missing = images
    with pytest.raises(ValueError, match="None of the images could be loaded"):
        extract_reid_embeddings(_FakeReid(), _manifest([(missing, 0.0)]), resize=8, num_workers=0)
