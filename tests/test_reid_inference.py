from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from animl.reid import inference


class FakeMiew:
    """Embedding = the image's mean red value, so each row can be traced to its image"""
    model_type = "miewid"

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, _names, feed):
        images = feed["input"]
        return [np.stack([images[:, 0].mean(axis=(1, 2)), np.ones(len(images))], axis=1).astype(np.float32)]


def test_embeddings_match_their_rows(tmp_path):
    """A missing image shifted every later embedding to the wrong row"""
    paths = []
    for name, red in (("a", 0), ("missing", None), ("c", 255)):
        path = tmp_path / f"{name}.png"
        if red is not None:
            Image.new("RGB", (16, 16), color=(red, 0, 0)).save(path)
        paths.append(str(path))
    manifest = pd.DataFrame({"filepath": paths, "frame": 0, "bbox_x": 0.0, "bbox_y": 0.0, "bbox_w": 1.0, "bbox_h": 1.0})
    emb = inference.extract_miew_embeddings(FakeMiew(), manifest, batch_size=2, num_workers=0)
    assert emb.shape == (3, 2)
    assert emb[0, 0] < emb[2, 0]
    assert np.isnan(emb[1]).all()


def test_nothing_loaded(tmp_path):
    manifest = pd.DataFrame({"filepath": [str(tmp_path / "missing.png")], "frame": [0]})
    assert inference.extract_miew_embeddings(FakeMiew(), manifest, num_workers=0).shape == (1, 0)
