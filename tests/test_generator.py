from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from animl.generator import ManifestGenerator


def _images(tmp_path: Path, n: int, size=(20, 10), prefix="img") -> pd.DataFrame:
    paths = []
    for i in range(n):
        path = tmp_path / f"{prefix}_{i}.jpg"
        Image.new("RGB", size, color=(i * 20, 0, 0)).save(path)
        paths.append(str(path))
    return pd.DataFrame({"filepath": paths})


def _generator(manifest, **kwargs):
    kwargs = {"resize_height": 8, "resize_width": 8, "crop": False, "batch_size": 1, **kwargs}
    return ManifestGenerator(manifest, **kwargs)


class TestResize:
    @pytest.mark.parametrize("letterbox", [False, True])
    def test_every_image_gets_the_resize_size(self, tmp_path, letterbox):
        """The resize size is used whatever the image's aspect ratio, as in animl-py"""
        wide = _images(tmp_path, 1, size=(40, 10), prefix="wide")
        tall = _images(tmp_path, 1, size=(10, 40), prefix="tall")
        gen = _generator(pd.concat([wide, tall]), num_workers=0, letterbox=letterbox,
                         resize_width=16, resize_height=12)
        assert [batch[0].shape[1:] for batch, _ in gen] == [(3, 12, 16), (3, 12, 16)]
        assert (gen.resize_width, gen.resize_height) == (16, 12)

    @pytest.mark.parametrize("width,height", [(16, 0), (0, 16), (-1, 16)])
    def test_resize_size_must_be_positive(self, tmp_path, width, height):
        """A size of 0 or less made the size follow the first image, then stuck for every later image"""
        with pytest.raises(ValueError, match="must be positive"):
            _generator(_images(tmp_path, 1), resize_width=width, resize_height=height)
