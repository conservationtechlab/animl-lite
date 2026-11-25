from __future__ import annotations

import time
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


class TestPrefetch:
    def test_yields_every_batch(self, tmp_path):
        manifest = _images(tmp_path, 5)
        gen = _generator(manifest, num_workers=2, batch_size=2)
        files = [f for batch, _ in gen for f in batch[1]]
        assert files == manifest["filepath"].tolist()

    def test_worker_error_is_raised(self, tmp_path, monkeypatch):
        """An error in the worker used to end the iteration as if all batches were done"""
        gen = _generator(_images(tmp_path, 6), num_workers=2, batch_size=2)
        load_batch = gen._load_batch

        def failing(indices):
            if indices[0] == 2:
                raise ValueError("broken batch")
            return load_batch(indices)

        monkeypatch.setattr(gen, "_load_batch", failing)
        with pytest.raises(ValueError, match="broken batch"):
            list(gen)

    def test_slow_batch_is_waited_for(self, tmp_path, monkeypatch):
        """A batch slower than the queue poll is waited for, a 30s timeout used to end the iteration"""
        gen = _generator(_images(tmp_path, 4), num_workers=2, batch_size=2)
        load_batch = gen._load_batch

        def slow(indices):
            if indices[0] == 2:
                time.sleep(1.5)   # longer than the poll interval
            return load_batch(indices)

        monkeypatch.setattr(gen, "_load_batch", slow)
        assert len(list(gen)) == 2

    def test_interrupted_iteration_doesnt_leak_into_the_next(self, tmp_path):
        manifest = _images(tmp_path, 8)
        gen = _generator(manifest, num_workers=2, batch_size=1)
        first, _ = next(iter(gen))
        assert first[1] == [manifest["filepath"][0]]
        files = [f for batch, _ in gen for f in batch[1]]
        assert files == manifest["filepath"].tolist()

    def test_interrupted_slow_worker_doesnt_leak_into_the_next(self, tmp_path, monkeypatch):
        """A worker still loading when its iteration stopped must not put batches into the next one"""
        manifest = _images(tmp_path, 4)
        gen = _generator(manifest, num_workers=2, batch_size=1)
        load_batch = gen._load_batch
        calls = []

        def slow_first_run(indices):
            calls.append(indices[0])
            if len(calls) == 2:
                time.sleep(6)   # longer than the 5s join, so the worker outlives its iteration
            return load_batch(indices)

        monkeypatch.setattr(gen, "_load_batch", slow_first_run)
        it = iter(gen)
        next(it)
        it.close()
        files = [f for batch, _ in gen for f in batch[1]]
        assert files == manifest["filepath"].tolist()
