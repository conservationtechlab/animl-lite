from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from animl.generator import ManifestGenerator


def _images(tmp_path: Path, n: int, size=(20, 10)) -> pd.DataFrame:
    paths = []
    for i in range(n):
        path = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", size, color=(i * 20, 0, 0)).save(path)
        paths.append(str(path))
    return pd.DataFrame({"filepath": paths})


def _generator(manifest, **kwargs):
    kwargs = {"crop": False, "resize_height": 8, "resize_width": 8, "batch_size": 2, **kwargs}
    return ManifestGenerator(manifest, **kwargs)


class TestPrefetch:
    def test_yields_every_batch(self, tmp_path):
        manifest = _images(tmp_path, 5)
        files = [f for _, paths, _, _ in _generator(manifest, num_workers=2) for f in paths]
        assert files == manifest["filepath"].tolist()

    def test_worker_error_is_raised(self, tmp_path, monkeypatch):
        """An error in the worker used to end the iteration as if all batches were done"""
        gen = _generator(_images(tmp_path, 6), num_workers=2)
        load_batch = gen._load_batch

        def failing(indices):
            if indices[0] == 2:
                raise ValueError("broken batch")
            return load_batch(indices)

        monkeypatch.setattr(gen, "_load_batch", failing)
        with pytest.raises(ValueError, match="broken batch"):
            list(gen)

    def test_slow_batch_is_waited_for(self, tmp_path, monkeypatch):
        """Batches slower than the queue poll are waited for (a 30s timeout used to end the iteration)"""
        gen = _generator(_images(tmp_path, 4), num_workers=2)
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
        first = next(iter(gen))
        assert first[1] == [manifest["filepath"][0]]
        files = [f for _, paths, _, _ in gen for f in paths]
        assert files == manifest["filepath"].tolist()


class TestReturnIndices:
    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_indices_of_loaded_items(self, tmp_path, num_workers):
        """Unloadable items are left out, so batch positions don't match manifest rows"""
        manifest = _images(tmp_path, 5)
        manifest.loc[1, "filepath"] = str(tmp_path / "missing.jpg")
        manifest.index = [10, 11, 12, 13, 14]   # positions, not labels, are returned
        gen = _generator(manifest, num_workers=num_workers, return_indices=True)
        batches = list(gen)
        assert [b[4].tolist() for b in batches] == [[0], [2, 3], [4]]
        for images, paths, _, _, indices in batches:
            assert len(images) == len(paths) == len(indices)
            assert paths == manifest["filepath"].iloc[indices].tolist()

    def test_four_items_by_default(self, tmp_path):
        (batch,) = list(_generator(_images(tmp_path, 2), num_workers=0))
        assert len(batch) == 4
