from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from animl import file_management
from animl.video_processing import (
    _count_frames,
    extract_frames,
    get_images,
    get_videos,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mixed_manifest(tmp_path: Path) -> pd.DataFrame:
    img = tmp_path / "photo.jpg"
    vid = tmp_path / "clip.mp4"
    img.write_bytes(b"fake-image")
    vid.write_bytes(b"fake-video")
    return pd.DataFrame({"filepath": [str(img), str(vid)]})


@pytest.fixture
def image_manifest(tmp_path: Path) -> pd.DataFrame:
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-image")
    return pd.DataFrame({"filepath": [str(img)]})


@pytest.fixture
def video_manifest(tmp_path: Path) -> pd.DataFrame:
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake-video")
    return pd.DataFrame({"filepath": [str(vid)]})


# ---------------------------------------------------------------------------
# get_images / get_videos
# ---------------------------------------------------------------------------

def test_get_images_returns_only_images(mixed_manifest: pd.DataFrame):
    images = get_images(mixed_manifest)
    assert not images.empty
    assert all(
        Path(p).suffix.lower() in file_management.IMAGE_EXTENSIONS
        for p in images["filepath"]
    )


def test_get_images_assigns_frame_zero(image_manifest: pd.DataFrame):
    images = get_images(image_manifest)
    assert list(images["frame"]) == [0]


def test_get_videos_returns_only_videos(mixed_manifest: pd.DataFrame):
    videos = get_videos(mixed_manifest)
    assert not videos.empty
    assert all(
        Path(p).suffix.lower() in file_management.VIDEO_EXTENSIONS
        for p in videos["filepath"]
    )


def test_get_images_empty_when_no_images(video_manifest: pd.DataFrame):
    images = get_images(video_manifest)
    assert images.empty


def test_get_videos_empty_when_no_videos(image_manifest: pd.DataFrame):
    videos = get_videos(image_manifest)
    assert videos.empty


# ---------------------------------------------------------------------------
# extract_frames
# ---------------------------------------------------------------------------

def test_extract_frames_raises_without_fps_and_frames(mixed_manifest: pd.DataFrame):
    with pytest.raises(AssertionError):
        extract_frames(mixed_manifest, frames=None, fps=None)


def test_extract_frames_raises_missing_file_col(mixed_manifest: pd.DataFrame):
    with pytest.raises(ValueError):
        extract_frames(mixed_manifest, frames=3, file_col="nonexistent_col")


def test_extract_frames_images_only_returns_manifest(image_manifest: pd.DataFrame):
    result = extract_frames(image_manifest, frames=3)
    assert isinstance(result, pd.DataFrame)
    assert "filepath" in result.columns
    assert "frame" in result.columns
    assert list(result["frame"]) == [0]


def test_extract_frames_includes_videos(video_manifest: pd.DataFrame):
    result = extract_frames(video_manifest, frames=3, parallel=False)
    assert isinstance(result, pd.DataFrame)
    assert "frame" in result.columns
    assert len(result) >= 1


def test_extract_frames_saves_to_file(tmp_path: Path, image_manifest: pd.DataFrame):
    out = tmp_path / "frames.csv"
    result = extract_frames(image_manifest, frames=1, out_file=str(out))
    assert out.exists()
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# _count_frames
# ---------------------------------------------------------------------------

def test_count_frames_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _count_frames(str(tmp_path / "nonexistent.mp4"), frames=5)


def test_count_frames_returns_list_of_pairs(tmp_path: Path):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake")
    result = _count_frames(str(vid), frames=3)
    assert result is not None
    assert isinstance(result, list)
    for pair in result:
        assert len(pair) == 2


def test_count_frames_with_fps(tmp_path: Path):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake")
    result = _count_frames(str(vid), fps=2)
    assert result is not None
    assert isinstance(result, list)
