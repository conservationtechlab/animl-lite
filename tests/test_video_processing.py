from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
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


def test_count_frames_invalid_file_returns_none(tmp_path: Path):
    """Test that _count_frames returns None for invalid/corrupted video files."""
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake")
    result = _count_frames(str(vid), frames=3)
    assert result is None


def test_count_frames_with_fps_invalid_file_returns_none(tmp_path: Path):
    """Test that _count_frames returns None for invalid files even with fps parameter."""
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake")
    result = _count_frames(str(vid), fps=2)
    assert result is None


def test_count_frames_with_frames_parameter_returns_list(tmp_path: Path):
    """Test that _count_frames returns a list of frame pairs when using frames parameter."""
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake")
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 100.0,
    }.get(prop, 30.0)
    
    with patch("cv2.VideoCapture", return_value=mock_cap):
        result = _count_frames(str(vid), frames=3)
    
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 3
    for pair in result:
        assert len(pair) == 2
        assert pair[0] == str(vid)
        assert isinstance(pair[1], int)


def test_count_frames_with_fps_parameter_returns_list(tmp_path: Path):
    """Test that _count_frames returns a list of frame pairs when using fps parameter."""
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake")
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 300.0,  # 10 seconds at 30 fps
        cv2.CAP_PROP_FPS: 30.0,
    }.get(prop, 30.0)
    
    with patch("cv2.VideoCapture", return_value=mock_cap):
        result = _count_frames(str(vid), fps=2)
    
    assert result is not None
    assert isinstance(result, list)
    assert len(result) > 0
    for pair in result:
        assert len(pair) == 2
        assert pair[0] == str(vid)
        assert isinstance(pair[1], int)


def test_count_frames_corrupted_video_returns_none(tmp_path: Path):
    """Test that _count_frames returns None when VideoCapture fails to open."""
    vid = tmp_path / "corrupted.mp4"
    vid.write_bytes(b"fake")
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    
    with patch("cv2.VideoCapture", return_value=mock_cap):
        result = _count_frames(str(vid), frames=5)
    
    assert result is None


def test_count_frames_zero_frame_count_returns_none(tmp_path: Path):
    """Test that _count_frames returns None when frame count is 0."""
    vid = tmp_path / "empty.mp4"
    vid.write_bytes(b"fake")
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 0.0,  # No frames
    }.get(prop, 30.0)
    
    with patch("cv2.VideoCapture", return_value=mock_cap):
        result = _count_frames(str(vid), frames=5)
    
    assert result is None


def test_count_frames_fps_zero_falls_back_to_default(tmp_path: Path):
    """Test that _count_frames handles fps=0 by falling back to ffmpeg or default."""
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake")
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 300.0,
        cv2.CAP_PROP_FPS: 0.0,  # fps detection fails
    }.get(prop, 0.0)
    
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with patch("animl.video_processing.get_fps_from_ffmpeg", return_value=None):
            result = _count_frames(str(vid), fps=2)
    
    assert result is not None
    assert isinstance(result, list)


def test_count_frames_frame_indices_within_bounds(tmp_path: Path):
    """Test that returned frame indices are within the valid range."""
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake")
    
    frame_count = 100
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: float(frame_count),
    }.get(prop, 30.0)
    
    with patch("cv2.VideoCapture", return_value=mock_cap):
        result = _count_frames(str(vid), frames=5)
    
    assert result is not None
    for pair in result:
        frame_idx = pair[1]
        assert 0 <= frame_idx < frame_count
