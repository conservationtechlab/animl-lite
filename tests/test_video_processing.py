from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import animl
from animl import video_processing


def _video(path: Path, n_frames: int, fps: int = 10) -> str:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    for i in range(n_frames):
        writer.write(np.full((24, 32, 3), i % 255, np.uint8))
    writer.release()
    return str(path)


def _selected(path, **kwargs):
    return [frame for _, frame in video_processing._count_frames(path, **kwargs)]


class TestCountFramesInvalidInput:
    def test_missing_file_returns_none(self, tmp_path, capsys):
        missing = str(tmp_path / "missing.mp4")
        assert video_processing._count_frames(missing) is None
        assert "does not exist" in capsys.readouterr().out

    def test_corrupted_file_returns_none(self, tmp_path, capsys):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")
        assert video_processing._count_frames(str(broken)) is None
        assert "could not be opened" in capsys.readouterr().out

    def test_zero_frame_video_returns_none(self, tmp_path, monkeypatch, capsys):
        path = _video(tmp_path / "clip.mp4", 5)

        class _EmptyCapture:
            def __init__(self, *_a):
                pass

            def isOpened(self):
                return True

            def get(self, prop):
                return 0

            def release(self):
                pass

        monkeypatch.setattr(video_processing.cv2, "VideoCapture", _EmptyCapture)
        assert video_processing._count_frames(path) is None
        assert "has 0 frames" in capsys.readouterr().out

    def test_quiet_suppresses_messages(self, tmp_path, capsys):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")
        assert video_processing._count_frames(str(tmp_path / "missing.mp4"), quiet=True) is None
        assert video_processing._count_frames(str(broken), quiet=True) is None
        assert capsys.readouterr().out == ""


class TestCountFramesSelection:
    def test_fixed_number_of_frames(self, tmp_path):
        assert _selected(_video(tmp_path / "clip.mp4", 100), frames=3) == [0, 33, 66]

    def test_fps_samples_whole_duration(self, tmp_path):
        """2.5 s at 1 fps -> ceil(2.5) = 3 frames; the old int() truncation gave 2"""
        assert _selected(_video(tmp_path / "clip.mp4", 25, fps=10), fps=1) == [0, 10, 20]

    def test_fps_clip_shorter_than_a_second_gets_a_frame(self, tmp_path):
        """0.7 s at 1 fps used to yield no frames"""
        assert _selected(_video(tmp_path / "short.mp4", 7, fps=10), fps=1) == [0]

    def test_fps_frames_capped_at_last_frame(self, tmp_path):
        frames = _selected(_video(tmp_path / "clip.mp4", 10, fps=5), fps=12)
        assert max(frames) <= 9

    def test_no_repeated_frames_when_sampling_faster_than_the_video(self, tmp_path):
        frames = _selected(_video(tmp_path / "clip.mp4", 10, fps=5), fps=12)
        assert frames == sorted(set(frames))

    def test_no_repeated_frames_when_asking_for_more_than_the_video_has(self, tmp_path):
        """increment was 0, so frame 0 was returned for every requested frame"""
        assert _selected(_video(tmp_path / "clip.mp4", 3), frames=5) == [0, 1, 2]

    def test_returns_filepath_with_each_frame(self, tmp_path):
        path = _video(tmp_path / "clip.mp4", 20)
        out = video_processing._count_frames(path, frames=2)
        assert all(fp == path for fp, _ in out)


class TestFpsFallback:
    def _zero_fps_capture(self, monkeypatch):
        real = cv2.VideoCapture

        class _ZeroFps:
            def __init__(self, path):
                self._cap = real(path)

            def __getattr__(self, name):
                return getattr(self._cap, name)

            def get(self, prop):
                if prop == cv2.CAP_PROP_FPS:
                    return 0
                return self._cap.get(prop)

        monkeypatch.setattr(video_processing.cv2, "VideoCapture", _ZeroFps)

    def test_uses_ffmpeg_fps_when_opencv_reports_zero(self, tmp_path, monkeypatch):
        path = _video(tmp_path / "clip.mp4", 20, fps=10)
        self._zero_fps_capture(monkeypatch)
        calls = []

        def fake_ffmpeg(video_path, quiet=False):
            calls.append((video_path, quiet))
            return 10.0

        monkeypatch.setattr(video_processing, "_get_fps_from_ffmpeg", fake_ffmpeg)
        assert _selected(path, fps=1, quiet=True) == [0, 10]
        assert calls == [(path, True)]

    def test_defaults_to_30_fps_when_ffmpeg_fails(self, tmp_path, monkeypatch, capsys):
        path = _video(tmp_path / "clip.mp4", 60, fps=10)
        self._zero_fps_capture(monkeypatch)
        monkeypatch.setattr(video_processing, "_get_fps_from_ffmpeg", lambda *_a, **_k: None)
        assert _selected(path, fps=1) == [0, 30]
        assert "defaulting to 30 FPS" in capsys.readouterr().out

    def test_default_fps_message_suppressed_when_quiet(self, tmp_path, monkeypatch, capsys):
        path = _video(tmp_path / "clip.mp4", 60, fps=10)
        self._zero_fps_capture(monkeypatch)
        monkeypatch.setattr(video_processing, "_get_fps_from_ffmpeg", lambda *_a, **_k: None)
        _selected(path, fps=1, quiet=True)
        assert capsys.readouterr().out == ""


class TestGetFpsFromFfmpeg:
    def _fake_run(self, monkeypatch, stderr):
        result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)
        monkeypatch.setattr(video_processing.subprocess, "run", lambda *_a, **_k: result)

    def test_parses_fps(self, monkeypatch, capsys):
        self._fake_run(monkeypatch, "Stream #0:0: Video: h264, 1920x1080, 29.97 fps, 30 tbr")
        assert video_processing._get_fps_from_ffmpeg("clip.mp4") == pytest.approx(29.97)
        assert "FPS: 29.97" in capsys.readouterr().out

    def test_no_fps_in_output(self, monkeypatch, capsys):
        self._fake_run(monkeypatch, "no stream info")
        assert video_processing._get_fps_from_ffmpeg("clip.mp4") is None
        assert "Could not find fps" in capsys.readouterr().out

    def test_quiet(self, monkeypatch, capsys):
        self._fake_run(monkeypatch, "Video: h264, 25 fps")
        assert video_processing._get_fps_from_ffmpeg("clip.mp4", quiet=True) == 25.0
        self._fake_run(monkeypatch, "no stream info")
        assert video_processing._get_fps_from_ffmpeg("clip.mp4", quiet=True) is None
        assert capsys.readouterr().out == ""

    def test_subprocess_error_returns_none(self, monkeypatch):
        def boom(*_a, **_k):
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr(video_processing.subprocess, "run", boom)
        assert video_processing._get_fps_from_ffmpeg("clip.mp4") is None

    def test_not_part_of_public_api(self):
        assert not hasattr(animl, "get_fps_from_ffmpeg")
        assert "get_fps_from_ffmpeg" not in animl.__all__


class TestExtractFrames:
    def test_missing_video_is_skipped(self, tmp_path):
        good = _video(tmp_path / "good.mp4", 30)
        files = pd.DataFrame({"filepath": [good, str(tmp_path / "missing.mp4")]})
        out = video_processing.extract_frames(files, frames=3, parallel=False)
        assert out["filepath"].unique().tolist() == [good]
        assert out["frame"].tolist() == [0, 10, 20]

    def test_images_pass_through_with_frame_zero(self, tmp_path):
        img = tmp_path / "a.jpg"
        cv2.imwrite(str(img), np.zeros((8, 8, 3), np.uint8))
        out = video_processing.extract_frames(pd.DataFrame({"filepath": [str(img)]}), parallel=False)
        assert out["frame"].tolist() == [0]

    def test_missing_file_col_raises(self):
        with pytest.raises(ValueError):
            video_processing.extract_frames(pd.DataFrame({"path": ["a.mp4"]}))

    def test_requires_frames_or_fps(self):
        with pytest.raises(AssertionError):
            video_processing.extract_frames(pd.DataFrame({"filepath": ["a.mp4"]}), frames=None, fps=None)


def test_get_frame_as_image_returns_rgb(tmp_path):
    path = _video(tmp_path / "clip.mp4", 5)
    image = video_processing.get_frame_as_image(path, frame=2)
    assert image.shape == (24, 32, 3)


def _frames(files, **kwargs):
    kwargs = {"frames": 3, "num_workers": 2, **kwargs}
    return video_processing.extract_frames(pd.DataFrame({"filepath": files}), **kwargs)


class TestParallel:
    def test_same_frames_as_serial(self, tmp_path):
        files = [_video(tmp_path / f"v{i}.mp4", 20 + 10 * i) for i in range(3)]
        parallel = _frames(files, parallel=True)
        serial = _frames(files, parallel=False)
        pd.testing.assert_frame_equal(parallel, serial)
        assert len(parallel) == 9

    @pytest.mark.parametrize("parallel", [True, False])
    def test_unreadable_videos(self, tmp_path, parallel):
        """np.vstack([]) raised in parallel mode when no video could be read"""
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")
        frames = _frames([str(broken)], parallel=parallel)
        assert frames.empty

    def test_videos_run_concurrently(self, tmp_path, monkeypatch):
        """pool.apply waits for each video, so 'parallel' ran one video at a time"""
        def slow(filepath, frames=5, fps=None, quiet=False):
            time.sleep(0.5)
            return [[str(filepath), 0]]

        monkeypatch.setattr(video_processing, "_count_frames", slow)
        files = [str(tmp_path / f"v{i}.mp4") for i in range(4)]
        start = time.time()
        frames = _frames(files, parallel=True, num_workers=4)
        assert time.time() - start < 1.5
        assert frames["filepath"].tolist() == files
