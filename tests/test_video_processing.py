from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from animl import video_processing


def _video(path: Path, n_frames: int, fps: int = 10) -> str:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    for i in range(n_frames):
        writer.write(np.full((24, 32, 3), i % 255, np.uint8))
    writer.release()
    return str(path)


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
        def slow(filepath, frames=5, fps=None):
            time.sleep(0.5)
            return [[str(filepath), 0]]

        monkeypatch.setattr(video_processing, "_count_frames", slow)
        files = [str(tmp_path / f"v{i}.mp4") for i in range(4)]
        start = time.time()
        frames = _frames(files, parallel=True, num_workers=4)
        assert time.time() - start < 1.5
        assert frames["filepath"].tolist() == files


class TestFrameSelection:
    def _selected(self, path, **kwargs):
        return [frame for _, frame in video_processing._count_frames(path, **kwargs)]

    def test_video_shorter_than_a_second_gets_a_frame(self, tmp_path):
        """The duration was truncated to whole seconds, so short clips got no frames"""
        assert self._selected(_video(tmp_path / "short.mp4", 7, fps=10), fps=1) == [0]

    def test_whole_duration_is_sampled(self, tmp_path):
        """A 2.5 s clip at 1 fps lost its last half second"""
        assert self._selected(_video(tmp_path / "clip.mp4", 25, fps=10), fps=1) == [0, 10, 20]

    def test_no_repeated_frames_when_sampling_faster_than_the_video(self, tmp_path):
        frames = self._selected(_video(tmp_path / "clip.mp4", 10, fps=5), fps=12)
        assert frames == sorted(set(frames))

    def test_no_repeated_frames_when_asking_for_more_than_the_video_has(self, tmp_path):
        """increment was 0, so frame 0 was returned for every requested frame"""
        assert self._selected(_video(tmp_path / "clip.mp4", 3), frames=5) == [0, 1, 2]

    def test_frames_spread_as_before(self, tmp_path):
        assert self._selected(_video(tmp_path / "clip.mp4", 100), frames=3) == [0, 33, 66]
