"""Tests for the frame score used to pick the best frames of a video (issue #20)."""
from __future__ import annotations

import numpy as np
import pytest

from animl.detection import _convert_detections
from animl.utils.general import (
    ASPECT_MODES,
    SCORE_WEIGHTS,
    _aspect_score,
    _crop_clarity,
    _frame_score,
)


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

def test_default_weights_sum_to_one():
    assert sum(SCORE_WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _aspect_score
# ---------------------------------------------------------------------------

def test_aspect_wide_rewards_wide_boxes():
    wide = _aspect_score(0.4, 0.2, 1000, 1000, "wide")
    tall = _aspect_score(0.2, 0.4, 1000, 1000, "wide")
    assert wide > tall


def test_aspect_tall_rewards_tall_boxes():
    wide = _aspect_score(0.4, 0.2, 1000, 1000, "tall")
    tall = _aspect_score(0.2, 0.4, 1000, 1000, "tall")
    assert tall > wide


def test_aspect_none_is_zero():
    assert _aspect_score(0.4, 0.2, 1000, 1000, "none") == 0.0


def test_aspect_is_capped_at_one():
    assert _aspect_score(0.9, 0.1, 1000, 1000, "wide") == 1.0


def test_aspect_uses_pixels_not_normalized_values():
    # a box that is 0.5 x 0.5 of a 1920x1080 frame is 960 x 540 px (ratio 1.78), not square
    assert _aspect_score(0.5, 0.5, 1920, 1080, "wide") == pytest.approx((960 / 540) / 2)
    assert _aspect_score(0.5, 0.5, 1000, 1000, "wide") == pytest.approx(0.5)


def test_aspect_zero_size_box_is_zero():
    assert _aspect_score(0.0, 0.3, 1000, 1000, "wide") == 0.0


def test_aspect_rejects_unknown_mode():
    with pytest.raises(ValueError):
        _aspect_score(0.3, 0.3, 1000, 1000, "diagonal")


# ---------------------------------------------------------------------------
# _crop_clarity
# ---------------------------------------------------------------------------

def _image_with_texture_in_box():
    rng = np.random.default_rng(0)
    img = np.full((200, 200, 3), 0.5, dtype=np.float32)
    img[50:150, 50:150] = rng.random((100, 100, 3)).astype(np.float32)
    return img


def test_crop_clarity_is_high_for_sharp_box():
    img = _image_with_texture_in_box()
    assert _crop_clarity(img, (0.25, 0.25, 0.5, 0.5)) > 0.5


def test_crop_clarity_ignores_texture_outside_the_box():
    img = _image_with_texture_in_box()
    # box in the flat, featureless corner
    assert _crop_clarity(img, (0.0, 0.0, 0.2, 0.2)) == pytest.approx(0.0)


def test_crop_clarity_is_between_zero_and_one():
    img = _image_with_texture_in_box()
    assert 0.0 <= _crop_clarity(img, (0.0, 0.0, 1.0, 1.0)) <= 1.0


def test_crop_clarity_empty_box_is_zero():
    img = _image_with_texture_in_box()
    assert _crop_clarity(img, (0.5, 0.5, 0.0, 0.0)) == 0.0


# ---------------------------------------------------------------------------
# _frame_score
# ---------------------------------------------------------------------------

def test_larger_box_scores_higher():
    small = _frame_score(0.9, 0.1, 0.1, 0.5, 1000, 1000)
    large = _frame_score(0.9, 0.5, 0.5, 0.5, 1000, 1000)
    assert large > small


def test_sharper_crop_scores_higher():
    blurry = _frame_score(0.9, 0.3, 0.3, 0.1, 1000, 1000)
    sharp = _frame_score(0.9, 0.3, 0.3, 0.9, 1000, 1000)
    assert sharp > blurry


def test_custom_weights_are_used():
    only_conf = {'conf': 1.0, 'area': 0.0, 'aspect': 0.0, 'clarity': 0.0}
    assert _frame_score(0.7, 0.9, 0.9, 1.0, 1000, 1000, weights=only_conf) == pytest.approx(0.7)


def test_aspect_mode_changes_score_for_a_wide_box():
    args = (0.9, 0.4, 0.2, 0.5, 1000, 1000)
    assert _frame_score(*args, aspect_mode="wide") > _frame_score(*args, aspect_mode="tall")
    assert _frame_score(*args, aspect_mode="none") < _frame_score(*args, aspect_mode="wide")


# ---------------------------------------------------------------------------
# _convert_detections end to end (clarity + score)
# ---------------------------------------------------------------------------

def _batch(height=1080, width=1920, size=64):
    rng = np.random.default_rng(1)
    tensor = rng.random((1, 3, size, size)).astype(np.float32)
    pred = [np.array([[8, 16, 40, 32, 0.9, 0]], dtype=np.float32)]  # absolute x1,y1,x2,y2,conf,class
    batch = (tensor, ["vid.mp4"], [0], np.array([[height, width]]))
    return pred, batch


def test_convert_detections_adds_clarity_and_score():
    pred, batch = _batch()
    det = _convert_detections(pred, batch, letterbox=False, calculate_clarity=True)[0]['detections'][0]
    assert 0.0 <= det['clarity'] <= 1.0
    assert det['score'] > 0


def test_convert_detections_without_clarity_has_no_score():
    pred, batch = _batch()
    det = _convert_detections(pred, batch, letterbox=False)[0]['detections'][0]
    assert 'score' not in det


def test_convert_detections_aspect_mode_changes_score():
    pred, batch = _batch()
    wide = _convert_detections(pred, batch, letterbox=False, calculate_clarity=True,
                               aspect_mode="wide")[0]['detections'][0]['score']
    pred, batch = _batch()
    tall = _convert_detections(pred, batch, letterbox=False, calculate_clarity=True,
                               aspect_mode="tall")[0]['detections'][0]['score']
    assert wide > tall  # the box is wider than tall


def test_convert_detections_rejects_bad_aspect_mode():
    pred, batch = _batch()
    with pytest.raises(ValueError):
        _convert_detections(pred, batch, letterbox=False, calculate_clarity=True, aspect_mode="sideways")


def test_aspect_modes_constant():
    assert set(ASPECT_MODES) == {"wide", "tall", "none"}
