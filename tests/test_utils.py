from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from animl import __version__
from animl.utils import animlr
from animl.utils import general


@pytest.mark.parametrize(
    "values",
    [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 2.0]]),
        np.array([[-1.0, -2.0, -3.0]]),
        np.array([[10.0, 10.0, 10.0]]),
        np.array([[0.1, 0.2, 0.3, 0.4]]),
        np.array([[5.0, 1.0, -5.0]]),
        np.array([[100.0, 99.0]]),
        np.array([[-100.0, -99.0]]),
        np.array([[2.5, 2.5, 0.0]]),
        np.array([[9.0, 1.0, 1.0, 1.0]]),
    ],
)
def test_softmax_rows_sum_to_one(values):
    out = general.softmax(values)
    assert np.allclose(out.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "values,expected_argmax",
    [
        (np.array([[1.0, 2.0, 3.0]]), 2),
        (np.array([[3.0, 2.0, 1.0]]), 0),
        (np.array([[-3.0, -2.0, -1.0]]), 2),
        (np.array([[0.0, 5.0, 1.0]]), 1),
        (np.array([[8.0, 8.0, 1.0]]), 0),
    ],
)
def test_softmax_argmax_matches_input(values, expected_argmax):
    out = general.softmax(values)
    assert int(np.argmax(out[0])) == expected_argmax


def test_get_version_matches_package_version():
    assert animlr.get_version() == __version__


@pytest.mark.parametrize(
    "name,value",
    [
        ("MEGADETECTORv5_SIZE", general.MEGADETECTORv5_SIZE),
        ("SDZWA_CLASSIFIER_SIZE", general.SDZWA_CLASSIFIER_SIZE),
    ],
)
def test_constants_are_positive_ints(name, value):
    assert isinstance(value, int), f"{name} should be int"
    assert value > 0, f"{name} should be positive"


@pytest.mark.parametrize("model", ["megadetector", "yolo", "miewid", "classifier"])
def test_model_types_contains_expected_values(model):
    assert model in general.MODEL_TYPES


@pytest.mark.parametrize(
    "providers,user_set,quiet,expected",
    [
        (["CUDAExecutionProvider", "CPUExecutionProvider"], "cpu", False, ["CPUExecutionProvider"]),
        (["CUDAExecutionProvider", "CPUExecutionProvider"], "cpu", True, ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda:1",
            True,
            [("CUDAExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"],
        ),
        (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "cuda",
            True,
            [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        ),
        (["CUDAExecutionProvider", "CPUExecutionProvider"], None, True, ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        (["CUDAExecutionProvider", "CPUExecutionProvider"], "weird", True, ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        (["CPUExecutionProvider"], "cuda:0", True, ["CPUExecutionProvider"]),
        (["CPUExecutionProvider"], None, True, ["CPUExecutionProvider"]),
    ],
)
def test_get_onnx_device_variants(monkeypatch, providers, user_set, quiet, expected):
    monkeypatch.setattr(general.ort, "get_available_providers", lambda: providers)
    result = general.get_onnx_device(user_set=user_set, quiet=quiet)
    assert result == expected


def test_check_onnx_cuda_true(monkeypatch):
    fake_ort = types.SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    assert animlr.check_onnx_cuda() is True


def test_check_onnx_cuda_false(monkeypatch):
    fake_ort = types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    assert animlr.check_onnx_cuda() is False


def test_check_exiftool_success(monkeypatch):
    class OkExif:
        version = "12.0"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setitem(sys.modules, "exiftool", types.SimpleNamespace(ExifToolHelper=OkExif))
    assert animlr.check_exiftool() == "12.0"


def test_check_exiftool_failure(monkeypatch):
    class BadExif:
        def __enter__(self):
            raise RuntimeError("boom")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setitem(sys.modules, "exiftool", types.SimpleNamespace(ExifToolHelper=BadExif))
    assert animlr.check_exiftool() is False


@pytest.mark.parametrize(
    "bbox,expected",
    [
        (np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.1, 0.2, 0.4, 0.6])),
        (np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])),
        (np.array([0.2, 0.2, 0.1, 0.1]), np.array([0.2, 0.2, 0.3, 0.3])),
    ],
)
def test_xywh2xyxy(bbox, expected):
    out = general._xywh2xyxy(bbox)
    assert np.allclose(out, expected)


@pytest.mark.parametrize(
    "bbox,expected",
    [
        (np.array([0.1, 0.2, 0.4, 0.6]), np.array([0.1, 0.2, 0.3, 0.4])),
        (np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])),
        (np.array([0.2, 0.2, 0.3, 0.3]), np.array([0.2, 0.2, 0.1, 0.1])),
    ],
)
def test_xyxy2xywh(bbox, expected):
    out = general._xyxy2xywh(bbox)
    assert np.allclose(out, expected)


@pytest.mark.parametrize(
    "bbox,expected",
    [
        (np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.25, 0.4, 0.3, 0.4])),
        (np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.5, 0.5, 1.0, 1.0])),
    ],
)
def test_xywh_to_xywhc(bbox, expected):
    out = general._xywh_to_xywhc(bbox)
    assert np.allclose(out, expected)


@pytest.mark.parametrize(
    "bbox,width,height,expected",
    [
        ([0.1, 0.2, 0.3, 0.4], 100, 50, [10, 10, 40, 30]),
        ([0.0, 0.0, 1.0, 1.0], 10, 10, [0, 0, 10, 10]),
        ([0.5, 0.5, 0.2, 0.2], 200, 100, [100, 50, 140, 70]),
    ],
)
def test_xywh_to_absxyxy(bbox, width, height, expected):
    assert general._xywh_to_absxyxy(bbox, width, height) == expected


@pytest.mark.parametrize(
    "bbox,image_sizes,expected",
    [
        (np.array([10, 10, 50, 50], dtype=np.float32), (100, 100), np.array([0.1, 0.1, 0.5, 0.5])),
        (np.array([-5, -5, 120, 120], dtype=np.float32), (100, 100), np.array([0.0, 0.0, 1.0, 1.0])),
    ],
)
def test_normalize_boxes_clips(bbox, image_sizes, expected):
    out = general._normalize_boxes(bbox, image_sizes)
    assert np.allclose(out, expected)


@pytest.mark.parametrize(
    "bbox,resized,original",
    [
        (np.array([0.1, 0.1, 0.5, 0.5]), (640, 640), (480, 640)),
        (np.array([0.0, 0.0, 1.0, 1.0]), (640, 640), (320, 320)),
        (np.array([0.3, 0.2, 0.2, 0.2]), (1280, 1280), (720, 1280)),
    ],
)
def test_scale_letterbox_returns_valid_normalized_bbox(bbox, resized, original):
    out = general._scale_letterbox(bbox, resized, original)
    assert out.shape == (4,)
    assert np.all(out >= 0)
    assert np.all(out <= 1)
