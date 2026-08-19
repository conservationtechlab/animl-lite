from __future__ import annotations

import numpy as np
import pytest

from animl.reid.distance import (
    compute_batched_distance_matrix,
    compute_distance_matrix,
    cosine_distance,
    euclidean_squared_distance,
    remove_diagonal,
)


def test_remove_diagonal_shape_and_values():
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = remove_diagonal(mat)
    assert out.shape == (3, 2)
    assert out.tolist() == [[2, 3], [4, 6], [7, 8]]


@pytest.mark.parametrize(
    "bad_input",
    [
        np.array([1, 2, 3]),
        np.array([[[1]]]),
    ],
)
def test_remove_diagonal_requires_2d_square_array(bad_input):
    with pytest.raises(ValueError):
        remove_diagonal(bad_input)


def test_remove_diagonal_requires_square_matrix():
    with pytest.raises(ValueError):
        remove_diagonal(np.zeros((2, 3)))


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 2.0]]),
        ),
        (
            np.array([[1.0, 1.0]]),
            np.array([[2.0, 1.0], [1.0, 3.0]]),
            np.array([[1.0, 4.0]]),
        ),
    ],
)
def test_euclidean_squared_distance_values(x, y, expected):
    out = euclidean_squared_distance(x, y)
    assert np.allclose(out, expected)


@pytest.mark.parametrize(
    "x,y",
    [
        (np.array([1, 2]), np.array([[1, 2]])),
        (np.array([[1, 2]]), np.array([1, 2])),
        (np.array([[1, 2, 3]]), np.array([[1, 2]])),
    ],
)
def test_euclidean_squared_distance_validates_shapes(x, y):
    with pytest.raises(ValueError):
        euclidean_squared_distance(x, y)


def test_cosine_distance_identity_and_orthogonal_cases():
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = cosine_distance(x, x)
    assert np.allclose(np.diag(out), 0.0)
    assert np.isclose(out[0, 1], 1.0)


def test_cosine_distance_handles_zero_vectors():
    x = np.array([[0.0, 0.0]])
    y = np.array([[1.0, 0.0]])
    out = cosine_distance(x, y)
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    "x,y",
    [
        (np.array([1, 2]), np.array([[1, 2]])),
        (np.array([[1, 2]]), np.array([1, 2])),
        (np.array([[1, 2, 3]]), np.array([[1, 2]])),
    ],
)
def test_cosine_distance_validates_shapes(x, y):
    with pytest.raises(ValueError):
        cosine_distance(x, y)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_compute_distance_matrix_dispatches(metric):
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([[1.0, 0.0]])
    out = compute_distance_matrix(x, y, metric=metric)
    assert out.shape == (2, 1)


def test_compute_distance_matrix_unknown_metric_raises():
    with pytest.raises(ValueError):
        compute_distance_matrix(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]), metric="manhattan")


@pytest.mark.parametrize(
    "x,y",
    [
        (np.array([1, 2]), np.array([[1, 2]])),
        (np.array([[1, 2]]), np.array([1, 2])),
        (np.array([[1, 2, 3]]), np.array([[1, 2]])),
    ],
)
def test_compute_distance_matrix_validates_shapes(x, y):
    with pytest.raises(ValueError):
        compute_distance_matrix(x, y)


def test_compute_batched_distance_matrix_matches_non_batched_euclidean():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0, 0.0], [0.0, 1.0]])
    full = compute_distance_matrix(x, y, metric="euclidean")
    batched = compute_batched_distance_matrix(x, y, metric="euclidean", batch_size=2)
    assert np.allclose(full, batched)


def test_compute_batched_distance_matrix_matches_non_batched_cosine():
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([[1.0, 0.0], [1.0, 1.0]])
    full = compute_distance_matrix(x, y, metric="cosine")
    batched = compute_batched_distance_matrix(x, y, metric="cosine", batch_size=1)
    assert np.allclose(full, batched)


@pytest.mark.parametrize(
    "batch_size",
    [0, -1],
)
def test_compute_batched_distance_matrix_requires_positive_batch_size(batch_size):
    with pytest.raises(ValueError):
        compute_batched_distance_matrix(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]), batch_size=batch_size)


@pytest.mark.parametrize(
    "x,y",
    [
        (np.array([1, 2]), np.array([[1, 2]])),
        (np.array([[1, 2]]), np.array([1, 2])),
        (np.array([[1, 2, 3]]), np.array([[1, 2]])),
    ],
)
def test_compute_batched_distance_matrix_validates_shapes(x, y):
    with pytest.raises(ValueError):
        compute_batched_distance_matrix(x, y)
