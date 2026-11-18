"""
Code to run Miew_ID and other re-identification models

(https://github.com/WildMeOrg/wbia-plugin-miew-id)

"""
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

import onnxruntime as ort

from animl.utils.general import get_onnx_device
from animl.generator import ManifestGenerator


MIEWID_SIZE = 440
MIEWID_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

MEGADESCRIPTOR_L_SIZE = 384
MEGADESCRIPTOR_L_NORMALIZE = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


def load_ried(model_path: str,
              model_type: str,
              device: Optional[str] = None):
    """
    Load re-identification model from file path

    Args:
        model_path (str): file path to model file
        device (str): device to load model to
        model_type (str): type of the re-identification model (e.g., "miewid", "megadescriptor_l")

    Returns:
        loaded re-identification model object
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    providers = get_onnx_device(user_set=device)
    model = ort.InferenceSession(model_path, providers=providers)
    model.model_type = model_type.lower()
    return model


def extract_reid_embeddings(model,
                            manifest: pd.DataFrame,
                            resize: int,
                            normalize: dict = None,
                            file_col: str = "filepath",
                            batch_size: int = 1,
                            prefetch_size: int = 2,
                            num_workers: int = 2,
                            use_progress_bar: bool = False):
    """
    Wrapper for re-identification embedding extraction

    Args:
        model: re-identification model object
        manifest (pd.DataFrame): dataframe with columns 'filepath', 'emb_id'
        file_col (str): column name for file paths in manifest
        batch_size (int): number of images per batch
        prefetch_size (int): number of batches to prefetch
        num_workers (int): number of worker threads for data loading
        use_progress_bar (bool): whether to display a progress bar

    Returns:
        output (np.ndarray): array of extracted embeddings
    """
    if not {file_col}.issubset(manifest.columns):
        raise ValueError(f"DataFrame must contain '{file_col}' column.")

    output = []
    if isinstance(manifest, pd.DataFrame):

        dataloader = ManifestGenerator(manifest,
                                       resize_width=resize,
                                       resize_height=resize,
                                       file_col=file_col,
                                       crop=True,
                                       normalize=normalize,
                                       batch_size=batch_size,
                                       prefetch_size=prefetch_size,
                                       num_workers=num_workers,
                                       use_progress_bar=use_progress_bar)
        for batch_images, _, _, _ in dataloader:
            outputs = model.run(None, {model.get_inputs()[0].name: batch_images})
            output.extend(outputs)
        output = np.vstack(output)
    return output


def remove_diagonal(A: np.ndarray) -> np.ndarray:
    """
    Removes the diagonal elements from a square matrix.

    Args:
        A (np.ndarray): Input square matrix of shape (N, N).

    Returns:
        np.ndarray: Matrix with diagonal elements removed, shape (N, N-1).

    Raises:
        ValueError: if input is not a 2-D square array.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {A.ndim}-D")
    n, m = A.shape
    if n != m:
        raise ValueError("Input must be a square matrix")

    # create mask that is True for non-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    return A[mask].reshape(n, n - 1)


def euclidean_squared_distance(input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Euclidean squared distances between rows of input1 and input2.

    Args:
        input1 (np.ndarray): shape (M, D)
        input2 (np.ndarray): shape (N, D)

    Returns:
        np.ndarray: distance matrix shape (M, N) where entry (i,j) is ||input1[i] - input2[j]||^2
    """
    x = np.asarray(input1, dtype=np.float64)
    y = np.asarray(input2, dtype=np.float64)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Both inputs must be 2-D arrays")
    if x.shape[1] != y.shape[1]:
        raise ValueError("The feature dimension (number of columns) must match")

    # Use the identity: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    x_sq = np.sum(x * x, axis=1).reshape(-1, 1)   # (M,1)
    y_sq = np.sum(y * y, axis=1).reshape(1, -1)   # (1,N)
    cross = x.dot(y.T)                            # (M,N)
    dist = x_sq + y_sq - 2.0 * cross

    # Numerical errors can make tiny negative values; clamp to zero
    np.maximum(dist, 0.0, out=dist)
    return dist


def cosine_distance(input1: np.ndarray, input2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes pairwise cosine distance: 1 - cosine_similarity

    Args:
        input1 (np.ndarray): shape (M, D)
        input2 (np.ndarray): shape (N, D)
        eps (float): small constant to avoid divide-by-zero for zero vectors

    Returns:
        np.ndarray: distance matrix shape (M, N) where entry (i,j) = 1 - (x_i . y_j) / (||x_i|| * ||y_j||)
    """
    x = np.asarray(input1, dtype=np.float64)
    y = np.asarray(input2, dtype=np.float64)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Both inputs must be 2-D arrays")
    if x.shape[1] != y.shape[1]:
        raise ValueError("The feature dimension (number of columns) must match")

    # normalize rows
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)

    # avoid division by zero
    x_norm = np.maximum(x_norm, eps)
    y_norm = np.maximum(y_norm, eps)

    x_unit = x / x_norm
    y_unit = y / y_norm

    sim = x_unit.dot(y_unit.T)  # cosine similarity in [-1,1]
    # Clip for numerical stability
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def compute_distance_matrix(input1, input2, metric: str = 'euclidean') -> np.ndarray:
    """
    A wrapper function for computing the distance matrix.

    Args:
        input1: array-like shape (M, D)
        input2: array-like shape (N, D)
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        np.ndarray: distance matrix shape (M, N)
    """
    x = np.asarray(input1)
    y = np.asarray(input2)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Both inputs must be 2-D arrays")
    if x.shape[1] != y.shape[1]:
        raise ValueError("The feature dimension (number of columns) must match")

    if metric == 'euclidean':
        return euclidean_squared_distance(x, y)
    elif metric == 'cosine':
        return cosine_distance(x, y)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )


def compute_batched_distance_matrix(input1: np.ndarray,
                                    input2: np.ndarray,
                                    metric: str = 'cosine',
                                    batch_size: int = 10) -> np.ndarray:
    """
    Computes the distance matrix in a batched manner to save memory.

    Args:
        input1 (np.ndarray): 2-D array of query features (M, D).
        input2 (np.ndarray): 2-D array of database features (N, D).
        metric (str): The distance metric to use: 'euclidean' or 'cosine'.
        batch_size (int): The number of rows from input1 to process at a time.

    Returns:
        np.ndarray: The computed distance matrix of shape (M, N).
    """
    x = np.asarray(input1, dtype=np.float64)
    y = np.asarray(input2, dtype=np.float64)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Both inputs must be 2-D arrays")
    if x.shape[1] != y.shape[1]:
        raise ValueError("The feature dimension (number of columns) must match")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    m = x.shape[0]
    num_batches = int(np.ceil(m / batch_size))
    blocks = []
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, m)
        batch = x[start:end]
        block = compute_distance_matrix(batch, y, metric=metric)
        blocks.append(block)
    return np.vstack(blocks)
