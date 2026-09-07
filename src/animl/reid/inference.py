"""
Code to run Miew_ID and other re-identification models

(https://github.com/WildMeOrg/wbia-plugin-miew-id)

"""
from typing import Optional
import pandas as pd
import numpy as np

import onnxruntime as ort

from animl.utils.general import get_onnx_device
from animl.generator import ManifestGenerator

MIEWID_SIZE = 440


def load_miew(file_path: str,
              device: Optional[str] = None):
    """
    Load MiewID from file path

    Args:
        file_path (str): file path to model file
        device (str): device to load model to

    Returns:
        loaded miewid model object
    """
    device = get_onnx_device(user_set=device)
    print(f'Sending model to {device}')
    miew = ort.InferenceSession(file_path, providers=device)
    miew.model_type = 'miewid'
    return miew


def extract_miew_embeddings(miew_model,
                            manifest: pd.DataFrame,
                            file_col: str = "filepath",
                            batch_size: int = 4,
                            prefetch_size: int = 2,
                            num_workers: int = 2,
                            use_progress_bar: bool = False):
    """
    Wrapper for MiewID embedding extraction

    Args:
        miew_model: MiewID model object
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
                                       resize_width=MIEWID_SIZE,
                                       resize_height=MIEWID_SIZE,
                                       file_col=file_col,
                                       crop=True,
                                       normalize={"mean": [0.485, 0.456, 0.406],
                                                  "std": [0.229, 0.224, 0.225]},
                                       batch_size=batch_size,
                                       prefetch_size=prefetch_size,
                                       num_workers=num_workers,
                                       use_progress_bar=use_progress_bar)
        for batch_images, _, _, _ in dataloader:
            outputs = miew_model.run(None, {miew_model.get_inputs()[0].name: batch_images})
            output.extend(outputs)
        output = np.vstack(output)
    return output
