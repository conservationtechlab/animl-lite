"""
Code to run Miew_ID and other re-identification models

(https://github.com/WildMeOrg/wbia-plugin-miew-id)

"""
from typing import Optional
from tqdm import tqdm
import pandas as pd
import numpy as np

import onnxruntime as ort

from animl.utils.general import get_device
from animl.generator import manifest_dataloader

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
    if device is None:
        device = get_device()
    print(f'Sending model to {device}')
    miew = ort.InferenceSession(file_path, providers=["CPUExecutionProvider"])
    return miew


def extract_miew_embeddings(miew_model,
                            manifest: pd.DataFrame,
                            file_col: str = "filepath",
                            batch_size: int = 1,
                            num_workers: int = 1,
                            device: Optional[str] = None):
    """
    Wrapper for MiewID embedding extraction

    Args:
        miew_model: MiewID model object
        manifest (pd.DataFrame): dataframe with columns 'filepath', 'emb_id'
        file_col (str): column name for file paths in manifest
        batch_size (int): batch size for dataloader
        num_workers (int): number of workers for dataloader
        device (str): device to run model on

    Returns:
        output (np.ndarray): array of extracted embeddings
    """
    if device is None:
        device = get_device()

    if not {file_col}.issubset(manifest.columns):
        raise ValueError(f"DataFrame must contain '{file_col}' column.")

    output = []
    if isinstance(manifest, pd.DataFrame):

        dataloader = manifest_dataloader(manifest, batch_size=batch_size, num_workers=num_workers,
                                         file_col=file_col, crop=True, 
                                         normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                                         resize_width=MIEWID_SIZE,
                                         resize_height=MIEWID_SIZE,)
        for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            img = batch[0]
            inp = miew_model.get_inputs()[0]
            outputs = miew_model.run(None, {inp.name: img})[0]
            output.extend(outputs)
        output = np.vstack(output)
    return output
