"""
Generators and Dataloaders

Custom generators for training and inference

This version removes torch dependencies from the dataset and dataloader
so the module can be used without requiring PyTorch at runtime.
Images are returned as numpy arrays (C, H, W) with dtype float32 and
values scaled to [0, 1]. Batching is provided by a simple Python generator.
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional, Iterable
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile, ImageOps


from animl.file_management import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Letterbox:
    """
    Pads a crop to given size using PIL and resizes.

    Works on PIL.Image objects and returns a PIL.Image.

    Args:
        resize_height (int)
        resize_width (int)
    """
    def __init__(self, resize_height: int, resize_width: int) -> None:
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)

    def __call__(self, image: Image.Image) -> Image.Image:
        # PIL image size: (width, height)
        width, height = image.size
        target_w, target_h = self.resize_width, self.resize_height

        # If aspect ratios are already the same (rounded to 2 decimals), just resize
        if round((width / height), 2) == round((target_w / target_h), 2):
            return image.resize((target_w, target_h), Image.BILINEAR)

        # Compute required padding to achieve target aspect ratio, do center pad
        target_ar = target_w / target_h
        src_ar = width / height

        if src_ar < target_ar:
            # source narrower -> pad left/right
            new_width = int(target_ar * height)
            pad_total = max(0, new_width - width)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded = ImageOps.expand(image, border=(pad_left, 0, pad_right, 0), fill=0)
        else:
            # source wider -> pad top/bottom
            new_height = int(width / target_ar)
            pad_total = max(0, new_height - height)
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            padded = ImageOps.expand(image, border=(0, pad_top, 0, pad_bottom), fill=0)

        return padded.resize((target_w, target_h), Image.BILINEAR)


def pil_to_numpy_array(img: Image.Image) -> np.ndarray:
    """
    Convert PIL RGB image to numpy array with shape (C, H, W), dtype float32 scaled to [0,1].
    """
    arr = np.asarray(img, dtype=np.float32)  # H, W, C
    if arr.ndim == 2:  # grayscale -> replicate channels
        arr = np.stack([arr, arr, arr], axis=-1)
    # ensure RGB ordering; PIL.Image.convert("RGB") ensures this
    arr = arr.transpose(2, 0, 1)  # C, H, W
    arr = arr / 255.0
    return arr


class ManifestGenerator:
    '''
    Data generator that crops images on the fly, requires relative bbox coordinates,
    i.e. from MegaDetector.

    This class does NOT inherit from torch.utils.data.Dataset. It behaves like a
    sequence/iterable and can be indexed; it returns numpy arrays instead of torch.Tensors.
    '''
    def __init__(self, x: pd.DataFrame,
                 file_col: str = "filepath",
                 resize_height: int = 299,
                 resize_width: int = 299,
                 crop: bool = True,
                 crop_coord: str = 'relative',
                 normalize: bool = True,
                 letterbox: bool = False,
                 transform=None) -> None:
        self.x = x.reset_index(drop=True)
        self.file_col = file_col
        if self.file_col not in self.x.columns:
            raise ValueError(f"file_col '{self.file_col}' not found in dataframe columns")
        self.crop = crop
        if self.crop and not {'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'}.issubset(self.x.columns):
            raise ValueError("No bbox columns found for cropping")
        if crop_coord not in ['relative', 'absolute']:
            raise ValueError("crop_coord must be either 'relative' or 'absolute'")

        self.crop_coord = crop_coord
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.buffer = 0
        self.normalize = bool(normalize)
        self.letterbox = bool(letterbox)
        self.transform = transform  # optional callable(PIL.Image)->PIL.Image

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Optional[Tuple[np.ndarray, str, int, np.ndarray]]:
        row = self.x.iloc[idx]
        filepath = row[self.file_col]
        frame = int(row.get('frame', 0))
        ext = Path(str(filepath)).suffix.lower()

        # read image or video frame
        if ext in VIDEO_EXTENSIONS:
            img = self.extract_frames(idx, filepath)
            if img is None:
                return None
        elif ext in IMAGE_EXTENSIONS:
            try:
                img = Image.open(filepath).convert('RGB')
            except OSError:
                print(f"Image {filepath} cannot be opened. Skipping.")
                return None
        else:
            print(f"File {filepath} is not a video or image. Skipping.")
            return None

        width, height = img.size

        # maintain aspect ratio if one dimension is zero (fallbacks)
        if self.resize_width > 0 and self.resize_height <= 0:
            self.resize_height = int(width / height * self.resize_width)
        elif self.resize_width <= 0 and self.resize_height > 0:
            self.resize_width = int(height / width * self.resize_height)

        # cropping if requested
        if self.crop:
            bbox_x = float(row['bbox_x'])
            bbox_y = float(row['bbox_y'])
            bbox_w = float(row['bbox_w'])
            bbox_h = float(row['bbox_h'])

            if self.crop_coord == 'relative':
                left = width * bbox_x
                top = height * bbox_y
                right = width * (bbox_x + bbox_w)
                bottom = height * (bbox_y + bbox_h)
            else:
                left = bbox_x
                top = bbox_y
                right = bbox_x + bbox_w
                bottom = bbox_y + bbox_h

            left = max(0, int(left) - self.buffer)
            top = max(0, int(top) - self.buffer)
            right = min(width, int(right) + self.buffer)
            bottom = min(height, int(bottom) + self.buffer)
            img = img.crop((left, top, right, bottom))

        # apply transforms (PIL -> PIL) if provided, otherwise perform resize/letterbox
        if self.transform is not None:
            img = self.transform(img)
        else:
            if self.letterbox:
                img = Letterbox(self.resize_height, self.resize_width)(img)
            else:
                img = img.resize((self.resize_width, self.resize_height), Image.BILINEAR)

        img_arr = pil_to_numpy_array(img)  # C,H,W
        img.close()

        if not self.normalize:
            img_arr = img_arr * 255.0

        # return: img as numpy array (C,H,W), filepath str, frame int, shape np.array(height,width)
        return img_arr, str(filepath), int(frame), np.array((height, width), dtype=np.int32)

    def extract_frames(self, idx: int, filepath: str) -> Optional[Image.Image]:
        frame = int(self.x.loc[idx, 'frame'])
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():  # corrupted video
            print(f"Video {filepath} cannot be opened. Skipping.")
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame_img = cap.read()
        if not ret:
            print(f"Frame {frame} in video {filepath} cannot be read. Skipping.")
            cap.release()
            return None
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_img)
        cap.release()
        cv2.destroyAllWindows()
        return img


def manifest_dataloader(manifest: pd.DataFrame,
                        file_col: str = "filepath",
                        crop: bool = True,
                        crop_coord: str = 'relative',
                        normalize: bool = True,
                        letterbox: bool = False,
                        resize_height: int = 299,
                        resize_width: int = 299,
                        transform=None) -> Iterable[Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]]:
    '''
    Return a simple generator that yields batches of data as numpy arrays.

    Yields tuples: (batch_images, batch_filepaths, batch_frames, batch_hw)
      - batch_images: numpy array shape (B, C, H, W), dtype float32
      - batch_filepaths: list of file path strings length B
      - batch_frames: numpy array shape (B,) dtype int32
      - batch_hw: numpy array shape (B, 2) dtype int32 with (height, width)

    Notes:
      - This is a single-process generator. num_workers is intentionally ignored
        (no multiprocessing support) to avoid pulling in torch.
      - If a dataset item is None (skipped due to read errors), it will not be included.
    '''
    if crop is True and not any(manifest.columns.isin(["bbox_x"])):
        crop = False

    dataset_instance = ManifestGenerator(manifest,
                                         file_col=file_col,
                                         crop=crop,
                                         crop_coord=crop_coord,
                                         normalize=normalize, 
                                         letterbox=letterbox,
                                         resize_width=resize_width,
                                         resize_height=resize_height,
                                         transform=transform)
    def batch_generator():
        for i in range(len(dataset_instance)):
            item = dataset_instance[i]
            if item is None:
                continue
            img_arr, path, frame, hw = item  # img_arr: C,H,W
            batch_np = np.stack([img_arr], axis=0)  # B,C,H,W
            yield batch_np, [path], np.array([frame], dtype=np.int32), np.stack([hw], axis=0)

    return batch_generator()