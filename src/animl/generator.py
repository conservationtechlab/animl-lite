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
from typing import Tuple, Optional, Sequence, Generator
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile, ImageOps
from collections import deque
import threading
import queue

from animl.file_management import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from animl.utils.general import SDZWA_CLASSIFIER_SIZE

ImageFile.LOAD_TRUNCATED_IMAGES = True

# end of the batches, from the prefetch worker
_END = object()


class _WorkerError:
    '''An exception raised in the prefetch worker, re-raised by the consumer'''
    def __init__(self, error: Exception) -> None:
        self.error = error


class ManifestGenerator:
    '''
    Data generator that crops images on the fly, requires relative bbox coordinates,
    i.e. from MegaDetector.

    This class does NOT inherit from torch.utils.data.Dataset. It behaves like a
    sequence/iterable and can be indexed; it returns numpy arrays instead of torch.Tensors.
    '''
    def __init__(self, 
                 manifest: pd.DataFrame,
                 resize_height: int = SDZWA_CLASSIFIER_SIZE,
                 resize_width: int = SDZWA_CLASSIFIER_SIZE,
                 file_col: str = "filepath",
                 crop: bool = True,
                 crop_coord: str = 'relative',
                 normalize: bool = True,
                 letterbox: bool = False,
                 batch_size: int = 16,
                 prefetch_size: int = 2,
                 num_workers: int = 2,
                 dtype: np.dtype = np.float32,
                 use_progress_bar: bool = False,
                 return_indices: bool = False) -> None:

        self.manifest = manifest.reset_index(drop=True)
        self.file_col = file_col
        self.batch_size = batch_size
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.crop = crop
        self.crop_coord = crop_coord
        self.normalize = normalize
        self.letterbox = bool(letterbox)
        self.prefetch_size = max(1, prefetch_size)
        self.num_workers = max(0, num_workers)
        self.dtype = dtype
        self.use_progress_bar = use_progress_bar
        # items that can't be loaded are left out of their batch; with return_indices,
        # each batch also gives the manifest positions of the items it holds
        self.return_indices = return_indices
        
        self._validate_config()
        
        # Memory pools for reuse
        self._img_pool = deque(maxlen=self.prefetch_size * 2)
        self._batch_queue = queue.Queue(maxsize=self.prefetch_size)
        self._stop_event = threading.Event()
        self._prefetch_thread = None


    def _validate_config(self) -> None:
        '''Validate file column and cropping configuration'''
    
        if self.file_col not in self.manifest.columns:
            raise ValueError(f"file_col '{self.file_col}' not found in dataframe columns")
        
        if self.crop and not {'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'}.issubset(self.manifest.columns):
            print("Bbox columns not found; disabling cropping")
            self.crop = False
        
        if self.crop_coord not in ['relative', 'absolute']:
            raise ValueError("crop_coord must be 'relative' or 'absolute'")
        
        if 'frame' not in self.manifest.columns:
            self.manifest['frame'] = 0

    def __len__(self) -> int:
        '''Number of batches.'''
        return (len(self.manifest) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Generator:
        '''Iterate over batches with optional async prefetching.'''
        if self.num_workers > 0:
            return self._iter_with_prefetch()
        else:
            return self._iter_single_threaded()
        
    def _iter_single_threaded(self) -> Generator:
        '''Single-threaded batch iteration.'''
        try:
            iterator = range(len(self.manifest))
            if self.use_progress_bar:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(iterator, desc="Loading batches", unit="batch")
                except ImportError:
                    pass
            
            batch_indices = []
            for idx in iterator:
                batch_indices.append(idx)
                
                if len(batch_indices) == self.batch_size or idx == len(self.manifest) - 1:
                    batch = self._load_batch(batch_indices)
                    if batch is not None:
                        yield batch
                    batch_indices = []
        except Exception as e:
            print(f"Error during iteration: {e}")
            raise
    
    def _iter_with_prefetch(self) -> Generator:
        '''Multi-threaded iteration with async prefetching.'''
        # a fresh queue, so batches left over from an earlier, interrupted iteration aren't yielded
        self._batch_queue = queue.Queue(maxsize=self.prefetch_size)
        self._stop_event.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True
        )
        self._prefetch_thread.start()

        progress = None
        if self.use_progress_bar:
            try:
                from tqdm import tqdm
                progress = tqdm(total=len(self), desc="Loading batches", unit="batch")
            except ImportError:
                pass

        try:
            while True:
                # wait as long as the worker is running: a slow batch (large images,
                # videos, network drives) must not end the iteration early
                try:
                    batch = self._batch_queue.get(timeout=1)
                except queue.Empty:
                    if not self._prefetch_thread.is_alive():
                        raise RuntimeError("Prefetch worker stopped without finishing")
                    continue
                if batch is _END:
                    break
                if isinstance(batch, _WorkerError):
                    raise batch.error
                if progress is not None:
                    progress.update(1)
                yield batch
        finally:
            self._stop_event.set()
            if progress is not None:
                progress.close()
            if self._prefetch_thread:
                self._prefetch_thread.join(timeout=5)

    def _prefetch_worker(self) -> None:
        '''Worker thread that prefetches batches.'''
        try:
            batch_indices = []
            for idx in range(len(self.manifest)):
                if self._stop_event.is_set():
                    return

                batch_indices.append(idx)

                if len(batch_indices) == self.batch_size or idx == len(self.manifest) - 1:
                    batch = self._load_batch(batch_indices)
                    if batch is not None and not self._put(batch):
                        return
                    batch_indices = []

            self._put(_END)
        except Exception as e:
            # hand the error to the consumer instead of ending the iteration as if it had finished
            self._put(_WorkerError(e))

    def _put(self, item) -> bool:
        '''Queue an item for the consumer; False if the iteration was stopped meanwhile.'''
        while not self._stop_event.is_set():
            try:
                self._batch_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _load_batch(self, indices: list) -> Optional[Tuple]:
        '''Load a batch of samples.'''
        batch_images = []
        batch_filepaths = []
        batch_frames = []
        batch_hw = []
        batch_indices = []
        
        for idx in indices:
            item = self._load_item(idx)
            if item is not None:
                img_arr, filepath, frame, hw = item
                batch_indices.append(idx)
                batch_images.append(img_arr)
                batch_filepaths.append(filepath)
                batch_frames.append(frame)
                batch_hw.append(hw)
        
        if not batch_images:
            return None
        
        # Stack into batch arrays
        batch_np = np.stack(batch_images, axis=0)  # (B, C, H, W)
        batch_frames_arr = np.array(batch_frames, dtype=np.int32)
        batch_hw_arr = np.stack(batch_hw, axis=0)
        
        if self.return_indices:
            return batch_np, batch_filepaths, batch_frames_arr, batch_hw_arr, np.array(batch_indices, dtype=np.int64)
        return batch_np, batch_filepaths, batch_frames_arr, batch_hw_arr
    
    def _load_item(self, idx: int) -> Optional[Tuple]:
        '''Load a single item from dataset.'''
        try:
            row = self.manifest.iloc[idx]
            filepath = row[self.file_col]
            frame = int(row.get('frame', 0))
            ext = Path(str(filepath)).suffix.lower()
            
            # Read image or video frame
            if ext in VIDEO_EXTENSIONS:
                img = self._extract_frame(filepath, frame)
                if img is None:
                    return None
            elif ext in IMAGE_EXTENSIONS:
                try:
                    img = Image.open(filepath).convert('RGB')
                except OSError as e:
                    print(f"Cannot open image {filepath}: {e}")
                    return None
            else:
                print(f"Unsupported file type: {filepath}")
                return None
            
            width, height = img.size
            
            # Cropping if requested
            if self.crop:
                img = self._apply_crop(img, row, width, height)
            
            # Resizing
            target_w, target_h = self._target_size(*img.size)
            if self.letterbox:
                img = self._letterbox_resize(img, target_w, target_h)
            else:
                img = img.resize((target_w, target_h), Image.BILINEAR)
            
            # Convert to numpy array
            img_arr = self._pil_to_numpy(img)
            img.close()
            
            # Normalize
            if isinstance(self.normalize, dict):
                img_arr = self._normalize(
                    img_arr,
                    mean=self.normalize.get("mean", [0.485, 0.456, 0.406]),
                    std=self.normalize.get("std", [0.229, 0.224, 0.225])
                )
            elif self.normalize is False:
                img_arr = img_arr * 255.0
            
            return img_arr, str(filepath), int(frame), np.array((height, width), dtype=np.int32)
        
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return None
    
    def _extract_frame(self, filepath: str, frame_num: int) -> Optional[Image.Image]:
        '''Extract frame from video file.'''
        try:
            cap = cv2.VideoCapture(str(filepath))
            if not cap.isOpened():
                print(f"Cannot open video: {filepath}")
                return None
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame_img = cap.read()
            cap.release()
            
            if not ret:
                print(f"Cannot read frame {frame_num} from {filepath}")
                return None
            
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_img)
        except Exception as e:
            print(f"Video extraction error: {e}")
            return None
    
    def _apply_crop(self, img: Image.Image, row: pd.Series, 
                    width: int, height: int) -> Image.Image:
        '''Apply bounding box crop to image.'''
        bbox_x = float(row.get('bbox_x', 0))
        bbox_y = float(row.get('bbox_y', 0))
        bbox_w = float(row.get('bbox_w', 1))
        bbox_h = float(row.get('bbox_h', 1))
        
        if self.crop_coord == 'relative':
            left = width * bbox_x
            top = height * bbox_y
            right = width * (bbox_x + bbox_w)
            bottom = height * (bbox_y + bbox_h)
        else:
            left, top, right, bottom = bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h
        
        # Add buffer and clip
        buffer = 0
        left = max(0, int(left) - buffer)
        top = max(0, int(top) - buffer)
        right = min(width, int(right) + buffer)
        bottom = min(height, int(bottom) + buffer)
        
        return img.crop((left, top, right, bottom))
    
    def _target_size(self, width: int, height: int) -> Tuple[int, int]:
        '''
        Output size for an image of the given size

        If one of resize_width / resize_height is 0 or less, it follows the image's
        aspect ratio. Images of different aspect ratios then get different sizes,
        which can only be batched with batch_size=1.
        '''
        if self.resize_width > 0 and self.resize_height <= 0:
            return self.resize_width, max(1, round(self.resize_width * height / width))
        if self.resize_width <= 0 and self.resize_height > 0:
            return max(1, round(self.resize_height * width / height)), self.resize_height
        return self.resize_width, self.resize_height

    def _letterbox_resize(self, img: Image.Image,
                          target_w: Optional[int] = None,
                          target_h: Optional[int] = None) -> Image.Image:
        '''Resize with letterboxing to maintain aspect ratio.'''
        width, height = img.size
        target_w = self.resize_width if target_w is None else target_w
        target_h = self.resize_height if target_h is None else target_h
        
        if round((width / height), 2) == round((target_w / target_h), 2):
            return img.resize((target_w, target_h), Image.BILINEAR)
        
        target_ar = target_w / target_h
        src_ar = width / height
        
        if src_ar < target_ar:
            new_width = int(target_ar * height)
            pad_total = max(0, new_width - width)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded = ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=0)
        else:
            new_height = int(width / target_ar)
            pad_total = max(0, new_height - height)
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            padded = ImageOps.expand(img, border=(0, pad_top, 0, pad_bottom), fill=0)
        
        return padded.resize((target_w, target_h), Image.BILINEAR)
    
    def _pil_to_numpy(self, img: Image.Image) -> np.ndarray:
        '''Convert PIL image to numpy (C, H, W) float32 [0, 1].'''
        arr = np.asarray(img, dtype=self.dtype)  # (H, W, C)
        if arr.ndim == 2:  # Grayscale
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr.transpose(2, 0, 1) / 255.0  # (C, H, W), [0, 1]
        return arr
    
    def _normalize(self, img: np.ndarray, 
                   mean: Sequence[float], 
                   std: Sequence[float]) -> np.ndarray:
        '''Normalize image with (img - mean) / std.'''
        mean = np.array(mean, dtype=self.dtype).reshape(3, 1, 1)
        std = np.array(std, dtype=self.dtype).reshape(3, 1, 1)
        return (img - mean) / std