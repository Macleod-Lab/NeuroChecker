import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from PyQt5 import QtCore

from neurochecker.mask_io import MaskEntry


class ImageSampler:
    def __init__(self, images_dir: Path, neurotracer_root: Optional[Path] = None) -> None:
        self.images_dir = Path(images_dir)
        self.neurotracer_root = neurotracer_root
        self.sequence = None
        self.source_sequence = None
        self.has_pyramid = False
        self.original_width = 0
        self.original_height = 0

    def open(self) -> None:
        if self.neurotracer_root and str(self.neurotracer_root) not in sys.path:
            sys.path.insert(0, str(self.neurotracer_root))
        try:
            from viewer.image_sequence import ImageSequence
            from viewer.tiled_image_sequence import TiledImageSequence
        except Exception as exc:
            raise RuntimeError("Could not import NeuroTracer viewer modules.") from exc
        self.source_sequence = ImageSequence.from_directory(self.images_dir)
        self.sequence = TiledImageSequence(self.images_dir)
        self.has_pyramid = bool(getattr(self.sequence, "has_pyramid", False))
        self.original_width = int(self.sequence.original_width)
        self.original_height = int(self.sequence.original_height)

    def get_viewport_image(
        self,
        frame: int,
        viewport_rect: Tuple[int, int, int, int],
        target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, float]:
        if self.sequence is None:
            raise RuntimeError("Image sampler not initialized.")
        return self.sequence.get_viewport_image(frame, viewport_rect, target_size)

    def get_source_viewport_image(
        self,
        frame: int,
        viewport_rect: Tuple[int, int, int, int],
        target_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, float]:
        if self.source_sequence is None:
            raise RuntimeError("Source image sequence not initialized.")
        full_frame = self.source_sequence.load_frame(frame)
        vx, vy, vw, vh = viewport_rect
        tw, th = target_size
        h, w = full_frame.shape[:2]
        vx = max(0, min(int(vx), max(0, w - 1)))
        vy = max(0, min(int(vy), max(0, h - 1)))
        vw = max(1, min(int(vw), w - vx))
        vh = max(1, min(int(vh), h - vy))
        region = full_frame[vy:vy + vh, vx:vx + vw]
        if region.size == 0:
            return np.zeros((th, tw, 3), dtype=np.uint8), 1.0
        if (vw, vh) != (tw, th):
            if region.ndim == 2:
                pil_img = Image.fromarray(region, mode="L")
            else:
                pil_img = Image.fromarray(region)
            region = np.asarray(pil_img.resize((tw, th), Image.Resampling.LANCZOS))
        if region.ndim == 2:
            region = np.stack([region, region, region], axis=-1)
        return region, vw / max(1, tw)

    def shutdown(self) -> None:
        if self.sequence is not None:
            self.sequence.shutdown()
        if self.source_sequence is not None:
            self.source_sequence.clear_cache()


class MaskCache:
    def __init__(self, capacity: int = 12) -> None:
        self.capacity = capacity
        self._cache: "OrderedDict[int, List[Tuple[MaskEntry, np.ndarray]]]" = OrderedDict()

    def get(self, frame: int) -> Optional[List[Tuple[MaskEntry, np.ndarray]]]:
        if frame not in self._cache:
            return None
        self._cache.move_to_end(frame)
        return self._cache[frame]

    def set(self, frame: int, masks: List[Tuple[MaskEntry, np.ndarray]]) -> None:
        self._cache[frame] = masks
        self._cache.move_to_end(frame)
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


@dataclass
class ComponentInfo:
    entry: MaskEntry
    label_id: int
    labeled: np.ndarray
    slices: Tuple[slice, slice]
    segment_id: int
    centroid: Tuple[float, float]
    area: int


class ComponentCache:
    def __init__(self, capacity: int = 64) -> None:
        self.capacity = capacity
        self._cache: "OrderedDict[int, List[ComponentInfo]]" = OrderedDict()

    def get(self, frame: int) -> Optional[List[ComponentInfo]]:
        if frame not in self._cache:
            return None
        self._cache.move_to_end(frame)
        return self._cache[frame]

    def set(self, frame: int, items: List[ComponentInfo]) -> None:
        self._cache[frame] = items
        self._cache.move_to_end(frame)
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
