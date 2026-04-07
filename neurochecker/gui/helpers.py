from typing import Any, Dict, Optional

import numpy as np
from PyQt5 import QtGui
from scipy.ndimage import binary_erosion

from neurochecker.gui.constants import MASK_OUTLINE_THICKNESS


class QImageWithBuffer(QtGui.QImage):
    def __init__(self, buffer: np.ndarray, width: int, height: int, bytes_per_line: int, fmt: QtGui.QImage.Format):
        self._buffer = buffer
        super().__init__(buffer.data, width, height, bytes_per_line, fmt)


def numpy_to_qimage(image: np.ndarray) -> QtGui.QImage:
    if image.ndim == 2:
        h, w = image.shape
        buffer = np.ascontiguousarray(image)
        return QImageWithBuffer(buffer, w, h, w, QtGui.QImage.Format_Grayscale8)
    if image.ndim == 3:
        h, w, c = image.shape
        buffer = np.ascontiguousarray(image)
        fmt = QtGui.QImage.Format_RGB888 if c == 3 else QtGui.QImage.Format_RGBA8888
        return QImageWithBuffer(buffer, w, h, buffer.strides[0], fmt)
    raise ValueError(f"Unsupported image shape {image.shape}")


def _mask_outline(mask: np.ndarray, *, thickness: int = MASK_OUTLINE_THICKNESS) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    mask_bool = mask.astype(bool, copy=False)
    if not mask_bool.any():
        return mask_bool
    thickness = max(1, int(thickness))
    eroded = binary_erosion(
        mask_bool,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
        iterations=thickness,
    )
    return mask_bool & ~eroded


def _flag_point_record(
    segment_id: int,
    frame: int,
    *,
    step: Optional[int] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    z: Optional[float] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {"segment": int(segment_id), "frame": int(frame)}
    if step is not None:
        record["step"] = int(step)
    if x is not None and y is not None and z is not None:
        record["x"] = float(x)
        record["y"] = float(y)
        record["z"] = float(z)
    return record
