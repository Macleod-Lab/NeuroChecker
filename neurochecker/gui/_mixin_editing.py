import json
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets
from scipy.ndimage import (
    label as nd_label,
    binary_fill_holes,
    binary_opening,
    binary_closing,
)

from neurochecker.gui.constants import logger
from neurochecker.mask_io import MaskEntry, load_mask_array


class EditingMixin:
    def _toggle_edit_shortcut(self) -> None:
        if not hasattr(self, "edit_mode_check"):
            return
        self.edit_mode_check.setChecked(not self.edit_mode_check.isChecked())

    def _toggle_edit_mode(self, state: int) -> None:
        self._edit_mode = state == QtCore.Qt.Checked
        cursor = QtCore.Qt.CrossCursor if self._edit_mode else QtCore.Qt.ArrowCursor
        self.view.setCursor(cursor)
        if not self._edit_mode and self._edit_dirty:
            self._commit_edit_mask()

    def _set_brush_mode(self) -> None:
        if hasattr(self, "edit_brush_add_btn") and self.edit_brush_add_btn.isChecked():
            self._edit_brush_add = True
        elif hasattr(self, "edit_brush_erase_btn") and self.edit_brush_erase_btn.isChecked():
            self._edit_brush_add = False

    def _set_brush_size(self, value: int) -> None:
        self._edit_brush_size = max(1, int(value))

    def _brush_mask(self, radius: int) -> np.ndarray:
        radius = max(1, int(radius))
        cached = self._brush_cache.get(radius)
        if cached is not None:
            return cached
        size = radius * 2 + 1
        yy, xx = np.ogrid[:size, :size]
        cy = radius
        cx = radius
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
        self._brush_cache[radius] = mask
        return mask

    def _ensure_mask_bounds(
        self, entry: MaskEntry, mask: np.ndarray, x_px: int, y_px: int, radius: int
    ) -> np.ndarray:
        h, w = mask.shape[:2]
        cx = int(round(x_px - entry.x))
        cy = int(round(y_px - entry.y))
        r = max(1, int(radius))
        pad_left = max(0, r - cx) if cx - r < 0 else 0
        pad_top = max(0, r - cy) if cy - r < 0 else 0
        pad_right = max(0, cx + r - (w - 1)) if cx + r >= w else 0
        pad_bottom = max(0, cy + r - (h - 1)) if cy + r >= h else 0
        full_w = entry.full_width if entry.full_width > 0 else None
        full_h = entry.full_height if entry.full_height > 0 else None
        if pad_left and entry.x - pad_left < 0:
            pad_left = entry.x
        if pad_top and entry.y - pad_top < 0:
            pad_top = entry.y
        if full_w is not None:
            max_right = full_w - (entry.x - pad_left) - w
            pad_right = max(0, min(pad_right, max_right))
        if full_h is not None:
            max_bottom = full_h - (entry.y - pad_top) - h
            pad_bottom = max(0, min(pad_bottom, max_bottom))
        if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
            return mask
        new_h = h + pad_top + pad_bottom
        new_w = w + pad_left + pad_right
        new_mask = np.zeros((new_h, new_w), dtype=mask.dtype)
        new_mask[pad_top : pad_top + h, pad_left : pad_left + w] = mask
        entry.x -= pad_left
        entry.y -= pad_top
        entry.width = new_w
        entry.height = new_h
        if entry.run_id in self.run_stats:
            stats = self.run_stats[entry.run_id]
            stats.max_width = max(stats.max_width, new_w)
            stats.max_height = max(stats.max_height, new_h)
        if self._edit_active_frame is not None:
            cached = self.mask_cache.get(self._edit_active_frame)
            if cached is not None:
                for idx, (e, _mask) in enumerate(cached):
                    if e.path == entry.path:
                        cached[idx] = (entry, new_mask)
                        break
                self.mask_cache.set(self._edit_active_frame, cached)
        self._edit_index_dirty = True
        return new_mask

    def _event_to_full_coords(self, event: QtCore.QEvent) -> Optional[Tuple[int, int, int]]:
        if self._last_view_context is None:
            return None
        frame, viewport_rect, target_size, _image, scale_factor = self._last_view_context
        pos = event.pos()
        scene_pos = self.view.mapToScene(pos)
        tx = float(scene_pos.x())
        ty = float(scene_pos.y())
        tw, th = target_size
        if tx < 0 or ty < 0 or tx > tw or ty > th:
            return None
        vx, vy, _vw, _vh = viewport_rect
        fx = vx + tx * scale_factor
        fy = vy + ty * scale_factor
        return frame, int(round(fx)), int(round(fy))

    def _pick_edit_entry(self, frame: int, x_px: float, y_px: float) -> Optional[MaskEntry]:
        comps = self._get_components_for_frame(frame)
        if not comps:
            return None
        if self.active_segment_id is not None:
            seg_id = int(self.active_segment_id)
            seg_comps = [c for c in comps if c.segment_id == seg_id]
            if seg_comps:
                comps = seg_comps
        containing = []
        for comp in comps:
            entry = comp.entry
            lx = int(round(x_px - entry.x))
            ly = int(round(y_px - entry.y))
            if lx < 0 or ly < 0:
                continue
            if ly >= comp.labeled.shape[0] or lx >= comp.labeled.shape[1]:
                continue
            if comp.labeled[ly, lx] == comp.label_id:
                containing.append(comp)
        candidates = containing if containing else comps
        best = None
        best_dist = None
        for comp in candidates:
            cx, cy = comp.centroid
            dx = cx - x_px
            dy = cy - y_px
            dist = dx * dx + dy * dy
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = comp
        return best.entry if best is not None else None

    def _get_mask_for_entry(self, frame: int, entry: MaskEntry) -> Optional[np.ndarray]:
        masks = self._get_masks_for_frame(frame)
        for e, mask in masks:
            if e.path == entry.path:
                return mask
        mask = load_mask_array(entry)
        if mask is None:
            return None
        masks.append((entry, mask))
        self.mask_cache.set(frame, masks)
        return mask

    def _apply_brush(self, mask: np.ndarray, entry: MaskEntry, x_px: int, y_px: int, add: bool) -> None:
        mask = self._ensure_mask_bounds(entry, mask, x_px, y_px, self._edit_brush_size)
        if self._edit_active_entry is entry:
            self._edit_active_mask = mask
        h, w = mask.shape[:2]
        cx = int(round(x_px - entry.x))
        cy = int(round(y_px - entry.y))
        r = max(1, int(self._edit_brush_size))
        if cx < -r or cy < -r or cx >= w + r or cy >= h + r:
            return
        y0 = max(0, cy - r)
        y1 = min(h - 1, cy + r)
        x0 = max(0, cx - r)
        x1 = min(w - 1, cx + r)
        brush = self._brush_mask(r)
        by0 = y0 - (cy - r)
        bx0 = x0 - (cx - r)
        sub_brush = brush[by0 : by0 + (y1 - y0 + 1), bx0 : bx0 + (x1 - x0 + 1)]
        sub_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
        if add:
            sub_mask[sub_brush] = 1
        else:
            sub_mask[sub_brush] = 0

    def _apply_edit_from_event(self, event: QtCore.QEvent) -> None:
        coords = self._event_to_full_coords(event)
        if coords is None:
            return
        frame, x_px, y_px = coords
        if self._edit_active_entry is None or self._edit_active_frame != frame:
            if not self._ensure_edit_mask_for_frame(frame, x_px=x_px, y_px=y_px):
                return
        if self._edit_active_mask is None or self._edit_active_entry is None:
            return
        add = self._edit_brush_add
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            add = not add
        if self._edit_last_pos is None:
            self._apply_brush(self._edit_active_mask, self._edit_active_entry, x_px, y_px, add)
        else:
            x0, y0 = self._edit_last_pos
            dx = x_px - x0
            dy = y_px - y0
            steps = max(1, int(max(abs(dx), abs(dy))))
            for step in range(steps + 1):
                tx = int(round(x0 + dx * step / float(steps)))
                ty = int(round(y0 + dy * step / float(steps)))
                self._apply_brush(self._edit_active_mask, self._edit_active_entry, tx, ty, add)
        self._edit_last_pos = (x_px, y_px)
        if self._edit_active_frame is not None:
            self.component_cache._cache.pop(self._edit_active_frame, None)
        self._edit_dirty = True
        self._refresh_current_view()

    def _commit_edit_mask(self) -> None:
        if not self._edit_dirty and not self._edit_index_dirty:
            return
        if self._edit_active_entry is None or self._edit_active_mask is None:
            return
        try:
            np.savez_compressed(self._edit_active_entry.path, mask=self._edit_active_mask.astype(np.uint8))
        except Exception as exc:
            logger.warning("Failed to save edited mask: %s (%s)", self._edit_active_entry.path, exc)
        self._edit_dirty = False
        if self._edit_index_dirty:
            self._write_mask_index_entry(self._edit_active_entry)
            self._edit_index_dirty = False

    def _fill_holes_current_mask(self) -> None:
        if not self.frame_order:
            return
        frame = self.frame_order[self.current_frame_index]
        if not self._ensure_edit_mask_for_frame(frame):
            return
        if self._edit_active_mask is None or self._edit_active_entry is None:
            return
        filled = binary_fill_holes(self._edit_active_mask > 0)
        self._edit_active_mask[:, :] = filled.astype(np.uint8)
        self.component_cache._cache.pop(frame, None)
        self._edit_dirty = True
        self._commit_edit_mask()
        self._refresh_current_view()

    def _remove_dust_current_mask(self) -> None:
        if not self.frame_order:
            return
        frame = self.frame_order[self.current_frame_index]
        if not self._ensure_edit_mask_for_frame(frame):
            return
        if self._edit_active_mask is None:
            return
        min_area = int(self.edit_dust_size_spin.value()) if hasattr(self, "edit_dust_size_spin") else 1
        min_area = max(1, min_area)
        labeled, num = nd_label(self._edit_active_mask > 0)
        if num <= 0:
            return
        counts = np.bincount(labeled.ravel())
        keep = counts >= min_area
        if keep.size:
            keep[0] = False
        cleaned = keep[labeled]
        self._edit_active_mask[:, :] = cleaned.astype(np.uint8)
        self.component_cache._cache.pop(frame, None)
        self._edit_dirty = True
        self._commit_edit_mask()
        self._refresh_current_view()

    def _smooth_current_mask(self) -> None:
        if not self.frame_order:
            return
        frame = self.frame_order[self.current_frame_index]
        if not self._ensure_edit_mask_for_frame(frame):
            return
        if self._edit_active_mask is None:
            return
        kernel = int(self.smooth_kernel_spin.value()) if hasattr(self, "smooth_kernel_spin") else 1
        kernel = max(1, kernel)
        if kernel % 2 == 0:
            kernel += 1
        structure = np.ones((kernel, kernel), dtype=bool)
        mask = self._edit_active_mask > 0
        smoothed = binary_opening(mask, structure=structure)
        smoothed = binary_closing(smoothed, structure=structure)
        self._edit_active_mask[:, :] = smoothed.astype(np.uint8)
        self.component_cache._cache.pop(frame, None)
        self._edit_dirty = True
        self._commit_edit_mask()
        self._refresh_current_view()

    def _confirm_batch_edit(self, title: str) -> bool:
        reply = QtWidgets.QMessageBox.question(
            self,
            "Batch Edit",
            f"{title} across all masks in this neuron?\nThis overwrites mask files.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return reply == QtWidgets.QMessageBox.Yes

    def _apply_mask_op_all(self, label: str, op: Callable[[np.ndarray], np.ndarray]) -> None:
        if not self.entries:
            return
        if self._edit_dirty:
            self._commit_edit_mask()
        if not self._confirm_batch_edit(label):
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        start_t = time.perf_counter()
        total = 0
        changed = 0
        for entry in self.entries:
            mask = load_mask_array(entry)
            if mask is None:
                continue
            total += 1
            new_mask = op(mask)
            if new_mask is None:
                continue
            if np.array_equal(new_mask, mask):
                continue
            try:
                np.savez_compressed(entry.path, mask=new_mask.astype(np.uint8))
                changed += 1
            except Exception as exc:
                logger.warning("Batch edit save failed: %s (%s)", entry.path, exc)
        self.mask_cache.clear()
        self.component_cache.clear()
        self._refresh_current_view()
        QtWidgets.QApplication.restoreOverrideCursor()
        elapsed = time.perf_counter() - start_t
        logger.info("Batch edit %s: total=%d changed=%d time=%.2fs", label, total, changed, elapsed)
        QtWidgets.QMessageBox.information(
            self,
            "Batch Edit",
            f"{label} complete.\nChanged {changed} of {total} masks.",
        )

    def _fill_holes_all_masks(self) -> None:
        def _op(mask: np.ndarray) -> np.ndarray:
            return binary_fill_holes(mask > 0).astype(np.uint8)

        self._apply_mask_op_all("Fill holes", _op)

    def _remove_dust_all_masks(self) -> None:
        min_area = int(self.edit_dust_size_spin.value()) if hasattr(self, "edit_dust_size_spin") else 1
        min_area = max(1, min_area)

        def _op(mask: np.ndarray) -> np.ndarray:
            labeled, num = nd_label(mask > 0)
            if num <= 0:
                return mask
            counts = np.bincount(labeled.ravel())
            keep = counts >= min_area
            if keep.size:
                keep[0] = False
            return keep[labeled].astype(np.uint8)

        self._apply_mask_op_all(f"Remove dust (<{min_area}px)", _op)

    def _smooth_all_masks(self) -> None:
        kernel = int(self.smooth_kernel_spin.value()) if hasattr(self, "smooth_kernel_spin") else 1
        kernel = max(1, kernel)
        if kernel % 2 == 0:
            kernel += 1
        structure = np.ones((kernel, kernel), dtype=bool)

        def _op(mask: np.ndarray) -> np.ndarray:
            smoothed = binary_opening(mask > 0, structure=structure)
            smoothed = binary_closing(smoothed, structure=structure)
            return smoothed.astype(np.uint8)

        self._apply_mask_op_all(f"Smooth (k={kernel})", _op)

    def _write_mask_index_entry(self, entry: MaskEntry) -> None:
        if self.data_root is None or not self._current_neuron_name:
            return
        from pathlib import Path
        index_path = self.data_root / self._current_neuron_name / "masks" / "index.json"
        if not index_path.exists():
            return
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read mask index for update: %s", index_path)
            return
        if not isinstance(data, list):
            return
        mask_dir = index_path.parent
        rel_path = None
        try:
            rel_path = str(entry.path.relative_to(mask_dir))
        except Exception:
            rel_path = entry.path.name
        updated = False
        for item in data:
            if not isinstance(item, dict):
                continue
            path_val = str(item.get("path", "") or "")
            id_val = str(item.get("id", "") or "")
            if path_val == rel_path or Path(path_val).name == entry.path.name or id_val == entry.mask_id:
                item["x"] = int(entry.x)
                item["y"] = int(entry.y)
                item["width"] = int(entry.width)
                item["height"] = int(entry.height)
                updated = True
                break
        if not updated:
            return
        try:
            index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("Failed to write mask index update: %s", index_path)

    def _ensure_edit_mask_for_frame(
        self, frame: int, *, x_px: Optional[float] = None, y_px: Optional[float] = None
    ) -> bool:
        if self._edit_active_entry is not None and self._edit_active_frame == frame and self._edit_active_mask is not None:
            return True
        if x_px is None or y_px is None:
            center = self._current_focus_center_px
            if center is None:
                mesh_point = self._current_mesh_point()
                if mesh_point is not None:
                    center = (mesh_point[0], mesh_point[1])
            if center is None:
                return False
            x_px, y_px = center
        entry = self._pick_edit_entry(frame, float(x_px), float(y_px))
        if entry is None:
            return False
        self._edit_active_entry = entry
        self._edit_active_frame = frame
        self._edit_active_mask = self._get_mask_for_entry(frame, entry)
        self._edit_dirty = False
        return self._edit_active_mask is not None
