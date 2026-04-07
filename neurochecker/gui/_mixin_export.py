import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from PyQt5 import QtWidgets

from neurochecker.gui.constants import SAMPLE_EXPORT_MIN_LONG_SIDE_PX, logger
from neurochecker.gui.data import ComponentInfo
from neurochecker.mask_io import MaskEntry


class ExportMixin:
    def _sample_export_crop_size(self, crop_w: int, crop_h: int) -> Tuple[int, int]:
        crop_w = max(1, int(crop_w))
        crop_h = max(1, int(crop_h))
        long_side = max(crop_w, crop_h)
        if long_side <= 0 or long_side >= SAMPLE_EXPORT_MIN_LONG_SIDE_PX:
            return crop_w, crop_h
        scale = SAMPLE_EXPORT_MIN_LONG_SIDE_PX / float(long_side)
        return max(1, int(round(crop_w * scale))), max(1, int(round(crop_h * scale)))

    def _prompt_export_options(self, *, title: str, action_label: str) -> Optional[bool]:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.resize(520, 180)

        layout = QtWidgets.QVBoxLayout(dialog)
        summary = QtWidgets.QLabel(f"Choose how NeuroChecker should load image crops for this {action_label}.")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        detail = QtWidgets.QLabel(
            "Checked: bypass the NeuroTracer pyramid and crop directly from the original TIFF/image stack files at full resolution."
        )
        detail.setWordWrap(True)
        layout.addWidget(detail)

        force_source_box = QtWidgets.QCheckBox(
            "Always export this run at full resolution from source TIFF/image files"
        )
        force_source_box.setChecked(False)
        layout.addWidget(force_source_box)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None
        return force_source_box.isChecked()

    def _export_viewport_image(
        self,
        frame: int,
        viewport_rect: Tuple[int, int, int, int],
        target_size: Tuple[int, int],
        *,
        force_source_full_res: bool,
    ) -> Tuple[np.ndarray, float, str]:
        if self.image_sampler is None:
            raise RuntimeError("Image sampler not initialized.")
        if force_source_full_res:
            image, scale_factor = self.image_sampler.get_source_viewport_image(
                frame, viewport_rect, target_size
            )
            return image, scale_factor, "source_stack"
        image, scale_factor = self.image_sampler.get_viewport_image(
            frame, viewport_rect, target_size
        )
        image_source = "pyramid" if getattr(self.image_sampler, "has_pyramid", False) else "source_stack"
        return image, scale_factor, image_source

    def _toggle_flag_frame(self) -> None:
        from neurochecker.gui.helpers import _flag_point_record
        if not self.frame_order:
            return
        frame = self.frame_order[self.current_frame_index]
        if self.active_segment_id is not None:
            seg_id = int(self.active_segment_id)
            key = (seg_id, int(frame))
            if key in self.flagged_points:
                self.flagged_points.pop(key, None)
            else:
                step = int(self.current_frame_index)
                mesh_point = self._current_mesh_point()
                if mesh_point is not None:
                    _x_px, _y_px, x, y, z = mesh_point
                else:
                    x = y = z = None
                self.flagged_points[key] = _flag_point_record(seg_id, int(frame), step=step, x=x, y=y, z=z)
            self._save_flagged_points()
            self._refresh_current_view()
            self._refresh_minimap()
            return
        entries = self._selected_entries_for_frame(frame)
        if not entries:
            return
        ids = [entry.mask_id for entry in entries]
        if all(mask_id in self.flagged_masks for mask_id in ids):
            for mask_id in ids:
                self.flagged_masks.discard(mask_id)
        else:
            for mask_id in ids:
                self.flagged_masks.add(mask_id)
        self._refresh_current_view()

    def _export_flags_csv(self) -> None:
        if not self.entries:
            return
        neuron_name = self._selected_neuron_name() or "neuron"
        default_name = f"{neuron_name}_flags.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save flags CSV",
            default_name,
            "CSV files (*.csv)",
        )
        if not path:
            return
        self._write_flags_csv(Path(path))

    def _export_crops_and_masks(self) -> None:
        if not self.entries:
            return
        if self.image_sampler is None:
            QtWidgets.QMessageBox.warning(self, "Export", "Select an image stack directory first.")
            return
        force_source_full_res = self._prompt_export_options(
            title="Export Options",
            action_label="export",
        )
        if force_source_full_res is None:
            return
        base_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export directory")
        if not base_dir:
            return
        neuron_name = self._selected_neuron_name() or "neuron"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_root = Path(base_dir) / f"neurochecker_export_{neuron_name}_{timestamp}"
        segments_root = export_root / "segments"
        segments_root.mkdir(parents=True, exist_ok=True)

        if not self.mesh_segments:
            QtWidgets.QMessageBox.warning(self, "Export", "Mesh skeleton not available for export.")
            return

        segment_paths = self._build_segment_tree_paths()

        segment_node_data: List[Tuple[int, Path, List[int]]] = []
        total_nodes = 0
        for seg_id, rel_path in sorted(segment_paths.items(), key=lambda item: str(item[1])):
            node_ids = list(self.mesh_segments[seg_id])
            total_nodes += len(node_ids)
            segment_node_data.append((seg_id, rel_path, node_ids))

        progress = QtWidgets.QProgressDialog("Exporting...", "Cancel", 0, total_nodes, self)
        progress.setWindowTitle("Export")
        progress.setMinimumDuration(0)
        progress.setAutoReset(False)
        progress.setAutoClose(False)
        progress.setValue(0)
        exported = 0
        logger.info(
            "Export started: total_nodes=%d segments=%d force_source_full_res=%s",
            total_nodes,
            len(segment_node_data),
            force_source_full_res,
        )

        metadata_rows: List[Dict[str, Any]] = []
        sample_rows: List[Dict[str, Any]] = []
        for seg_id, rel_path, node_ids in segment_node_data:
            seg_root = segments_root / rel_path
            images_dir = seg_root / "images"
            masks_dir = seg_root / "masks"
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)

            frame_points: Dict[int, Tuple[float, float]] = {}
            for nid in node_ids:
                node = self.nodes[nid]
                frame_points[node.frame] = (node.x_px, node.y_px)

            for step_idx, node_id in enumerate(node_ids):
                if progress.wasCanceled():
                    break
                node = self.nodes[node_id]
                frame = node.frame
                centerline_x_px = float(node.x_px)
                centerline_y_px = float(node.y_px)
                centerline_x = float(node.x)
                centerline_y = float(node.y)
                centerline_z = float(node.z)

                comp = self._select_component_for_segment_frame(
                    frame,
                    seg_id,
                    center_hint_px=(centerline_x_px, centerline_y_px),
                )
                centroid_x_px = None
                centroid_y_px = None
                center_x_px = centerline_x_px
                center_y_px = centerline_y_px
                if comp is not None:
                    centroid_x_px, centroid_y_px = comp.centroid
                    center_x_px = (center_x_px + float(centroid_x_px)) * 0.5
                    center_y_px = (center_y_px + float(centroid_y_px)) * 0.5

                base_w, base_h = self._segment_focus_size(
                    frame,
                    seg_id,
                    frame_points=frame_points,
                )
                crop_w, crop_h = self._crop_size_for_frame(base_w, base_h)
                crop_w, crop_h = self._sample_export_crop_size(crop_w, crop_h)
                x0, y0, crop_w, crop_h = self._clamp_viewport_center(center_x_px, center_y_px, crop_w, crop_h)
                target_size = (crop_w, crop_h)
                image, scale_factor, image_source = self._export_viewport_image(
                    frame,
                    (x0, y0, crop_w, crop_h),
                    target_size,
                    force_source_full_res=force_source_full_res,
                )
                image_path = images_dir / f"frame_{frame:05d}_step_{step_idx:05d}.png"
                Image.fromarray(image).save(image_path)

                mask_path_str = ""
                mask_id = ""
                run_id = ""
                flagged = 0
                if comp is not None:
                    mask_crop = self._component_crop_for_export(
                        comp,
                        (x0, y0, crop_w, crop_h),
                        scale_factor,
                        target_size,
                    )
                    if mask_crop is not None:
                        mask_path = masks_dir / f"frame_{frame:05d}_step_{step_idx:05d}_mask.png"
                        Image.fromarray(mask_crop).save(mask_path)
                        mask_path_str = str(mask_path)
                        mask_id = comp.entry.mask_id
                        run_id = comp.entry.run_id
                        flagged = 1 if comp.entry.mask_id in self.flagged_masks else 0
                        metadata_rows.append(
                            dict(
                                neuron=neuron_name,
                                segment=rel_path.as_posix(),
                                segment_index=seg_id,
                                node_id=node_id,
                                frame=frame,
                                step=step_idx,
                                run_id=comp.entry.run_id,
                                mask_id=comp.entry.mask_id,
                                mask_path=str(mask_path),
                                image_path=str(image_path),
                                image_source=image_source,
                                force_source_full_res=int(force_source_full_res),
                                crop_x=x0,
                                crop_y=y0,
                                crop_w=crop_w,
                                crop_h=crop_h,
                                scale_factor=scale_factor,
                                centerline_x_px=centerline_x_px,
                                centerline_y_px=centerline_y_px,
                                centroid_x_px=float(centroid_x_px) if centroid_x_px is not None else None,
                                centroid_y_px=float(centroid_y_px) if centroid_y_px is not None else None,
                                sample_center_x_px=float(center_x_px),
                                sample_center_y_px=float(center_y_px),
                                mask_x=comp.entry.x,
                                mask_y=comp.entry.y,
                                mask_w=comp.entry.width,
                                mask_h=comp.entry.height,
                                full_width=comp.entry.full_width,
                                full_height=comp.entry.full_height,
                                flagged=flagged,
                            )
                        )
                sample_rows.append(
                    dict(
                        neuron=neuron_name,
                        segment=rel_path.as_posix(),
                        segment_index=seg_id,
                        node_id=node_id,
                        frame=frame,
                        step=step_idx,
                        image_path=str(image_path),
                        mask_path=mask_path_str,
                        image_source=image_source,
                        force_source_full_res=int(force_source_full_res),
                        run_id=run_id,
                        mask_id=mask_id,
                        centerline_x_px=centerline_x_px,
                        centerline_y_px=centerline_y_px,
                        centerline_x=centerline_x,
                        centerline_y=centerline_y,
                        centerline_z=centerline_z,
                        centroid_x_px=float(centroid_x_px) if centroid_x_px is not None else None,
                        centroid_y_px=float(centroid_y_px) if centroid_y_px is not None else None,
                        sample_center_x_px=float(center_x_px),
                        sample_center_y_px=float(center_y_px),
                        crop_x=x0,
                        crop_y=y0,
                        crop_w=crop_w,
                        crop_h=crop_h,
                        scale_factor=scale_factor,
                        has_mask=1 if mask_path_str else 0,
                        flagged=flagged,
                    )
                )
                exported += 1
                progress.setValue(exported)
                progress.setLabelText(f"Exporting segment {seg_id} node {node_id} — {exported}/{total_nodes}")
                QtWidgets.QApplication.processEvents()
            if progress.wasCanceled():
                break

        was_canceled = progress.wasCanceled()
        progress.setValue(total_nodes)
        progress.close()

        if was_canceled:
            logger.info("Export cancelled by user at %d/%d nodes", exported, total_nodes)
            QtWidgets.QMessageBox.information(self, "Export", f"Export cancelled. Partial output in {export_root}")
            return

        if metadata_rows:
            metadata_path = export_root / "metadata.csv"
            with metadata_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
                writer.writeheader()
                writer.writerows(metadata_rows)
        if sample_rows:
            sample_json_path = export_root / "sample_points.json"
            with sample_json_path.open("w", encoding="utf-8") as handle:
                json.dump(sample_rows, handle, indent=2)
            sample_csv_path = export_root / "sample_points.csv"
            with sample_csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()))
                writer.writeheader()
                writer.writerows(sample_rows)
        flags_path = export_root / "flags.csv"
        self._write_flags_csv(flags_path)
        self._write_skeleton_json(export_root)
        self._write_segment_tree_json(export_root, segment_paths)
        self._copy_mesh_to_export(export_root)
        logger.info("Export complete: %s", export_root)
        QtWidgets.QMessageBox.information(self, "Export", f"Exported to {export_root}")

    def _mass_export(self) -> None:
        if self.data_root is None:
            QtWidgets.QMessageBox.warning(self, "Mass Export", "Set a data root first.")
            return
        if self.image_sampler is None:
            QtWidgets.QMessageBox.warning(self, "Mass Export", "Set an image stack directory first.")
            return

        neuron_dirs = sorted(
            p for p in self.data_root.iterdir()
            if p.is_dir() and p.name.startswith("neuron_")
        )
        if not neuron_dirs:
            QtWidgets.QMessageBox.information(self, "Mass Export", "No neurons found in data root.")
            return

        from neurochecker.gui._mass_export_dialog import MassExportDialog

        dlg = MassExportDialog(neuron_dirs, parent=self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        selected_neurons = dlg.selected_neurons()
        if not selected_neurons:
            return

        force_source_full_res = self._prompt_export_options(
            title="Mass Export Options",
            action_label="mass export",
        )
        if force_source_full_res is None:
            return

        base_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory for mass export")
        if not base_dir:
            return

        from neurochecker.pipeline import normalize_neuron_id

        progress = QtWidgets.QProgressDialog(
            "Mass exporting...", "Cancel", 0, len(selected_neurons), self
        )
        progress.setWindowTitle("Mass Export")
        progress.setMinimumDuration(0)
        progress.setValue(0)

        original_neuron = self._current_neuron_name
        succeeded = 0
        failed = []

        for idx, neuron_name in enumerate(selected_neurons):
            if progress.wasCanceled():
                break
            progress.setLabelText(f"Exporting {neuron_name} ({idx + 1}/{len(selected_neurons)})")
            progress.setValue(idx)
            QtWidgets.QApplication.processEvents()

            neuron_id = normalize_neuron_id(neuron_name)
            try:
                self._load_neuron(neuron_id, raise_on_error=True)
            except Exception as exc:
                logger.warning("Mass export: failed to load %s: %s", neuron_name, exc)
                failed.append(f"{neuron_name}: load failed")
                continue

            if not self.mesh_segments:
                failed.append(f"{neuron_name}: no segments")
                continue

            segment_paths = self._build_segment_tree_paths()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_root = Path(base_dir) / f"neurochecker_export_{neuron_name}_{timestamp}"
            segments_root = export_root / "segments"
            segments_root.mkdir(parents=True, exist_ok=True)

            try:
                self._mass_export_single(
                    neuron_name,
                    segment_paths,
                    segments_root,
                    export_root,
                    force_source_full_res=force_source_full_res,
                )
                succeeded += 1
            except Exception as exc:
                logger.warning("Mass export: failed to export %s: %s", neuron_name, exc)
                failed.append(f"{neuron_name}: {exc}")

        progress.setValue(len(selected_neurons))
        progress.close()

        if original_neuron:
            try:
                self._load_neuron(normalize_neuron_id(original_neuron))
            except Exception:
                pass

        summary = f"Exported {succeeded}/{len(selected_neurons)} neurons."
        if failed:
            summary += "\n\nFailed:\n" + "\n".join(failed)
        QtWidgets.QMessageBox.information(self, "Mass Export", summary)

    def _mass_export_single(
        self,
        neuron_name: str,
        segment_paths: Dict,
        segments_root: Path,
        export_root: Path,
        *,
        force_source_full_res: bool,
    ) -> None:
        metadata_rows: List[Dict[str, Any]] = []
        sample_rows: List[Dict[str, Any]] = []

        for seg_id, rel_path in sorted(segment_paths.items(), key=lambda item: str(item[1])):
            node_ids = list(self.mesh_segments[seg_id])
            seg_root = segments_root / rel_path
            images_dir = seg_root / "images"
            masks_dir = seg_root / "masks"
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)

            frame_points: Dict[int, Tuple[float, float]] = {}
            for nid in node_ids:
                node = self.nodes[nid]
                frame_points[node.frame] = (node.x_px, node.y_px)

            for step_idx, node_id in enumerate(node_ids):
                node = self.nodes[node_id]
                frame = node.frame
                cx_px = float(node.x_px)
                cy_px = float(node.y_px)

                comp = self._select_component_for_segment_frame(
                    frame, seg_id, center_hint_px=(cx_px, cy_px)
                )
                centroid_x_px = None
                centroid_y_px = None
                center_x_px = cx_px
                center_y_px = cy_px
                if comp is not None:
                    centroid_x_px, centroid_y_px = comp.centroid
                    center_x_px = (center_x_px + float(centroid_x_px)) * 0.5
                    center_y_px = (center_y_px + float(centroid_y_px)) * 0.5

                base_w, base_h = self._segment_focus_size(
                    frame, seg_id, frame_points=frame_points
                )
                crop_w, crop_h = self._crop_size_for_frame(base_w, base_h)
                crop_w, crop_h = self._sample_export_crop_size(crop_w, crop_h)
                x0, y0, crop_w, crop_h = self._clamp_viewport_center(
                    center_x_px, center_y_px, crop_w, crop_h
                )
                target_size = (crop_w, crop_h)
                image, scale_factor, image_source = self._export_viewport_image(
                    frame,
                    (x0, y0, crop_w, crop_h),
                    target_size,
                    force_source_full_res=force_source_full_res,
                )
                image_path = images_dir / f"frame_{frame:05d}_step_{step_idx:05d}.png"
                Image.fromarray(image).save(image_path)

                mask_path_str = ""
                mask_id = ""
                run_id = ""
                flagged = 0
                if comp is not None:
                    mask_crop = self._component_crop_for_export(
                        comp, (x0, y0, crop_w, crop_h), scale_factor, target_size
                    )
                    if mask_crop is not None:
                        mask_path = masks_dir / f"frame_{frame:05d}_step_{step_idx:05d}_mask.png"
                        Image.fromarray(mask_crop).save(mask_path)
                        mask_path_str = str(mask_path)
                        mask_id = comp.entry.mask_id
                        run_id = comp.entry.run_id
                        flagged = 1 if comp.entry.mask_id in self.flagged_masks else 0
                        metadata_rows.append(
                            dict(
                                neuron=neuron_name,
                                segment=rel_path.as_posix(),
                                segment_index=seg_id,
                                node_id=node_id,
                                frame=frame,
                                step=step_idx,
                                run_id=comp.entry.run_id,
                                mask_id=comp.entry.mask_id,
                                mask_path=str(mask_path),
                                image_path=str(image_path),
                                image_source=image_source,
                                force_source_full_res=int(force_source_full_res),
                                crop_x=x0,
                                crop_y=y0,
                                crop_w=crop_w,
                                crop_h=crop_h,
                                scale_factor=scale_factor,
                                centerline_x_px=cx_px,
                                centerline_y_px=cy_px,
                                centroid_x_px=float(centroid_x_px) if centroid_x_px is not None else None,
                                centroid_y_px=float(centroid_y_px) if centroid_y_px is not None else None,
                                sample_center_x_px=float(center_x_px),
                                sample_center_y_px=float(center_y_px),
                                mask_x=comp.entry.x,
                                mask_y=comp.entry.y,
                                mask_w=comp.entry.width,
                                mask_h=comp.entry.height,
                                full_width=comp.entry.full_width,
                                full_height=comp.entry.full_height,
                                flagged=flagged,
                            )
                        )
                sample_rows.append(
                    dict(
                        neuron=neuron_name,
                        segment=rel_path.as_posix(),
                        segment_index=seg_id,
                        node_id=node_id,
                        frame=frame,
                        step=step_idx,
                        image_path=str(image_path),
                        mask_path=mask_path_str,
                        image_source=image_source,
                        force_source_full_res=int(force_source_full_res),
                        run_id=run_id,
                        mask_id=mask_id,
                        centerline_x_px=cx_px,
                        centerline_y_px=cy_px,
                        centerline_x=float(node.x),
                        centerline_y=float(node.y),
                        centerline_z=float(node.z),
                        centroid_x_px=float(centroid_x_px) if centroid_x_px is not None else None,
                        centroid_y_px=float(centroid_y_px) if centroid_y_px is not None else None,
                        sample_center_x_px=float(center_x_px),
                        sample_center_y_px=float(center_y_px),
                        crop_x=x0,
                        crop_y=y0,
                        crop_w=crop_w,
                        crop_h=crop_h,
                        scale_factor=scale_factor,
                        has_mask=1 if mask_path_str else 0,
                        flagged=flagged,
                    )
                )

        if metadata_rows:
            with (export_root / "metadata.csv").open("w", newline="", encoding="utf-8") as h:
                writer = csv.DictWriter(h, fieldnames=list(metadata_rows[0].keys()))
                writer.writeheader()
                writer.writerows(metadata_rows)
        if sample_rows:
            with (export_root / "sample_points.json").open("w", encoding="utf-8") as h:
                json.dump(sample_rows, h, indent=2)
            with (export_root / "sample_points.csv").open("w", newline="", encoding="utf-8") as h:
                writer = csv.DictWriter(h, fieldnames=list(sample_rows[0].keys()))
                writer.writeheader()
                writer.writerows(sample_rows)
        self._write_flags_csv(export_root / "flags.csv")
        self._write_skeleton_json(export_root)
        self._write_segment_tree_json(export_root, segment_paths)
        self._copy_mesh_to_export(export_root)
        logger.info("Mass export single complete: %s -> %s", neuron_name, export_root)

    def _frame_focus_for_export(self, frame: int) -> Optional[Tuple[float, float, int, int]]:
        entries = self._selected_entries_for_frame(frame)
        if not entries:
            return None
        focus = max(entries, key=lambda e: e.width * e.height)
        base_w = int(focus.width)
        base_h = int(focus.height)
        stats = self.run_stats.get(focus.run_id)
        if stats:
            base_w = max(base_w, int(stats.max_width))
            base_h = max(base_h, int(stats.max_height))
        if base_w <= 0 or base_h <= 0:
            base_w = 256
            base_h = 256
        center_x = float(focus.x + base_w / 2.0)
        center_y = float(focus.y + base_h / 2.0)
        return center_x, center_y, base_w, base_h

    def _write_skeleton_json(self, export_root: Path) -> None:
        if not self.nodes or self.mesh_graph is None:
            return
        px = float(self.pixel_xy_spin.value()) if float(self.pixel_xy_spin.value()) > 0 else 1.0
        sz = float(self.slice_z_spin.value()) if float(self.slice_z_spin.value()) > 0 else 1.0
        hillock_cutoff = None
        if self._hillock_original_node_id is not None and self._distal_original_node_id is not None:
            hillock_cutoff = {
                "hillock_original_node_id": int(self._hillock_original_node_id),
                "distal_original_node_id": int(self._distal_original_node_id),
                "forward_neighbor_original_node_id": (
                    int(self._hillock_forward_original_node_id)
                    if self._hillock_forward_original_node_id is not None
                    else None
                ),
                "soma_original_node_ids": list(self._soma_original_node_ids),
                "soma_segment_index": self._soma_segment_id,
                "primary_neurite_segment_index": self._primary_neurite_segment_id,
            }
        data: Dict[str, Any] = {
            "nodes": [[float(n.x), float(n.y), float(n.z)] for n in self.nodes],
            "node_pixel_coords": [[float(n.x_px), float(n.y_px)] for n in self.nodes],
            "node_frames": [int(n.frame) for n in self.nodes],
            "edges": [[int(i), int(j)] for i, j in self.mesh_graph.edges],
            "segments": [list(seg) for seg in self.mesh_segments],
            "segment_special_paths": {
                str(int(seg_id)): (
                    raw_path.as_posix() if isinstance(raw_path, Path) else Path(str(raw_path)).as_posix()
                )
                for seg_id, raw_path in (self._segment_special_paths or {}).items()
            },
            "node_to_segments": {
                str(nid): segs
                for nid, segs in self.mesh_node_to_segments.items()
            },
            "pixel_size_xy": px,
            "slice_thickness_z": sz,
            "current_to_original_node_ids": list(self._current_original_node_ids),
            "hillock_cutoff": hillock_cutoff,
        }
        path = export_root / "skeleton.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        logger.info("Wrote skeleton.json: %d nodes, %d edges", len(self.nodes), len(self.mesh_graph.edges))

    def _write_segment_tree_json(
        self, export_root: Path, segment_paths: Dict[int, Path]
    ) -> None:
        if not self.mesh_segments:
            return
        normalized_paths: Dict[int, Path] = {}
        for seg_id, raw_path in segment_paths.items():
            normalized_paths[int(seg_id)] = raw_path if isinstance(raw_path, Path) else Path(str(raw_path))

        root = None
        root_builder = getattr(self, "_segment_tree_root_id", None)
        if callable(root_builder):
            root = root_builder()
        if root is None or root < 0 or root >= len(self.mesh_segments):
            if not normalized_paths:
                return
            root = max(normalized_paths, key=lambda sid: len(self.mesh_segments[sid]))

        parent_map: Dict[int, Optional[int]] = {}
        children_map: Dict[int, List[int]] = {i: [] for i in range(len(self.mesh_segments))}
        path_to_segment = {
            path.as_posix(): seg_id for seg_id, path in normalized_paths.items()
        }

        for seg_id in range(len(self.mesh_segments)):
            raw_path = normalized_paths.get(seg_id, Path(f"segment_{seg_id}"))
            parent: Optional[int] = None
            for ancestor in raw_path.parents:
                if not ancestor.parts:
                    continue
                parent = path_to_segment.get(ancestor.as_posix())
                if parent is not None:
                    break
            parent_map[seg_id] = parent
            if parent is not None:
                children_map[parent].append(seg_id)

        for seg_id, children in children_map.items():
            children.sort(
                key=lambda child_id: (
                    normalized_paths.get(child_id, Path(f"segment_{child_id}")).as_posix(),
                    child_id,
                )
            )

        segments_dict: Dict[str, Any] = {}
        for seg_id in range(len(self.mesh_segments)):
            node_ids = self.mesh_segments[seg_id]
            unique_frames = set()
            for nid in node_ids:
                unique_frames.add(int(self.nodes[nid].frame))
            raw_path = normalized_paths.get(seg_id, Path(f"segment_{seg_id}"))
            segments_dict[str(seg_id)] = {
                "path": raw_path.as_posix() if hasattr(raw_path, "as_posix") else str(raw_path),
                "parent": parent_map.get(seg_id),
                "children": children_map.get(seg_id, []),
                "node_count": len(node_ids),
                "frame_count": len(unique_frames),
            }

        data = {
            "root_segment_index": root,
            "total_segments": len(self.mesh_segments),
            "segments": segments_dict,
        }
        path = export_root / "segment_tree.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        logger.info("Wrote segment_tree.json: %d segments", len(self.mesh_segments))

    def _copy_mesh_to_export(self, export_root: Path) -> None:
        mesh_path = getattr(self, "_mesh_path", None)
        if mesh_path is None or not Path(mesh_path).exists():
            logger.warning("No mesh file to copy into export")
            return
        dest = export_root / "mesh.ply"
        shutil.copy2(str(mesh_path), str(dest))
        logger.info("Copied mesh to export: %s -> %s", mesh_path, dest)

    def _write_flags_csv(self, path: Path) -> None:
        rows = []
        for entry in self.entries:
            rows.append(
                dict(
                    mask_id=entry.mask_id,
                    run_id=entry.run_id,
                    frame=entry.frame,
                    x=entry.x,
                    y=entry.y,
                    width=entry.width,
                    height=entry.height,
                    full_width=entry.full_width,
                    full_height=entry.full_height,
                    path=str(entry.path),
                    flagged=1 if entry.mask_id in self.flagged_masks else 0,
                )
            )
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _mask_crop_for_export(
        self,
        entry: MaskEntry,
        mask: np.ndarray,
        viewport_rect: Tuple[int, int, int, int],
        scale_factor: float,
        target_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        vx, vy, vw, vh = viewport_rect
        tw, th = target_size
        px0 = entry.x
        py0 = entry.y
        ph, pw = mask.shape[:2]
        px1 = px0 + pw
        py1 = py0 + ph

        ix0 = max(px0, vx)
        iy0 = max(py0, vy)
        ix1 = min(px1, vx + vw)
        iy1 = min(py1, vy + vh)
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        crop_x0 = ix0 - px0
        crop_y0 = iy0 - py0
        crop_x1 = crop_x0 + (ix1 - ix0)
        crop_y1 = crop_y0 + (iy1 - iy0)
        mask_crop = mask[crop_y0:crop_y1, crop_x0:crop_x1]
        if mask_crop.size == 0:
            return None
        target_w = max(1, int(round(mask_crop.shape[1] / scale_factor)))
        target_h = max(1, int(round(mask_crop.shape[0] / scale_factor)))
        mask_img = Image.fromarray((mask_crop > 0).astype(np.uint8) * 255)
        mask_resized = np.asarray(mask_img.resize((target_w, target_h), Image.NEAREST))
        ox0 = int(round((ix0 - vx) / scale_factor))
        oy0 = int(round((iy0 - vy) / scale_factor))
        ox1 = min(tw, ox0 + target_w)
        oy1 = min(th, oy0 + target_h)
        if ox1 <= ox0 or oy1 <= oy0:
            return None
        out = np.zeros((th, tw), dtype=np.uint8)
        sub = out[oy0:oy1, ox0:ox1]
        sub_mask = mask_resized[: oy1 - oy0, : ox1 - ox0]
        sub[sub_mask > 0] = 255
        return out

    def _component_crop_for_export(
        self,
        comp: ComponentInfo,
        viewport_rect: Tuple[int, int, int, int],
        scale_factor: float,
        target_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        entry = comp.entry
        slc_y, slc_x = comp.slices
        vx, vy, vw, vh = viewport_rect
        tw, th = target_size
        comp_x0 = entry.x + slc_x.start
        comp_y0 = entry.y + slc_y.start
        comp_x1 = entry.x + slc_x.stop
        comp_y1 = entry.y + slc_y.stop

        ix0 = max(comp_x0, vx)
        iy0 = max(comp_y0, vy)
        ix1 = min(comp_x1, vx + vw)
        iy1 = min(comp_y1, vy + vh)
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        crop_x0 = ix0 - entry.x
        crop_y0 = iy0 - entry.y
        crop_x1 = crop_x0 + (ix1 - ix0)
        crop_y1 = crop_y0 + (iy1 - iy0)
        mask_crop = comp.labeled[crop_y0:crop_y1, crop_x0:crop_x1] == comp.label_id
        if mask_crop.size == 0 or not mask_crop.any():
            return None
        target_w = max(1, int(round(mask_crop.shape[1] / scale_factor)))
        target_h = max(1, int(round(mask_crop.shape[0] / scale_factor)))
        mask_img = Image.fromarray(mask_crop.astype(np.uint8) * 255)
        mask_resized = np.asarray(mask_img.resize((target_w, target_h), Image.NEAREST)) > 0
        ox0 = int(round((ix0 - vx) / scale_factor))
        oy0 = int(round((iy0 - vy) / scale_factor))
        ox1 = min(tw, ox0 + target_w)
        oy1 = min(th, oy0 + target_h)
        if ox1 <= ox0 or oy1 <= oy0:
            return None
        out = np.zeros((th, tw), dtype=np.uint8)
        sub = out[oy0:oy1, ox0:ox1]
        sub_mask = mask_resized[: oy1 - oy0, : ox1 - ox0]
        sub[sub_mask > 0] = 255
        return out
