import tempfile
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.ndimage import (
    label as nd_label,
    find_objects,
)

from neurochecker.gui.constants import FLAG_FRAME_BORDER_PX, logger
from neurochecker.gui.data import ComponentInfo
from neurochecker.gui.helpers import _mask_outline, numpy_to_qimage
from neurochecker.gui.mesh import _mesh_skeleton_graph
from neurochecker.gui.plotly_map import build_plotly_html
from neurochecker.mask_io import MaskEntry, load_mask_array


class RenderingMixin:
    def _show_current_frame(self) -> None:
        if not self.frame_order or not self.nodes:
            self.scene.clear()
            self.node_label.setText("Frame: -")
            self._branch_options = []
            self._branch_node_id = None
            self._branch_hint = None
            self._update_branch_controls()
            self._update_minimap_highlight(None, None)
            if self.segment_bar is not None:
                self.segment_bar.set_current_frame(None)
            self._refresh_full_skeleton_window()
            return
        frame, node_id = self._current_frame_and_node()
        if frame is None:
            return
        self._current_focus_center_px = None
        self._current_focus_source = None
        mesh_point = self._current_mesh_point()
        if mesh_point is not None:
            x_px, y_px, _, _, _ = mesh_point
            centroid = None
            if self.active_segment_id is not None:
                comp = self._select_component_for_segment_frame(
                    frame,
                    int(self.active_segment_id),
                    center_hint_px=(x_px, y_px),
                )
                if comp is not None:
                    centroid = comp.centroid
            if centroid is not None:
                self._current_focus_center_px = centroid
                self._current_focus_source = "mask-closest"
                focus = self._focus_from_point(centroid[0], centroid[1], frame)
            else:
                self._current_focus_center_px = (x_px, y_px)
                self._current_focus_source = "skeleton"
                focus = self._focus_from_point(x_px, y_px, frame)
        else:
            focus = None
        if focus is None:
            focus = self._last_focus
        else:
            self._last_focus = focus
        self._update_branch_options(node_id)
        self._update_branch_controls()
        self._update_minimap_highlight(frame, node_id)
        self._focus_minimap_view(frame, node_id=node_id)
        if focus is None:
            if self.image_sampler is None:
                self.node_label.setText("Select images directory.")
                return
            full_w = self.image_sampler.original_width
            full_h = self.image_sampler.original_height
            focus = (full_w / 2.0, full_h / 2.0, 512, 512)
        if self.image_sampler is None:
            self.node_label.setText("Select images directory.")
            return
        include_overlay = not (hasattr(self, "fast_scrub_check") and self.fast_scrub_check.isChecked())
        self._render_frame(frame, focus, include_overlay=include_overlay)
        if not include_overlay:
            self._schedule_overlay()
        node_count = len(self.nodes_by_frame.get(frame, []))
        if node_id is not None:
            label_text = (
                f"Frame {frame + 1} ({self.current_frame_index + 1}/{len(self.frame_order)}) | Node {node_id}"
            )
        else:
            label_text = (
                f"Frame {frame + 1} ({self.current_frame_index + 1}/{len(self.frame_order)}) | Nodes {node_count}"
            )
        if self.active_segment_id is not None:
            label_text = f"{label_text} | Segment {self._segment_label(int(self.active_segment_id))}"
        if node_id is None:
            label_text = f"{label_text} | Interpolated"
        if self._branch_options:
            options_text = self._branch_options_display_text()
            if self._branch_hint:
                label_text = f"{label_text} | Branches: {options_text} ({self._branch_hint})"
            else:
                label_text = f"{label_text} | Branches: {options_text}"
        ratio_text = ""
        if self.active_segment_id is not None:
            cur_stats, neigh_stats, final_ratio = self._segment_ratio_stats(
                frame, int(self.active_segment_id)
            )
            ratio_parts = []
            if cur_stats is not None:
                cw, ch, cr = cur_stats
                ratio_parts.append(f"Cur w/h {cr:.2f} ({cw:.0f}x{ch:.0f})")
            if neigh_stats is not None:
                nw, nh, nr = neigh_stats
                ratio_parts.append(f"Neigh w/h {nr:.2f} ({nw:.0f}x{nh:.0f})")
            if final_ratio is not None:
                ratio_parts.append(f"Final w/h {final_ratio:.2f}")
            if ratio_parts:
                ratio_text = " | ".join(ratio_parts)
        if self._current_focus_source == "mask-closest":
            ratio_text = f"{ratio_text} | Center: closest mask" if ratio_text else "Center: closest mask"
        elif self._current_focus_source == "skeleton":
            ratio_text = f"{ratio_text} | Center: skeleton" if ratio_text else "Center: skeleton"
        self.node_label.setText(label_text)
        if hasattr(self, "ratio_label"):
            self.ratio_label.setText(ratio_text)
        if hasattr(self, "status_label"):
            self.status_label.setText(ratio_text)
        if self.minimap_widget is not None:
            status_parts = []
            if self._branch_options:
                options_text = self._branch_options_display_text()
                if self._branch_hint:
                    status_parts.append(f"Branches: {options_text} ({self._branch_hint})")
                else:
                    status_parts.append(f"Branches: {options_text}")
            self.minimap_widget.set_status(" | ".join(status_parts))
        if self.segment_bar is not None:
            self.segment_bar.set_current_frame(frame)
        self._refresh_full_skeleton_window()

    def _refresh_current_view(self) -> None:
        self._overlay_timer.stop()
        self._show_current_frame()

    def _open_map_window(self) -> None:
        frame = None
        current_frame, _ = self._current_frame_and_node()
        if current_frame is not None:
            frame = current_frame
        neuron_name = None
        items = self.neuron_list.selectedItems()
        if items:
            neuron_name = items[0].text()
        title = f"NeuroChecker Graph{(' - ' + neuron_name) if neuron_name else ''}"
        mesh_path = self._find_neuron_mesh(neuron_name) if neuron_name else None
        if not (mesh_path and mesh_path.exists()):
            logger.warning("Open map skipped: mesh not found for neuron=%s", neuron_name)
            return
        logger.info("Open map: neuron=%s frame=%s mesh=%s (mesh skeleton only)", neuron_name, frame, mesh_path)
        mesh_graph = None
        if self.mesh_graph is not None and self._mesh_path == mesh_path:
            mesh_graph = self.mesh_graph
        if mesh_graph is None:
            mesh_graph = _mesh_skeleton_graph(
                mesh_path,
                pixel_size_xy=self.pixel_xy_spin.value(),
                slice_thickness_z=self.slice_z_spin.value(),
            )
        from neurochecker.graph import Node
        nodes: List[Node] = []
        edges: List[Tuple[int, int]] = []
        segments: Optional[Sequence[Sequence[int]]] = None
        segment_colors: Optional[Sequence[str]] = None
        if mesh_graph is not None:
            nodes = list(mesh_graph.nodes)
            edges = list(mesh_graph.edges)
            segments = mesh_graph.paths
            if self.mesh_graph is mesh_graph and self.mesh_segment_colors:
                segment_colors = self.mesh_segment_colors
            else:
                segment_colors = [self._segment_color(i) for i in range(len(segments))]
        else:
            logger.warning("Mesh skeleton unavailable; showing mesh without skeleton.")
        flagged_points = None
        if self.flagged_points:
            flagged_points = []
            for record in self.flagged_points.values():
                if "x" in record and "y" in record and "z" in record:
                    flagged_points.append((record["x"], record["y"], record["z"]))
        html = build_plotly_html(
            nodes,
            edges,
            highlight_frame=frame,
            title=title,
            mesh_path=mesh_path,
            segments=segments,
            segment_colors=segment_colors,
            flagged_points=flagged_points,
        )
        if self._map_html_path is None:
            name = f"neurochecker_map_{uuid.uuid4().hex}.html"
            self._map_html_path = Path(tempfile.gettempdir()) / name
        self._map_html_path.write_text(html, encoding="utf-8")
        webbrowser.open(self._map_html_path.as_uri())

    def _open_full_skeleton_window(self) -> None:
        from neurochecker.gui.full_skeleton_window import FullSkeleton3DWindow

        if self.skeleton_3d_window is None:
            self.skeleton_3d_window = FullSkeleton3DWindow(self)
            self.skeleton_3d_window.viewer.nodeContextRequested.connect(
                self._show_minimap_node_context_menu
            )
        self._refresh_full_skeleton_window()
        self.skeleton_3d_window.show()
        self.skeleton_3d_window.raise_()
        self.skeleton_3d_window.activateWindow()

    def _refresh_full_skeleton_window(self) -> None:
        window = self.skeleton_3d_window
        if window is None:
            return
        if not self.nodes or not self.graph:
            window.clear()
            return

        coords = np.asarray([[n.x, n.y, n.z] for n in self.nodes], dtype=np.float32)
        if coords.size == 0:
            window.clear()
            return
        center = (coords.min(axis=0) + coords.max(axis=0)) * 0.5
        positions = (coords - center).astype(np.float32, copy=False)
        node_ids = [int(n.id) for n in self.nodes]
        edges = list(self.graph.edges)

        _frame, current_node_id = self._current_frame_and_node()
        current_index = None
        if current_node_id is not None and 0 <= current_node_id < len(self.nodes):
            current_index = int(current_node_id)

        active_seg = self.active_segment_id
        branch_options = list(self._branch_options)
        active_color = QtGui.QColor(80, 220, 120, 230) if active_seg is not None else None
        branch_colors = {seg_id: self._segment_qcolor(seg_id) for seg_id in branch_options}

        node_colors: List[QtGui.QColor] = []
        for node in self.nodes:
            color = QtGui.QColor(220, 220, 220, 210)
            segs = self.mesh_node_to_segments.get(node.id, [])
            if active_seg is not None and active_seg in segs and active_color is not None:
                color = active_color
            else:
                for seg_id in branch_options:
                    if seg_id in segs:
                        color = branch_colors.get(seg_id, color)
                        break
            node_colors.append(color)

        edge_colors: List[QtGui.QColor] = []
        for i, j in edges:
            color = QtGui.QColor(150, 180, 200, 170)
            edge = (i, j) if i < j else (j, i)
            if active_seg is not None:
                edge_set = self.mesh_segment_edges.get(active_seg)
                if edge_set and edge in edge_set and active_color is not None:
                    color = active_color
            if active_seg is None or color.alpha() < 230:
                for seg_id in branch_options:
                    edge_set = self.mesh_segment_edges.get(seg_id)
                    if edge_set and edge in edge_set:
                        color = branch_colors.get(seg_id, color)
                        break
            edge_colors.append(color)

        legend_items: List[Tuple[str, QtGui.QColor]] = []
        if active_seg is not None and active_color is not None:
            legend_items.append(("Active", active_color))
        for idx, seg_id in enumerate(branch_options):
            legend_items.append(
                (str(idx + 1), branch_colors.get(seg_id, QtGui.QColor(220, 220, 220, 210)))
            )

        ghost_positions = None
        if self._mesh_preview_points is not None and self._mesh_preview_points.size:
            ghost_positions = (self._mesh_preview_points - center).astype(np.float32, copy=False)

        flagged_positions = None
        if self.flagged_points:
            flagged = [
                (
                    float(record["x"]) - float(center[0]),
                    float(record["y"]) - float(center[1]),
                    float(record["z"]) - float(center[2]),
                )
                for record in self.flagged_points.values()
                if "x" in record and "y" in record and "z" in record
            ]
            if flagged:
                flagged_positions = np.asarray(flagged, dtype=np.float32)

        hillock_positions = None
        if self._hillock_original_node_id is not None:
            current_hillock = self._original_to_current_node_ids.get(int(self._hillock_original_node_id))
            if current_hillock is not None and 0 <= current_hillock < len(positions):
                hillock_positions = np.asarray([positions[current_hillock]], dtype=np.float32)
                legend_items.append(("Hillock", QtGui.QColor(255, 160, 40, 245)))

        distal_positions = None
        if self._distal_original_node_id is not None:
            current_distal = self._original_to_current_node_ids.get(int(self._distal_original_node_id))
            if current_distal is not None and 0 <= current_distal < len(positions):
                distal_positions = np.asarray([positions[current_distal]], dtype=np.float32)
                legend_items.append(("Distal", QtGui.QColor(80, 220, 255, 245)))

        prev_dir = None
        next_dir = None
        if self._segment_frame_points_xyz:
            idx = self.current_frame_index
            if 0 <= idx < len(self._segment_frame_points_xyz):
                cx, cy, cz = self._segment_frame_points_xyz[idx]
                if idx > 0:
                    px, py, pz = self._segment_frame_points_xyz[idx - 1]
                    prev_dir = (px - cx, py - cy, pz - cz)
                if idx + 1 < len(self._segment_frame_points_xyz):
                    nx, ny, nz = self._segment_frame_points_xyz[idx + 1]
                    next_dir = (nx - cx, ny - cy, nz - cz)

        status_parts: List[str] = ["Right-click a node to set hillock/distal"]
        if self._branch_options:
            options_text = self._branch_options_display_text()
            if self._branch_hint:
                status_parts.append(f"Branches: {options_text} ({self._branch_hint})")
            else:
                status_parts.append(f"Branches: {options_text}")
        if hasattr(self, "hillock_status_label"):
            hillock_text = self.hillock_status_label.text().strip()
            if hillock_text:
                status_parts.append(hillock_text)

        neuron_name = self._current_neuron_name or "Neuron"
        window.update_graph(
            positions=positions,
            edges=edges,
            current_index=current_index,
            node_ids=node_ids,
            node_colors=node_colors,
            edge_colors=edge_colors,
            legend_items=legend_items,
            ghost_positions=ghost_positions,
            flagged_positions=flagged_positions,
            hillock_positions=hillock_positions,
            distal_positions=distal_positions,
            prev_dir=prev_dir,
            next_dir=next_dir,
            title=f"NeuroChecker Full Skeleton 3D - {neuron_name}",
            status_text=" | ".join(status_parts),
        )

    def _render_frame(self, frame: int, focus: Tuple[float, float, int, int], *, include_overlay: bool) -> None:
        if self.image_sampler is None:
            return
        viewport_size = self.view.viewport().size()
        target_w = max(1, viewport_size.width())
        target_h = max(1, viewport_size.height())
        center_x, center_y, base_w, base_h = focus
        crop_w, crop_h = self._crop_size_for_frame(base_w, base_h)
        x0, y0, crop_w, crop_h = self._clamp_viewport_center(center_x, center_y, crop_w, crop_h)
        target_w, target_h = self._fit_target_size(crop_w, crop_h, target_w, target_h)

        image, scale_factor = self.image_sampler.get_viewport_image(
            frame, (x0, y0, crop_w, crop_h), (target_w, target_h)
        )
        self._last_view_context = (frame, (x0, y0, crop_w, crop_h), (target_w, target_h), image, scale_factor)
        if include_overlay:
            overlay = self._build_overlay(frame, (x0, y0, crop_w, crop_h), scale_factor, (target_w, target_h))
            if overlay is not None:
                image = self._blend_overlay(image, overlay)
        focus_overlay = self._build_focus_overlay((x0, y0, crop_w, crop_h), scale_factor, (target_w, target_h))
        if focus_overlay is not None:
            image = self._blend_overlay(image, focus_overlay)
        marker_overlay = self._build_marker_overlay((x0, y0, crop_w, crop_h), scale_factor, (target_w, target_h))
        if marker_overlay is not None:
            image = self._blend_overlay(image, marker_overlay)
        if self._is_flagged_frame(frame):
            flag_overlay = self._build_flag_frame_overlay((target_w, target_h))
            image = self._blend_overlay(image, flag_overlay)
        self._show_image(image)

    def _schedule_overlay(self) -> None:
        if self.overlay_alpha.value() <= 0:
            return
        self._overlay_timer.start(140)

    def _apply_pending_overlay(self) -> None:
        if self._last_view_context is None:
            return
        if not self.frame_order:
            return
        current_frame, _ = self._current_frame_and_node()
        if current_frame is None:
            return
        frame, viewport_rect, target_size, base_image, scale_factor = self._last_view_context
        if frame != current_frame:
            return
        overlay = self._build_overlay(frame, viewport_rect, scale_factor, target_size)
        if overlay is None:
            self._show_image(base_image)
            return
        image = self._blend_overlay(base_image, overlay)
        focus_overlay = self._build_focus_overlay(viewport_rect, scale_factor, target_size)
        if focus_overlay is not None:
            image = self._blend_overlay(image, focus_overlay)
        marker_overlay = self._build_marker_overlay(viewport_rect, scale_factor, target_size)
        if marker_overlay is not None:
            image = self._blend_overlay(image, marker_overlay)
        if self._is_flagged_frame(frame):
            flag_overlay = self._build_flag_frame_overlay(target_size)
            image = self._blend_overlay(image, flag_overlay)
        self._show_image(image)

    def _crop_size_for_frame(self, base_w: int, base_h: int) -> Tuple[int, int]:
        if base_w <= 0 or base_h <= 0:
            return 256, 256
        return base_w, base_h

    def _clamp_viewport_center(self, center_x: float, center_y: float, crop_w: int, crop_h: int) -> Tuple[int, int, int, int]:
        if self.image_sampler is not None:
            full_w = self.image_sampler.original_width
            full_h = self.image_sampler.original_height
        else:
            full_w = max((e.full_width for e in self.entries), default=crop_w)
            full_h = max((e.full_height for e in self.entries), default=crop_h)
        crop_w = min(crop_w, full_w)
        crop_h = min(crop_h, full_h)
        x0 = int(round(center_x - crop_w / 2))
        y0 = int(round(center_y - crop_h / 2))
        x0 = max(0, min(x0, full_w - crop_w))
        y0 = max(0, min(y0, full_h - crop_h))
        return x0, y0, crop_w, crop_h

    def _fit_target_size(self, crop_w: int, crop_h: int, view_w: int, view_h: int) -> Tuple[int, int]:
        if crop_h <= 0 or crop_w <= 0:
            return view_w, view_h
        crop_aspect = crop_w / float(crop_h)
        view_aspect = view_w / float(view_h) if view_h else crop_aspect
        if crop_aspect >= view_aspect:
            target_w = view_w
            target_h = int(round(view_w / crop_aspect))
        else:
            target_h = view_h
            target_w = int(round(view_h * crop_aspect))
        return max(1, target_w), max(1, target_h)

    def _get_masks_for_frame(self, frame: int) -> List[Tuple[MaskEntry, np.ndarray]]:
        cached = self.mask_cache.get(frame)
        if cached is not None:
            return cached
        masks: List[Tuple[MaskEntry, np.ndarray]] = []
        selected_runs = set(self._selected_runs() or [])
        for entry in self.entries_by_frame.get(frame, []):
            if selected_runs and entry.run_id not in selected_runs:
                continue
            mask = load_mask_array(entry)
            if mask is None:
                continue
            masks.append((entry, mask))
        self.mask_cache.set(frame, masks)
        return masks

    def _segment_for_centroid(self, entry: MaskEntry, centroid: Tuple[float, float]) -> int:
        if self.mesh_kdtree is None or self.mesh_kdtree_segment_ids is None:
            return -1
        cy, cx = centroid
        px = float(self.pixel_xy_spin.value())
        sz = float(self.slice_z_spin.value())
        x_phys = (entry.x + float(cx)) * px
        y_phys = (entry.y + float(cy)) * px
        z_phys = float(entry.frame) * sz
        _, idx = self.mesh_kdtree.query([x_phys, y_phys, z_phys], k=1)
        return int(self.mesh_kdtree_segment_ids[int(idx)])

    def _get_components_for_frame(self, frame: int) -> List[ComponentInfo]:
        cached = self.component_cache.get(frame)
        if cached is not None:
            return cached
        masks = self._get_masks_for_frame(frame)
        if not masks:
            self.component_cache.set(frame, [])
            return []
        components: List[ComponentInfo] = []
        mask_count = 0
        for entry, mask in masks:
            mask_count += 1
            labeled, num = nd_label(mask > 0)
            if num <= 0:
                continue
            slices = find_objects(labeled)
            for label_id in range(1, num + 1):
                slc = slices[label_id - 1]
                if slc is None:
                    continue
                roi = labeled[slc] == label_id
                area = int(roi.sum())
                if area == 0:
                    continue
                local_ys, local_xs = np.nonzero(roi)
                cx = entry.x + float(slc[1].start + local_xs.mean())
                cy = entry.y + float(slc[0].start + local_ys.mean())
                local_centroid = (slc[0].start + float(local_ys.mean()),
                                  slc[1].start + float(local_xs.mean()))
                seg_id = self._segment_for_centroid(entry, local_centroid)
                components.append(
                    ComponentInfo(
                        entry=entry,
                        label_id=label_id,
                        labeled=labeled,
                        slices=(slc[0], slc[1]),
                        segment_id=seg_id,
                        centroid=(cx, cy),
                        area=area,
                    )
                )
        self.component_cache.set(frame, components)
        return components

    def _get_components_for_segment_frame(
        self, frame: int, segment_id: int
    ) -> List[ComponentInfo]:
        """Load components only for mask entries relevant to *segment_id* on *frame*.

        Falls back to the full ``_get_components_for_frame`` if no
        pre-computed entry map is available.
        """
        key = (segment_id, frame)
        entry_indices = self._segment_entry_map.get(key)
        if entry_indices is None:
            return [
                c
                for c in self._get_components_for_frame(frame)
                if c.segment_id == segment_id
            ]
        cached = self.component_cache.get(frame)
        if cached is not None:
            return [c for c in cached if c.segment_id == segment_id]
        all_entries = self.entries_by_frame.get(frame, [])
        selected_runs = set(self._selected_runs() or [])
        components: List[ComponentInfo] = []
        for idx in entry_indices:
            if idx >= len(all_entries):
                continue
            entry = all_entries[idx]
            if selected_runs and entry.run_id not in selected_runs:
                continue
            mask = load_mask_array(entry)
            if mask is None:
                continue
            labeled, num = nd_label(mask > 0)
            if num <= 0:
                continue
            slices = find_objects(labeled)
            for label_id in range(1, num + 1):
                slc = slices[label_id - 1]
                if slc is None:
                    continue
                roi = labeled[slc] == label_id
                area = int(roi.sum())
                if area == 0:
                    continue
                local_ys, local_xs = np.nonzero(roi)
                cx = entry.x + float(slc[1].start + local_xs.mean())
                cy = entry.y + float(slc[0].start + local_ys.mean())
                local_centroid = (slc[0].start + float(local_ys.mean()),
                                  slc[1].start + float(local_xs.mean()))
                seg_id = self._segment_for_centroid(entry, local_centroid)
                components.append(
                    ComponentInfo(
                        entry=entry,
                        label_id=label_id,
                        labeled=labeled,
                        slices=(slc[0], slc[1]),
                        segment_id=seg_id,
                        centroid=(cx, cy),
                        area=area,
                    )
                )
        return [c for c in components if c.segment_id == segment_id]

    def _build_marker_overlay(
        self,
        viewport_rect: Tuple[int, int, int, int],
        scale_factor: float,
        target_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if not self.frame_order:
            return None
        vx, vy, _, _ = viewport_rect
        tw, th = target_size
        mesh_point = self._current_mesh_point()
        if mesh_point is None:
            return None
        x_px, y_px, _, _, _ = mesh_point
        ox = int(round((x_px - vx) / scale_factor))
        oy = int(round((y_px - vy) / scale_factor))
        if ox < -10 or oy < -10 or ox > tw + 10 or oy > th + 10:
            return None
        overlay = np.zeros((th, tw, 4), dtype=np.uint8)
        radius = 6
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                tx = ox + dx
                ty = oy + dy
                if 0 <= tx < tw and 0 <= ty < th:
                    overlay[ty, tx] = [255, 0, 255, 220]
        frame = self.frame_order[self.current_frame_index]
        marker_specs = [
            (self._hillock_original_node_id, [255, 160, 40, 180], 8),
            (self._distal_original_node_id, [80, 220, 255, 180], 7),
        ]
        for original_node_id, color, marker_radius in marker_specs:
            if original_node_id is None:
                continue
            current_node_id = self._original_to_current_node_ids.get(int(original_node_id))
            if current_node_id is None or not (0 <= current_node_id < len(self.nodes)):
                continue
            node = self.nodes[current_node_id]
            if int(node.frame) != int(frame):
                continue
            mx = int(round((float(node.x_px) - vx) / scale_factor))
            my = int(round((float(node.y_px) - vy) / scale_factor))
            for dy in range(-marker_radius, marker_radius + 1):
                for dx in range(-marker_radius, marker_radius + 1):
                    dist2 = dx * dx + dy * dy
                    if dist2 > marker_radius * marker_radius:
                        continue
                    if dist2 < max(0, marker_radius - 3) ** 2:
                        continue
                    tx = mx + dx
                    ty = my + dy
                    if 0 <= tx < tw and 0 <= ty < th:
                        overlay[ty, tx] = color
        return overlay

    def _build_focus_overlay(
        self,
        viewport_rect: Tuple[int, int, int, int],
        scale_factor: float,
        target_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if self._current_focus_center_px is None:
            return None
        vx, vy, _, _ = viewport_rect
        tw, th = target_size
        cx, cy = self._current_focus_center_px
        ox = int(round((cx - vx) / scale_factor))
        oy = int(round((cy - vy) / scale_factor))
        if ox < -10 or oy < -10 or ox > tw + 10 or oy > th + 10:
            return None
        overlay = np.zeros((th, tw, 4), dtype=np.uint8)
        radius = 7
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                tx = ox + dx
                ty = oy + dy
                if 0 <= tx < tw and 0 <= ty < th:
                    overlay[ty, tx] = [60, 120, 255, 210]
        return overlay

    def _build_overlay(
        self,
        frame: int,
        viewport_rect: Tuple[int, int, int, int],
        scale_factor: float,
        target_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        masks = self._get_masks_for_frame(frame)
        if not masks:
            return None
        vx, vy, vw, vh = viewport_rect
        tw, th = target_size
        alpha_val = int(self.overlay_alpha.value())
        overlay = np.zeros((th, tw, 4), dtype=np.uint8)
        components = self._display_components_for_frame(frame)
        placed_count = 0
        outline_count = 0
        skip_bounds = 0
        skip_empty = 0
        skip_scale = 0
        for comp in components:
            entry = comp.entry
            slc_y, slc_x = comp.slices
            comp_x0 = entry.x + slc_x.start
            comp_y0 = entry.y + slc_y.start
            comp_x1 = entry.x + slc_x.stop
            comp_y1 = entry.y + slc_y.stop

            ix0 = max(comp_x0, vx)
            iy0 = max(comp_y0, vy)
            ix1 = min(comp_x1, vx + vw)
            iy1 = min(comp_y1, vy + vh)
            if ix1 <= ix0 or iy1 <= iy0:
                skip_bounds += 1
                continue
            crop_x0 = ix0 - entry.x
            crop_y0 = iy0 - entry.y
            crop_x1 = crop_x0 + (ix1 - ix0)
            crop_y1 = crop_y0 + (iy1 - iy0)
            mask_crop = comp.labeled[crop_y0:crop_y1, crop_x0:crop_x1] == comp.label_id
            if mask_crop.size == 0 or not mask_crop.any():
                skip_empty += 1
                continue
            target_w = max(1, int(round(mask_crop.shape[1] / scale_factor)))
            target_h = max(1, int(round(mask_crop.shape[0] / scale_factor)))
            if target_w <= 0 or target_h <= 0:
                skip_scale += 1
                continue
            mask_img = Image.fromarray(mask_crop.astype(np.uint8) * 255)
            mask_resized = np.asarray(mask_img.resize((target_w, target_h), Image.NEAREST)) > 0
            ox0 = int(round((ix0 - vx) / scale_factor))
            oy0 = int(round((iy0 - vy) / scale_factor))
            ox1 = min(tw, ox0 + target_w)
            oy1 = min(th, oy0 + target_h)
            if ox1 <= ox0 or oy1 <= oy0:
                skip_bounds += 1
                continue
            mask_resized = mask_resized[: oy1 - oy0, : ox1 - ox0]
            if entry.mask_id in self.flagged_masks:
                color = (255, 80, 80)
            else:
                color = entry.color or (255, 128, 0)
            sub = overlay[oy0:oy1, ox0:ox1]
            if alpha_val > 0:
                sub[mask_resized] = [color[0], color[1], color[2], alpha_val]
            outline = _mask_outline(mask_resized)
            if outline.any():
                sub[outline] = [color[0], color[1], color[2], 255]
                outline_count += 1
            placed_count += 1
        logger.info(
            "Overlay: frame=%s active_segment=%s comps=%d placed=%d outlines=%d "
            "skip_bounds=%d skip_empty=%d skip_scale=%d "
            "viewport=%s scale=%.3f size=%s",
            frame,
            self.active_segment_id,
            len(components),
            placed_count,
            outline_count,
            skip_bounds,
            skip_empty,
            skip_scale,
            viewport_rect,
            scale_factor,
            target_size,
        )
        return overlay

    def _blend_overlay(self, image: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        if overlay is None:
            return image
        if overlay.shape[:2] != image.shape[:2]:
            return image
        rgb = image.astype(np.float32)
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        overlay_rgb = overlay[:, :, :3].astype(np.float32)
        blended = rgb * (1.0 - alpha) + overlay_rgb * alpha
        return blended.round().astype(np.uint8)

    def _is_flagged_frame(self, frame: int) -> bool:
        if self.active_segment_id is None:
            return False
        key = (int(self.active_segment_id), int(frame))
        return key in self.flagged_points

    def _build_flag_frame_overlay(self, target_size: Tuple[int, int]) -> np.ndarray:
        tw, th = target_size
        overlay = np.zeros((th, tw, 4), dtype=np.uint8)
        t = max(1, int(FLAG_FRAME_BORDER_PX))
        overlay[:t, :, :] = [255, 60, 60, 255]
        overlay[-t:, :, :] = [255, 60, 60, 255]
        overlay[:, :t, :] = [255, 60, 60, 255]
        overlay[:, -t:, :] = [255, 60, 60, 255]
        return overlay

    def _show_image(self, image: np.ndarray) -> None:
        self.scene.clear()
        qimg = numpy_to_qimage(image)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)
        self.scene.setSceneRect(item.boundingRect())
        self.view.fitInView(item.boundingRect(), QtCore.Qt.KeepAspectRatio)
        if hasattr(self, "minimap_panel"):
            self.minimap_panel.raise_()
        self.view.setFocus(QtCore.Qt.OtherFocusReason)
