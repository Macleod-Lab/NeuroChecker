import colorsys
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.spatial import cKDTree

from neurochecker.graph import GraphResult
from neurochecker.gui.constants import logger


class NavigationMixin:
    def _segment_label(self, segment_id: int) -> str:
        if getattr(self, "_soma_segment_id", None) == int(segment_id):
            return "Soma"
        return f"B{int(segment_id)}"

    def _refresh_navigation(self, *, reset: bool = False) -> None:
        if not self._mesh_nav_enabled:
            self.frame_order = []
            self.current_frame_index = 0
            self.node_slider.blockSignals(True)
            self.node_slider.setRange(0, 0)
            self.node_slider.setValue(0)
            self.node_slider.blockSignals(False)
            return
        self.frame_order = list(self._segment_frame_order) if self._segment_frame_order else []
        if not self.frame_order:
            self.current_frame_index = 0
            self.node_slider.blockSignals(True)
            self.node_slider.setRange(0, 0)
            self.node_slider.setValue(0)
            self.node_slider.blockSignals(False)
            return
        if self._segment_anchor_node_id is not None:
            idx = None
            for i, node_id in enumerate(self._segment_frame_node_ids):
                if node_id == self._segment_anchor_node_id:
                    idx = i
                    break
            self.current_frame_index = idx if idx is not None else 0
            self._segment_anchor_node_id = None
        elif reset or self.current_frame_index >= len(self.frame_order):
            self.current_frame_index = 0
        self.node_slider.blockSignals(True)
        self.node_slider.setRange(0, max(0, len(self.frame_order) - 1))
        self.node_slider.setValue(self.current_frame_index)
        self.node_slider.blockSignals(False)

    def _set_active_segment(self, segment_id: int, *, anchor_node_id: Optional[int] = None) -> None:
        if segment_id < 0 or segment_id >= len(self.mesh_segments):
            return
        self.active_segment_id = segment_id
        self._segment_nodes = list(self.mesh_segments[segment_id])
        self._segment_anchor_node_id = anchor_node_id
        self._build_segment_frame_path(segment_id)
        logger.info(
            "Active segment set: segment=%s anchor_node=%s frames=%d",
            segment_id,
            anchor_node_id,
            len(self._segment_frame_order),
        )
        self._update_segment_bar()
        self._refresh_navigation(reset=True)
        self._show_current_frame()

    def _branch_options_display_text(self) -> str:
        if not self._branch_options:
            return ""
        return ", ".join(
            f"{idx + 1}={self._segment_label(int(seg_id))}" for idx, seg_id in enumerate(self._branch_options)
        )

    def _clear_branch_option_buttons(self) -> None:
        while self.branch_nav_buttons_layout.count():
            item = self.branch_nav_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._branch_option_buttons = []

    def _update_branch_controls(self) -> None:
        if not hasattr(self, "branch_nav_widget"):
            return
        if not self._mesh_nav_enabled:
            self.branch_nav_widget.setVisible(False)
            return
        self.branch_nav_widget.setVisible(True)
        self._clear_branch_option_buttons()
        if not self._branch_options:
            if self._branch_hint:
                self.branch_nav_label.setText(f"Branches: none ({self._branch_hint})")
            else:
                self.branch_nav_label.setText("Branches: none")
            return
        label = "Branches:"
        if self._branch_hint:
            label = f"Branches ({self._branch_hint}):"
        self.branch_nav_label.setText(label)
        for idx, seg_id in enumerate(self._branch_options):
            seg_label = self._segment_label(int(seg_id))
            btn = QtWidgets.QPushButton(f"{idx + 1}: {seg_label}")
            btn.setToolTip(f"Switch to {seg_label}")
            btn.setMaximumHeight(24)
            color = self._segment_qcolor(int(seg_id), alpha=255)
            r = color.red()
            g = color.green()
            b = color.blue()
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "#101010" if luminance >= 150.0 else "#f0f0f0"
            btn.setStyleSheet(
                f"QPushButton {{ background: rgb({r}, {g}, {b}); color: {text_color}; "
                "border: 1px solid rgba(255,255,255,110); border-radius: 4px; padding: 2px 6px; }}"
                "QPushButton:hover { border: 1px solid rgba(255,255,255,190); }"
            )
            btn.clicked.connect(lambda _checked=False, option_index=idx: self._select_branch_option(option_index))
            self._branch_option_buttons.append(btn)
            self.branch_nav_buttons_layout.addWidget(btn)
        self.branch_nav_buttons_layout.addStretch(1)

    def _try_follow_branch_from_boundary(self, *, direction: str) -> bool:
        if not self._branch_options:
            self._update_branch_options(None)
            self._update_branch_controls()
        if not self._branch_options:
            return False
        if len(self._branch_options) == 1:
            seg_id = int(self._branch_options[0])
            logger.info("Boundary branch follow: direction=%s -> segment=%s", direction, seg_id)
            self._select_branch_option(0)
            return True
        guidance = "Multiple branches: click a branch chip or press 1-9."
        if hasattr(self, "status_label"):
            self.status_label.setText(guidance)
        return False

    def _update_branch_options(self, node_id: Optional[int]) -> None:
        self._branch_options = []
        self._branch_node_id = None
        self._branch_hint = None
        if node_id is None:
            if not self._segment_frame_node_ids:
                return
            if not self.mesh_node_to_segments:
                return
            max_search = 25
            idx0 = self.current_frame_index
            found_node = None
            found_options: List[int] = []
            found_dist = None
            for dist in range(0, max_search + 1):
                for direction in (0, -1, 1):
                    if dist == 0 and direction != 0:
                        continue
                    idx = idx0 + direction * dist
                    if idx < 0 or idx >= len(self._segment_frame_node_ids):
                        continue
                    candidate = self._segment_frame_node_ids[idx]
                    if candidate is None:
                        continue
                    options = list(self.mesh_node_to_segments.get(candidate, []))
                    if self.active_segment_id is not None:
                        options = [seg for seg in options if seg != self.active_segment_id]
                    if options:
                        found_node = candidate
                        found_options = options
                        found_dist = dist
                        break
                if found_node is not None:
                    break
            if found_node is not None:
                self._branch_options = found_options
                self._branch_node_id = found_node
                self._branch_hint = f"{found_dist}f away" if found_dist is not None else None
                logger.info(
                    "Branch options (nearest): node=%s active_segment=%s options=%s distance=%s",
                    found_node,
                    self.active_segment_id,
                    self._branch_options,
                    found_dist,
                )
            return
        if not self.mesh_node_to_segments:
            return
        options = list(self.mesh_node_to_segments.get(node_id, []))
        if self.active_segment_id is not None:
            options = [seg for seg in options if seg != self.active_segment_id]
        if options:
            self._branch_options = options
            self._branch_node_id = node_id
        logger.info(
            "Branch options: node=%s active_segment=%s options=%s",
            node_id,
            self.active_segment_id,
            self._branch_options,
        )

    def _select_branch_option(self, option_index: int) -> None:
        if not self._branch_options:
            return
        if option_index < 0 or option_index >= len(self._branch_options):
            return
        seg_id = self._branch_options[option_index]
        logger.info(
            "Branch selected: option_index=%s segment=%s anchor_node=%s",
            option_index,
            seg_id,
            self._branch_node_id,
        )
        self._set_active_segment(seg_id, anchor_node_id=self._branch_node_id)

    def _segment_frame_path(
        self, segment_id: int
    ) -> Tuple[List[int], List[Tuple[float, float, float]], List[Tuple[float, float]], List[Optional[int]]]:
        """Walk a segment's nodes and group by frame.

        With the anisotropic skeleton every node already sits on an actual
        frame so this is a simple group-and-average -- no interpolation.
        """
        if segment_id < 0 or segment_id >= len(self.mesh_segments):
            return [], [], [], []
        if not self.nodes:
            return [], [], [], []
        seg_nodes = list(self.mesh_segments[segment_id])
        if not seg_nodes:
            return [], [], [], []
        px = float(self.pixel_xy_spin.value()) if float(self.pixel_xy_spin.value()) > 0 else 1.0

        accum: Dict[int, List[float]] = {}
        counts: Dict[int, int] = {}
        candidates: Dict[int, List[int]] = {}

        for node_id in seg_nodes:
            node = self.nodes[node_id]
            frame = int(node.frame)
            if frame not in accum:
                accum[frame] = [0.0, 0.0, 0.0]
                counts[frame] = 0
                candidates[frame] = []
            accum[frame][0] += float(node.x)
            accum[frame][1] += float(node.y)
            accum[frame][2] += float(node.z)
            counts[frame] += 1
            candidates[frame].append(node_id)

        if not accum:
            return [], [], [], []

        frames_out = sorted(accum.keys())
        points_xyz_out: List[Tuple[float, float, float]] = []
        points_px_out: List[Tuple[float, float]] = []
        node_ids_out: List[Optional[int]] = []
        for frame in frames_out:
            count = counts[frame] or 1
            sx, sy, sz = accum[frame]
            x = sx / count
            y = sy / count
            z = sz / count
            points_xyz_out.append((x, y, z))
            points_px_out.append((x / px, y / px))
            cand = candidates[frame]
            chosen = None
            if cand:
                best = None
                best_deg = -1
                for nid in cand:
                    deg = len(self.mesh_node_to_segments.get(nid, []))
                    if deg > best_deg:
                        best_deg = deg
                        best = nid
                chosen = best
            node_ids_out.append(chosen)

        return frames_out, points_xyz_out, points_px_out, node_ids_out

    def _build_segment_frame_path(self, segment_id: int) -> None:
        frames, points_xyz, points_px, node_ids = self._segment_frame_path(segment_id)
        self._segment_frame_order = frames
        self._segment_frame_points_xyz = points_xyz
        self._segment_frame_points_px = points_px
        self._segment_frame_node_ids = node_ids

    def _init_mesh_segments(
        self,
        graph: GraphResult,
        *,
        segment_nodes: Optional[List[List[int]]] = None,
        segment_edges: Optional[Dict[int, set]] = None,
        preferred_active_segment_id: Optional[int] = None,
    ) -> None:
        if segment_nodes is None:
            self.mesh_segments = [path for path in graph.paths if len(path) >= 2]
        else:
            self.mesh_segments = [list(path) for path in segment_nodes if path]
        self.mesh_segment_colors = [self._segment_color(idx) for idx in range(len(self.mesh_segments))]
        self.mesh_node_to_segments = {}
        self.mesh_segment_edges = {}
        for seg_id, path in enumerate(self.mesh_segments):
            edges = set()
            for node_id in path:
                self.mesh_node_to_segments.setdefault(node_id, []).append(seg_id)
            if segment_edges and seg_id in segment_edges:
                edges = {tuple(edge) for edge in segment_edges[seg_id]}
            else:
                for a, b in zip(path, path[1:]):
                    if a == b:
                        continue
                    edge = (a, b) if a < b else (b, a)
                    edges.add(edge)
            self.mesh_segment_edges[seg_id] = edges
        self._build_mesh_kdtree()
        self._build_segment_entry_map()
        if self.mesh_segments:
            active_id = preferred_active_segment_id
            if active_id is None or active_id < 0 or active_id >= len(self.mesh_segments):
                active_id = max(range(len(self.mesh_segments)), key=lambda i: len(self.mesh_segments[i]))
            self.active_segment_id = active_id
            self._segment_nodes = list(self.mesh_segments[active_id])
            self._segment_anchor_node_id = self._segment_nodes[0] if self._segment_nodes else None
            self._build_segment_frame_path(active_id)
        else:
            self.active_segment_id = None
            self._segment_nodes = []
            self._segment_anchor_node_id = None
            self._segment_frame_order = []
            self._segment_frame_points_px = []
            self._segment_frame_points_xyz = []
            self._segment_frame_node_ids = []
        self._update_segment_bar()

    def _build_mesh_kdtree(self) -> None:
        self.mesh_kdtree = None
        self.mesh_kdtree_segment_ids = None
        if not self.nodes or not self.mesh_node_to_segments:
            return
        coords = []
        seg_ids = []
        for node_id, segs in self.mesh_node_to_segments.items():
            if len(segs) != 1:
                continue
            node = self.nodes[node_id]
            coords.append([node.x, node.y, node.z])
            seg_ids.append(segs[0])
        if not coords:
            for node_id, segs in self.mesh_node_to_segments.items():
                node = self.nodes[node_id]
                coords.append([node.x, node.y, node.z])
                seg_ids.append(segs[0])
        if not coords:
            return
        self.mesh_kdtree = cKDTree(np.asarray(coords, dtype=np.float64))
        self.mesh_kdtree_segment_ids = np.asarray(seg_ids, dtype=np.int32)

    def _build_segment_entry_map(self) -> None:
        """Pre-compute which mask entries are relevant for each (segment, frame).

        For every segment, walk its skeleton nodes to get per-frame XY
        positions in pixel space, then check which ``MaskEntry`` bounding
        boxes on that frame contain (or are near) the skeleton position.
        The result is stored in ``_segment_entry_map`` keyed by
        ``(segment_id, frame)`` -> list of entry indices into
        ``entries_by_frame[frame]``.
        """
        self._segment_entry_map = {}
        if not self.mesh_segments or not self.entries_by_frame:
            return
        px = float(self.pixel_xy_spin.value()) if float(self.pixel_xy_spin.value()) > 0 else 1.0
        margin = 50
        for seg_id, path in enumerate(self.mesh_segments):
            frame_positions: Dict[int, Tuple[float, float]] = {}
            for node_id in path:
                node = self.nodes[node_id]
                frame = int(node.frame)
                if frame not in frame_positions:
                    frame_positions[frame] = (0.0, 0.0)
                ox, oy = frame_positions[frame]
                frame_positions[frame] = (ox + float(node.x) / px, oy + float(node.y) / px)
            frame_counts: Dict[int, int] = {}
            for node_id in path:
                frame = int(self.nodes[node_id].frame)
                frame_counts[frame] = frame_counts.get(frame, 0) + 1
            for frame, (sx, sy) in frame_positions.items():
                count = frame_counts.get(frame, 1)
                cx, cy = sx / count, sy / count
                entries = self.entries_by_frame.get(frame, [])
                relevant = []
                for idx, entry in enumerate(entries):
                    ex0 = entry.x - margin
                    ey0 = entry.y - margin
                    ex1 = entry.x + entry.width + margin
                    ey1 = entry.y + entry.height + margin
                    if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
                        relevant.append(idx)
                if relevant:
                    self._segment_entry_map[(seg_id, frame)] = relevant

    def _on_slider_changed(self, value: int) -> None:
        self.current_frame_index = int(value)
        self._show_current_frame()

    def _prev_frame(self) -> None:
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.node_slider.setValue(self.current_frame_index)
            return
        self._try_follow_branch_from_boundary(direction="backward")

    def _next_frame(self) -> None:
        if self.current_frame_index < len(self.frame_order) - 1:
            self.current_frame_index += 1
            self.node_slider.setValue(self.current_frame_index)
            return
        self._try_follow_branch_from_boundary(direction="forward")

    def _jump_to_segment_frame(self, target_frame: int) -> None:
        if not self.frame_order:
            return
        try:
            idx = self.frame_order.index(target_frame)
        except ValueError:
            best_idx = 0
            best_dist = None
            for i, frame in enumerate(self.frame_order):
                dist = abs(frame - target_frame)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = i
            idx = best_idx
        self.current_frame_index = int(idx)
        self.node_slider.blockSignals(True)
        self.node_slider.setValue(self.current_frame_index)
        self.node_slider.blockSignals(False)
        self._show_current_frame()

    def _segment_color(self, segment_id: int) -> str:
        rng = random.Random(segment_id + 2718)
        hue = rng.random()
        sat = 0.6
        val = 0.9
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"

    def _segment_qcolor(self, segment_id: int, *, alpha: int = 230) -> QtGui.QColor:
        text = None
        if 0 <= segment_id < len(self.mesh_segment_colors):
            text = self.mesh_segment_colors[segment_id]
        if not text:
            text = self._segment_color(segment_id)
        try:
            start = text.find("(")
            end = text.find(")")
            if start >= 0 and end > start:
                parts = text[start + 1 : end].split(",")
                rgb = [int(float(p.strip())) for p in parts[:3]]
                return QtGui.QColor(rgb[0], rgb[1], rgb[2], alpha)
        except Exception:
            pass
        return QtGui.QColor(200, 200, 200, alpha)

    def _compute_edge_median_px(self) -> float:
        if not self.graph or not self.graph.edges:
            return 0.0
        dists = []
        for i, j in self.graph.edges:
            if i >= len(self.nodes) or j >= len(self.nodes):
                continue
            a = self.nodes[i]
            b = self.nodes[j]
            dx = a.x_px - b.x_px
            dy = a.y_px - b.y_px
            dists.append((dx * dx + dy * dy) ** 0.5)
        if not dists:
            return 0.0
        dists.sort()
        mid = len(dists) // 2
        return float(dists[mid])

    def _update_segment_bar(self) -> None:
        if self.segment_bar is None:
            return
        if not self.mesh_segments:
            self.segment_bar.setVisible(False)
            return
        segments: List[Tuple[int, int, int, QtGui.QColor]] = []
        for seg_id, path in enumerate(self.mesh_segments):
            if not path:
                continue
            frames = [self.nodes[node_id].frame for node_id in path if 0 <= node_id < len(self.nodes)]
            if not frames:
                continue
            start = int(min(frames))
            end = int(max(frames))
            color = self._segment_qcolor(seg_id, alpha=255)
            segments.append((seg_id, start, end, color))
        if not segments:
            self.segment_bar.setVisible(False)
            return
        self.segment_bar.set_segments(segments)
        self.segment_bar.set_active(self.active_segment_id)
        self.segment_bar.setVisible(True)

    def _on_segment_bar_clicked(self, segment_id: int, frame: int) -> None:
        if not self.mesh_segments:
            return
        if self.active_segment_id != segment_id:
            self._set_active_segment(segment_id)
        self._jump_to_segment_frame(frame)

    def _segments_at_frame(self, frame: int) -> List[int]:
        """Return all segment IDs that have at least one node on *frame*."""
        hits: List[int] = []
        for seg_id, path in enumerate(self.mesh_segments):
            for nid in path:
                if 0 <= nid < len(self.nodes) and self.nodes[nid].frame == frame:
                    hits.append(seg_id)
                    break
        return hits

    def _goto_frame(self) -> None:
        target = self.goto_frame_spin.value()
        segs = self._segments_at_frame(target) if self.mesh_segments else []
        if not segs:
            if self.frame_order:
                self._jump_to_segment_frame(target)
            return
        if len(segs) == 1:
            seg_id = segs[0]
            if self.active_segment_id != seg_id:
                self._set_active_segment(seg_id)
            self._jump_to_segment_frame(target)
            return
        self._show_branch_chooser_for_frame(target, segs)

    def _show_branch_chooser_for_frame(self, frame: int, seg_ids: List[int]) -> None:
        menu = QtWidgets.QMenu(self)
        menu.setTitle(f"Branches at frame {frame}")
        for seg_id in seg_ids:
            color = self._segment_qcolor(seg_id, alpha=255)
            node_count = len(self.mesh_segments[seg_id])
            seg_nodes = [self.nodes[nid] for nid in self.mesh_segments[seg_id] if 0 <= nid < len(self.nodes)]
            frames = [n.frame for n in seg_nodes]
            fr = f"{min(frames)}\u2013{max(frames)}" if frames else ""
            label = f"{self._segment_label(seg_id)}  ({node_count} nodes, frames {fr})"
            if seg_id == self.active_segment_id:
                label += "  [active]"
            action = menu.addAction(label)
            px = QtGui.QPixmap(14, 14)
            px.fill(color)
            action.setIcon(QtGui.QIcon(px))
            action.triggered.connect(
                lambda _checked=False, s=seg_id, f=frame: self._goto_frame_on_segment(f, s)
            )
        pos = self.goto_frame_spin.mapToGlobal(
            QtCore.QPoint(0, self.goto_frame_spin.height())
        )
        menu.exec_(pos)

    def _goto_frame_on_segment(self, frame: int, seg_id: int) -> None:
        if self.active_segment_id != seg_id:
            self._set_active_segment(seg_id)
        self._jump_to_segment_frame(frame)
