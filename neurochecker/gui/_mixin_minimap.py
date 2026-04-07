from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from neurochecker.graph import Node
from neurochecker.gui.constants import MESH_PREVIEW_LOCAL_MAX_POINTS, logger
from neurochecker.gui.widgets import MiniMap3DWidget


class MinimapMixin:
    def _build_minimap(self) -> None:
        self.minimap_panel = QtWidgets.QFrame(self.view)
        self.minimap_panel.setObjectName("minimapPanel")
        self.minimap_panel.setStyleSheet(
            "QFrame#minimapPanel { background: rgba(18, 18, 18, 220); "
            "border: 1px solid #666; border-radius: 8px; }"
        )
        self.minimap_panel.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self.minimap_panel.setFixedSize(520, 520)
        layout = QtWidgets.QVBoxLayout(self.minimap_panel)
        layout.setContentsMargins(6, 6, 6, 6)
        self.minimap_widget = MiniMap3DWidget(self.minimap_panel)
        self.minimap_widget.setStyleSheet("background: transparent;")
        layout.addWidget(self.minimap_widget)
        self.minimap_panel.raise_()
        self.minimap_panel.show()

    def _position_minimap(self) -> None:
        if not hasattr(self, "minimap_panel"):
            return
        viewport = self.view.viewport()
        if viewport is None:
            return
        margin = 12
        x = max(0, viewport.width() - self.minimap_panel.width() - margin)
        y = margin
        vp_pos = viewport.pos()
        self.minimap_panel.move(vp_pos.x() + x, vp_pos.y() + y)
        self.minimap_panel.raise_()

    def _refresh_minimap(self) -> None:
        if self.minimap_widget is not None:
            node_id = None
            point = None
            point = self._current_mesh_point()
            idx = self.current_frame_index
            if 0 <= idx < len(self._segment_frame_node_ids):
                node_id = self._segment_frame_node_ids[idx]
            self._update_minimap_3d(node_id, point=point)
            self._refresh_full_skeleton_window()
            return
        if not hasattr(self, "minimap_scene"):
            return
        self.minimap_scene.clear()
        self._minimap_current_items = []
        if not self.nodes or not self.graph:
            self.minimap_panel.setVisible(bool(self.nodes))
            self._refresh_full_skeleton_window()
            return
        xs = [node.x_px for node in self.nodes]
        ys = [node.y_px for node in self.nodes]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max(1.0, max_x - min_x)
        height = max(1.0, max_y - min_y)
        pad = max(4.0, 0.01 * max(width, height))
        scene_rect = QtCore.QRectF(min_x - pad, min_y - pad, width + 2 * pad, height + 2 * pad)
        self.minimap_scene.setSceneRect(scene_rect)
        self.minimap_view.setSceneRect(scene_rect)

        if self.graph.edges:
            edge_path = QtGui.QPainterPath()
            for i, j in self.graph.edges:
                a = self.nodes[i]
                b = self.nodes[j]
                edge_path.moveTo(a.x_px, a.y_px)
                edge_path.lineTo(b.x_px, b.y_px)
            edge_item = QtWidgets.QGraphicsPathItem(edge_path)
            edge_item.setPen(QtGui.QPen(QtGui.QColor(90, 220, 255, 220), 2.5))
            self.minimap_scene.addItem(edge_item)

        node_path = QtGui.QPainterPath()
        for node in self.nodes:
            node_path.addEllipse(node.x_px - 3.5, node.y_px - 3.5, 7.0, 7.0)
        node_item = QtWidgets.QGraphicsPathItem(node_path)
        node_item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        node_item.setBrush(QtGui.QColor(245, 245, 245, 230))
        self.minimap_scene.addItem(node_item)

        self.minimap_view.resetTransform()
        self.minimap_view.fitInView(scene_rect, QtCore.Qt.KeepAspectRatio)
        if self._minimap_zoom > 1.0:
            self.minimap_view.scale(self._minimap_zoom, self._minimap_zoom)
        self.minimap_panel.setVisible(True)
        self.minimap_scene.update(scene_rect)
        self.minimap_view.viewport().update()
        if self.frame_order:
            frame = self.frame_order[self.current_frame_index]
            self._focus_minimap_view(frame)
        self._refresh_full_skeleton_window()

    def _update_minimap_highlight(self, frame: Optional[int], node_id: Optional[int]) -> None:
        if self.minimap_widget is not None:
            point = self._current_mesh_point()
            self._update_minimap_3d(node_id, point=point)
            return
        if not hasattr(self, "minimap_scene"):
            return
        if self._minimap_current_items:
            for item in self._minimap_current_items:
                self.minimap_scene.removeItem(item)
            self._minimap_current_items = []
        if frame is None:
            return
        if node_id is not None and 0 <= node_id < len(self.nodes):
            node = self.nodes[node_id]
            radius = 8.0
            item = QtWidgets.QGraphicsEllipseItem(
                node.x_px - radius,
                node.y_px - radius,
                radius * 2,
                radius * 2,
            )
            item.setPen(QtGui.QPen(QtGui.QColor(255, 80, 80), 2))
            item.setBrush(QtGui.QBrush(QtGui.QColor(255, 80, 80, 140)))
            self.minimap_scene.addItem(item)
            self._minimap_current_items.append(item)
            return
        nodes = self.nodes_by_frame.get(frame, [])
        if not nodes:
            return
        radius = 7.0
        for node in nodes:
            item = QtWidgets.QGraphicsEllipseItem(
                node.x_px - radius,
                node.y_px - radius,
                radius * 2,
                radius * 2,
            )
            item.setPen(QtGui.QPen(QtGui.QColor(255, 80, 80), 2))
            item.setBrush(QtGui.QBrush(QtGui.QColor(255, 80, 80, 120)))
            self.minimap_scene.addItem(item)
            self._minimap_current_items.append(item)

    def _focus_minimap_view(self, frame: int, *, node_id: Optional[int] = None) -> None:
        if self.minimap_widget is not None:
            return
        if not hasattr(self, "minimap_view"):
            return
        if not self.nodes:
            return
        if node_id is not None and 0 <= node_id < len(self.nodes):
            focus = self._focus_from_node(self.nodes[node_id])
        else:
            focus = self._frame_focus(frame)
        if focus is None:
            return
        center_x, center_y, _, _ = focus
        window_nodes = self._minimap_local_nodes(center_x, center_y)
        if not window_nodes:
            return
        min_x = min(n.x_px for n in window_nodes)
        max_x = max(n.x_px for n in window_nodes)
        min_y = min(n.y_px for n in window_nodes)
        max_y = max(n.y_px for n in window_nodes)
        width = max(1.0, max_x - min_x)
        height = max(1.0, max_y - min_y)

        median_edge = self._minimap_edge_median_px
        margin = median_edge * 3.0 if median_edge > 0 else max(width, height) * 0.2
        margin = max(60.0, min(500.0, margin))
        rect = QtCore.QRectF(
            min_x - margin,
            min_y - margin,
            width + 2 * margin,
            height + 2 * margin,
        )
        self.minimap_view.resetTransform()
        self.minimap_view.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def _minimap_local_nodes(self, center_x: float, center_y: float) -> List[Node]:
        if not self.nodes:
            return []
        nodes = self.nodes
        radius = 0.0
        if self._minimap_edge_median_px > 0:
            radius = max(120.0, min(900.0, self._minimap_edge_median_px * 8.0))
        if radius > 0:
            radius_sq = radius * radius
            local = [n for n in nodes if (n.x_px - center_x) ** 2 + (n.y_px - center_y) ** 2 <= radius_sq]
            if len(local) >= 6:
                return local
        nodes_sorted = sorted(nodes, key=lambda n: (n.x_px - center_x) ** 2 + (n.y_px - center_y) ** 2)
        return nodes_sorted[: min(len(nodes_sorted), self._minimap_local_k)]

    def _update_minimap_3d(
        self,
        node_id: Optional[int],
        *,
        point: Optional[Tuple[float, float, float, float, float]] = None,
    ) -> None:
        if self.minimap_widget is None:
            return
        if not self.nodes or not self.graph:
            self.minimap_widget.clear()
            self.minimap_panel.setVisible(False)
            return
        center_x = None
        center_y = None
        center_z = None
        center_x_px = None
        center_y_px = None
        if point is not None:
            center_x_px, center_y_px, center_x, center_y, center_z = point
        elif node_id is not None and 0 <= node_id < len(self.nodes):
            node = self.nodes[node_id]
            center_x = node.x
            center_y = node.y
            center_z = node.z
            center_x_px = node.x_px
            center_y_px = node.y_px
        if center_x is None or center_x_px is None or center_y_px is None:
            self.minimap_widget.clear()
            self.minimap_panel.setVisible(bool(self.nodes))
            return
        local_nodes = self._minimap_local_nodes(center_x_px, center_y_px)
        if not local_nodes:
            self.minimap_widget.clear()
            return
        ghost_bounds = None
        node_ids = [n.id for n in local_nodes]
        id_set = set(node_ids)
        id_to_index = {nid: idx for idx, nid in enumerate(node_ids)}
        positions = np.array(
            [[n.x - center_x, n.y - center_y, n.z - center_z] for n in local_nodes],
            dtype=np.float32,
        )
        min_x = float(np.min(positions[:, 0]))
        max_x = float(np.max(positions[:, 0]))
        min_y = float(np.min(positions[:, 1]))
        max_y = float(np.max(positions[:, 1]))
        min_z = float(np.min(positions[:, 2]))
        max_z = float(np.max(positions[:, 2]))
        span_xy = max(max_x - min_x, max_y - min_y)
        span_z = max_z - min_z
        margin_xy = max(5.0, span_xy * 0.6)
        margin_z = max(5.0, span_z * 1.2)
        base_bounds = (
            min_x - margin_xy,
            max_x + margin_xy,
            min_y - margin_xy,
            max_y + margin_xy,
            min_z - margin_z,
            max_z + margin_z,
        )
        if self._mesh_preview_points is not None and self._mesh_preview_points.size:
            ghost_bounds = base_bounds
            expanded = [
                n
                for n in self.nodes
                if (
                    (n.x - center_x) >= ghost_bounds[0]
                    and (n.x - center_x) <= ghost_bounds[1]
                    and (n.y - center_y) >= ghost_bounds[2]
                    and (n.y - center_y) <= ghost_bounds[3]
                    and (n.z - center_z) >= ghost_bounds[4]
                    and (n.z - center_z) <= ghost_bounds[5]
                )
            ]
            if len(expanded) > len(local_nodes):
                local_nodes = expanded
                node_ids = [n.id for n in local_nodes]
                id_set = set(node_ids)
                id_to_index = {nid: idx for idx, nid in enumerate(node_ids)}
                positions = np.array(
                    [[n.x - center_x, n.y - center_y, n.z - center_z] for n in local_nodes],
                    dtype=np.float32,
                )
        edges: List[Tuple[int, int]] = []
        edge_globals: List[Tuple[int, int]] = []
        if self.graph and self.graph.edges:
            for i, j in self.graph.edges:
                if i in id_set and j in id_set:
                    edges.append((id_to_index[i], id_to_index[j]))
                    edge_globals.append((i, j))
        active_seg = self.active_segment_id
        branch_options = list(self._branch_options)
        active_color = QtGui.QColor(80, 220, 120, 230) if active_seg is not None else None
        branch_colors = {seg_id: self._segment_qcolor(seg_id) for seg_id in branch_options}
        node_colors: List[QtGui.QColor] = []
        for node in local_nodes:
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
        for (i, j) in edge_globals:
            color = QtGui.QColor(150, 180, 200, 170)
            edge = (i, j) if i < j else (j, i)
            if active_seg is not None:
                edges_set = self.mesh_segment_edges.get(active_seg)
                if edges_set and edge in edges_set and active_color is not None:
                    color = active_color
            if active_seg is None or color.alpha() < 230:
                for seg_id in branch_options:
                    edges_set = self.mesh_segment_edges.get(seg_id)
                    if edges_set and edge in edges_set:
                        color = branch_colors.get(seg_id, color)
                        break
            edge_colors.append(color)
        current_index = id_to_index.get(node_id) if node_id is not None else None
        if current_index is None and point is not None:
            best_idx = None
            best_dist = None
            for idx, n in enumerate(local_nodes):
                dx = n.x - center_x
                dy = n.y - center_y
                dz = n.z - center_z
                dist = dx * dx + dy * dy + dz * dz
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            current_index = best_idx
        legend_items: List[Tuple[str, QtGui.QColor]] = []
        if active_seg is not None and active_color is not None:
            legend_items.append(("Active", active_color))
        for idx, seg_id in enumerate(branch_options):
            legend_items.append((str(idx + 1), branch_colors.get(seg_id, QtGui.QColor(220, 220, 220, 210))))
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
        ghost_positions = None
        if ghost_bounds is not None and self._mesh_preview_points is not None and self._mesh_preview_points.size:
            ghost = self._mesh_preview_points
            gx = ghost[:, 0] - center_x
            gy = ghost[:, 1] - center_y
            gz = ghost[:, 2] - center_z
            mask = (
                (gx >= ghost_bounds[0])
                & (gx <= ghost_bounds[1])
                & (gy >= ghost_bounds[2])
                & (gy <= ghost_bounds[3])
                & (gz >= ghost_bounds[4])
                & (gz <= ghost_bounds[5])
            )
            if np.any(mask):
                ghost_local = np.stack([gx, gy, gz], axis=1)[mask]
                if ghost_local.shape[0] > MESH_PREVIEW_LOCAL_MAX_POINTS:
                    rng = np.random.default_rng()
                    idx = rng.choice(
                        ghost_local.shape[0],
                        size=MESH_PREVIEW_LOCAL_MAX_POINTS,
                        replace=False,
                    )
                    ghost_local = ghost_local[idx]
                ghost_positions = ghost_local.astype(np.float32, copy=False)
        flagged_positions = None
        if self.flagged_points:
            bounds = ghost_bounds or base_bounds
            fx_list: List[Tuple[float, float, float]] = []
            for record in self.flagged_points.values():
                if "x" not in record or "y" not in record or "z" not in record:
                    continue
                fx = float(record["x"]) - center_x
                fy = float(record["y"]) - center_y
                fz = float(record["z"]) - center_z
                if bounds is not None:
                    if not (
                        bounds[0] <= fx <= bounds[1]
                        and bounds[2] <= fy <= bounds[3]
                        and bounds[4] <= fz <= bounds[5]
                    ):
                        continue
                fx_list.append((fx, fy, fz))
            if fx_list:
                flagged_positions = np.asarray(fx_list, dtype=np.float32)
        bbox_edges = None
        bbox_corners = None
        hillock_positions = None
        distal_positions = None
        if self._hillock_original_node_id is not None:
            current_hillock = self._original_to_current_node_ids.get(int(self._hillock_original_node_id))
            if current_hillock is not None and current_hillock in id_to_index:
                hillock_positions = np.asarray(
                    [positions[id_to_index[current_hillock]]],
                    dtype=np.float32,
                )
        if self._distal_original_node_id is not None:
            current_distal = self._original_to_current_node_ids.get(int(self._distal_original_node_id))
            if current_distal is not None and current_distal in id_to_index:
                distal_positions = np.asarray(
                    [positions[id_to_index[current_distal]]],
                    dtype=np.float32,
                )
        if hillock_positions is not None:
            legend_items.append(("Hillock", QtGui.QColor(255, 160, 40, 245)))
        if distal_positions is not None:
            legend_items.append(("Distal", QtGui.QColor(80, 220, 255, 245)))
        if self._last_focus is not None:
            fx, fy, fw, fh = self._last_focus
            px = float(self.pixel_xy_spin.value()) if float(self.pixel_xy_spin.value()) > 0 else 1.0
            x_phys = float(fx) * px
            y_phys = float(fy) * px
            half_w = float(fw) * px / 2.0
            half_h = float(fh) * px / 2.0
            z_phys = center_z
            corners = [
                (x_phys - half_w, y_phys - half_h, z_phys),
                (x_phys + half_w, y_phys - half_h, z_phys),
                (x_phys + half_w, y_phys + half_h, z_phys),
                (x_phys - half_w, y_phys + half_h, z_phys),
            ]
            corners = [(x - center_x, y - center_y, z - center_z) for x, y, z in corners]
            bbox_corners = corners
            bbox_edges = [
                (corners[0], corners[1]),
                (corners[1], corners[2]),
                (corners[2], corners[3]),
                (corners[3], corners[0]),
            ]
        self.minimap_widget.set_data(
            positions,
            edges,
            current_index,
            node_ids=node_ids,
            node_colors=node_colors,
            edge_colors=edge_colors,
            legend_items=legend_items,
            ghost_positions=ghost_positions,
            bbox_edges=bbox_edges,
            bbox_corners=bbox_corners,
            flagged_positions=flagged_positions,
            hillock_positions=hillock_positions,
            distal_positions=distal_positions,
        )
        self.minimap_widget.set_arrows(prev_dir, next_dir)
        self.minimap_panel.setVisible(True)
