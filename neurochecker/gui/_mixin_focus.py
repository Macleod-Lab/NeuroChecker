import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from neurochecker.graph import Node
from neurochecker.gui.constants import SEGMENT_RATIO_SCALE, logger
from neurochecker.gui.data import ComponentInfo


class FocusMixin:
    def _segment_tree_root_id(self) -> Optional[int]:
        if not self.mesh_segments:
            return None
        root = getattr(self, "_primary_neurite_segment_id", None)
        if isinstance(root, int) and 0 <= root < len(self.mesh_segments):
            return int(root)
        special_paths = getattr(self, "_segment_special_paths", {}) or {}
        for seg_id, raw_path in special_paths.items():
            try:
                candidate = int(seg_id)
            except (TypeError, ValueError):
                continue
            if candidate < 0 or candidate >= len(self.mesh_segments):
                continue
            path = raw_path if isinstance(raw_path, Path) else Path(str(raw_path))
            if path.as_posix() == "root":
                return candidate
        if self.active_segment_id is not None and 0 <= self.active_segment_id < len(self.mesh_segments):
            return int(self.active_segment_id)
        return max(range(len(self.mesh_segments)), key=lambda i: len(self.mesh_segments[i]))

    def _frame_focus(self, frame: int) -> Optional[Tuple[float, float, int, int]]:
        nodes = list(self.nodes_by_frame.get(frame, []))
        if not nodes:
            return None
        focus_run = None
        if hasattr(self, "focus_run_combo"):
            focus_run = self.focus_run_combo.currentData()
        if isinstance(focus_run, str) and focus_run:
            selected_runs = self._selected_runs()
            if selected_runs and focus_run not in selected_runs:
                focus_run = None
        if isinstance(focus_run, str) and focus_run:
            nodes = [n for n in nodes if n.run_id == focus_run]
        if not nodes:
            return None
        mode = "largest"
        if hasattr(self, "focus_mode_combo"):
            data = self.focus_mode_combo.currentData()
            if isinstance(data, str):
                mode = data
        if mode == "nearest" and self._last_focus is not None:
            last_x, last_y, _, _ = self._last_focus
            focus = min(nodes, key=lambda n: (n.x_px - last_x) ** 2 + (n.y_px - last_y) ** 2)
            return self._focus_tuple(focus)
        if mode == "centroid":
            x_px = float(sum(n.x_px for n in nodes) / len(nodes))
            y_px = float(sum(n.y_px for n in nodes) / len(nodes))
            focus = max(nodes, key=lambda n: n.mask_width * n.mask_height)
            base_w, base_h = self._focus_size(focus)
            return x_px, y_px, base_w, base_h
        focus = max(nodes, key=lambda n: n.mask_width * n.mask_height)
        return self._focus_tuple(focus)

    def _focus_from_node(self, node: Node) -> Tuple[float, float, int, int]:
        base_w, base_h = self._focus_size(node)
        return float(node.x_px), float(node.y_px), base_w, base_h

    def _current_mesh_point(self) -> Optional[Tuple[float, float, float, float, float]]:
        if not self._mesh_nav_enabled:
            return None
        idx = self.current_frame_index
        if idx < 0 or idx >= len(self._segment_frame_points_xyz):
            return None
        x, y, z = self._segment_frame_points_xyz[idx]
        if idx < len(self._segment_frame_points_px):
            x_px, y_px = self._segment_frame_points_px[idx]
        else:
            px = float(self.pixel_xy_spin.value()) if float(self.pixel_xy_spin.value()) > 0 else 1.0
            x_px, y_px = x / px, y / px
        return x_px, y_px, x, y, z

    def _segment_frame_centroid(self, frame: int, segment_id: int) -> Optional[Tuple[float, float]]:
        comps = [c for c in self._get_components_for_frame(frame) if c.segment_id == segment_id]
        if not comps:
            return None
        total = sum(c.area for c in comps)
        if total <= 0:
            return None
        sum_x = sum(c.centroid[0] * c.area for c in comps)
        sum_y = sum(c.centroid[1] * c.area for c in comps)
        return sum_x / total, sum_y / total

    def _closest_mask_centroid(
        self, frame: int, x_px: float, y_px: float
    ) -> Optional[Tuple[float, float]]:
        comps = self._get_components_for_frame(frame)
        if not comps:
            return None
        best = None
        best_dist = None
        for comp in comps:
            cx, cy = comp.centroid
            dx = cx - x_px
            dy = cy - y_px
            dist = dx * dx + dy * dy
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (cx, cy)
        return best

    def _segment_frame_point_map(
        self,
        segment_id: int,
        *,
        frames: Optional[Sequence[int]] = None,
        points_px: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Dict[int, Tuple[float, float]]:
        mapping: Dict[int, Tuple[float, float]] = {}
        if frames is not None and points_px is not None:
            count = min(len(frames), len(points_px))
            for idx in range(count):
                frame = int(frames[idx])
                x_px, y_px = points_px[idx]
                mapping[frame] = (float(x_px), float(y_px))
            return mapping
        if (
            segment_id == self.active_segment_id
            and self._segment_frame_order
            and self._segment_frame_points_px
        ):
            count = min(len(self._segment_frame_order), len(self._segment_frame_points_px))
            for idx in range(count):
                frame = int(self._segment_frame_order[idx])
                x_px, y_px = self._segment_frame_points_px[idx]
                mapping[frame] = (float(x_px), float(y_px))
        return mapping

    def _segment_components_for_frame(self, frame: int, segment_id: int) -> List[ComponentInfo]:
        return [c for c in self._get_components_for_frame(frame) if c.segment_id == segment_id]

    def _nearest_component_to_point(
        self,
        components: Sequence[ComponentInfo],
        x_px: float,
        y_px: float,
    ) -> Optional[ComponentInfo]:
        if not components:
            return None
        return min(
            components,
            key=lambda c: (
                (c.centroid[0] - x_px) ** 2 + (c.centroid[1] - y_px) ** 2,
                -float(c.area),
            ),
        )

    def _active_segment_centerline_px(self, frame: int) -> Optional[Tuple[float, float]]:
        if self.active_segment_id is None:
            return None
        frame_points = self._segment_frame_point_map(int(self.active_segment_id))
        return frame_points.get(int(frame))

    def _display_components_for_frame(self, frame: int) -> List[ComponentInfo]:
        if self.active_segment_id is None:
            return self._get_components_for_frame(frame)
        seg_id = int(self.active_segment_id)
        all_comps = self._get_components_for_frame(frame)
        seg_comps = [c for c in all_comps if c.segment_id == seg_id]
        if seg_comps:
            return seg_comps
        hint = self._active_segment_centerline_px(frame)
        if hint is not None:
            fallback = self._nearest_component_to_point(all_comps, hint[0], hint[1])
            if fallback is not None:
                return [fallback]
        return []

    def _component_dims_px(self, comp: ComponentInfo) -> Tuple[int, int]:
        slc_y, slc_x = comp.slices
        width = int(slc_x.stop - slc_x.start)
        height = int(slc_y.stop - slc_y.start)
        if width <= 0 or height <= 0:
            side = max(1, int(round(math.sqrt(max(1, comp.area)))))
            return side, side
        return width, height

    def _select_component_for_segment_frame(
        self,
        frame: int,
        segment_id: int,
        *,
        center_hint_px: Optional[Tuple[float, float]] = None,
    ) -> Optional[ComponentInfo]:
        comps = self._segment_components_for_frame(frame, segment_id)
        if comps:
            if center_hint_px is None:
                return max(comps, key=lambda c: c.area)
            hx, hy = center_hint_px
            return self._nearest_component_to_point(comps, hx, hy)
        all_comps = self._get_components_for_frame(frame)
        if not all_comps:
            return None
        if center_hint_px is not None:
            hx, hy = center_hint_px
            return self._nearest_component_to_point(all_comps, hx, hy)
        return max(all_comps, key=lambda c: c.area)

    def _focus_from_point(self, x_px: float, y_px: float, frame: int) -> Tuple[float, float, int, int]:
        base_w = 256
        base_h = 256
        if self.active_segment_id is not None:
            base_w, base_h = self._segment_focus_size(frame, int(self.active_segment_id))
        elif self._last_focus is not None:
            base_w = int(self._last_focus[2])
            base_h = int(self._last_focus[3])
        if base_w <= 0 or base_h <= 0:
            base_w = 256
            base_h = 256
        return float(x_px), float(y_px), base_w, base_h

    def _segment_bbox_samples(
        self,
        frame: int,
        segment_id: int,
        *,
        frame_points: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        dims_current: List[Tuple[int, int]] = []
        dims_neighbors: List[Tuple[int, int]] = []
        dims_window: List[Tuple[int, int]] = []
        if frame_points is None:
            frame_points = self._segment_frame_point_map(segment_id)
        for f in range(frame - 2, frame + 3):
            hint = frame_points.get(f) if frame_points else None
            comp = self._select_component_for_segment_frame(
                f,
                segment_id,
                center_hint_px=hint,
            )
            if comp is None:
                continue
            w, h = self._component_dims_px(comp)
            if w <= 0 or h <= 0:
                continue
            dims_window.append((w, h))
            if f == frame:
                dims_current.append((w, h))
            else:
                dims_neighbors.append((w, h))
        return dims_current, dims_neighbors, dims_window

    def _segment_focus_size(
        self,
        frame: int,
        segment_id: int,
        *,
        frame_points: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> Tuple[int, int]:
        dims_current, _dims_neighbors, dims_window = self._segment_bbox_samples(
            frame,
            segment_id,
            frame_points=frame_points,
        )
        if not dims_window:
            if self._last_focus is not None:
                return int(self._last_focus[2]), int(self._last_focus[3])
            return 256, 256
        ratios = [w / float(h) for w, h in dims_window if h > 0]
        if not ratios:
            return 256, 256
        ratio = sum(ratios) / float(len(ratios))
        areas = [w * h for w, h in dims_window]
        area = sum(areas) / float(len(areas)) if areas else 256 * 256
        width = max(1.0, math.sqrt(area * ratio)) * SEGMENT_RATIO_SCALE
        height = (max(1.0, width / ratio) if ratio > 0 else max(1.0, math.sqrt(area))) * SEGMENT_RATIO_SCALE
        return int(round(width)), int(round(height))

    def _segment_ratio_stats(
        self,
        frame: int,
        segment_id: int,
        *,
        frame_points: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]], Optional[float]]:
        dims_current, dims_neighbors, dims_window = self._segment_bbox_samples(
            frame,
            segment_id,
            frame_points=frame_points,
        )

        def _avg_dims(dims: List[Tuple[int, int]]) -> Optional[Tuple[float, float, float]]:
            if not dims:
                return None
            avg_w = sum(w for w, _ in dims) / float(len(dims))
            avg_h = sum(h for _, h in dims) / float(len(dims))
            if avg_h <= 0:
                return None
            return avg_w, avg_h, avg_w / avg_h

        cur = _avg_dims(dims_current)
        neigh = _avg_dims(dims_neighbors)
        final_ratio = None
        if dims_window:
            ratios = [w / float(h) for w, h in dims_window if h > 0]
            if ratios:
                final_ratio = sum(ratios) / float(len(ratios))
        return cur, neigh, final_ratio

    def _build_segment_tree_paths(self) -> Dict[int, Path]:
        if not self.mesh_segments:
            return {}
        root = self._segment_tree_root_id()
        if root is None:
            return {}
        neighbors: Dict[int, Set[int]] = {i: set() for i in range(len(self.mesh_segments))}
        for _node_id, segs in self.mesh_node_to_segments.items():
            if len(segs) < 2:
                continue
            segs_sorted = sorted(set(segs))
            for i in range(len(segs_sorted)):
                for j in range(i + 1, len(segs_sorted)):
                    a = segs_sorted[i]
                    b = segs_sorted[j]
                    neighbors[a].add(b)
                    neighbors[b].add(a)
        special_paths: Dict[int, Path] = {}
        raw_special_paths = getattr(self, "_segment_special_paths", {}) or {}
        for seg_id, raw_path in raw_special_paths.items():
            try:
                candidate = int(seg_id)
            except (TypeError, ValueError):
                continue
            if candidate < 0 or candidate >= len(self.mesh_segments):
                continue
            special_paths[candidate] = raw_path if isinstance(raw_path, Path) else Path(str(raw_path))
        special_paths.setdefault(int(root), Path("root"))

        visited: Set[int] = set()
        paths: Dict[int, Path] = {int(root): special_paths[int(root)]}

        def dfs(seg_id: int, base: Path) -> None:
            visited.add(seg_id)
            children = sorted(
                [
                    n
                    for n in neighbors.get(seg_id, set())
                    if n not in visited and n not in special_paths
                ]
            )
            for idx, child in enumerate(children, start=1):
                child_path = base / f"branch_{idx}"
                paths[child] = child_path
                dfs(child, child_path)

        dfs(int(root), special_paths[int(root)])
        for seg_id in sorted(
            [sid for sid in special_paths.keys() if sid != int(root)],
            key=lambda sid: (special_paths[sid].as_posix(), sid),
        ):
            base = special_paths[seg_id]
            paths[seg_id] = base
            if seg_id not in visited:
                dfs(seg_id, base)
        remaining = [s for s in range(len(self.mesh_segments)) if s not in visited]
        for idx, seg_id in enumerate(remaining, start=1):
            base = Path(f"disconnected_{idx}")
            paths[seg_id] = base
            dfs(seg_id, base)
        return paths

    def _current_frame_and_node(self) -> Tuple[Optional[int], Optional[int]]:
        if not self.frame_order:
            return None, None
        if self._mesh_nav_enabled:
            idx = self.current_frame_index
            if idx < 0 or idx >= len(self.frame_order):
                return None, None
            frame = self.frame_order[idx]
            node_id = None
            if idx < len(self._segment_frame_node_ids):
                node_id = self._segment_frame_node_ids[idx]
            return frame, node_id
        return self.frame_order[self.current_frame_index], None

    def _focus_tuple(self, focus: Node) -> Tuple[float, float, int, int]:
        base_w, base_h = self._focus_size(focus)
        return float(focus.x_px), float(focus.y_px), base_w, base_h

    def _focus_size(self, focus: Node) -> Tuple[int, int]:
        base_w = int(focus.mask_width)
        base_h = int(focus.mask_height)
        stats = self.run_stats.get(focus.run_id)
        if stats:
            base_w = max(base_w, int(stats.max_width))
            base_h = max(base_h, int(stats.max_height))
        return base_w, base_h
