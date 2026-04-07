import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from neurochecker.graph import GraphResult, Node
from neurochecker.hillock import build_soma_aware_segments, clone_graph
from neurochecker.gui.constants import MESH_PREVIEW_MAX_POINTS, logger
from neurochecker.gui.data import ImageSampler
from neurochecker.gui.helpers import _flag_point_record
from neurochecker.gui.mesh import _load_ascii_ply_mesh, _mesh_skeleton_graph
from neurochecker.mask_io import (
    MaskEntry,
    collect_run_stats,
    load_mask_entries,
    iter_mask_entries_by_frame,
)
from neurochecker.pipeline import normalize_neuron_id, load_mesh_skeleton


class DataMixin:
    def _browse_data_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select NeuroTracer Data Root")
        if not path:
            return
        self.data_root = Path(path)
        self.data_root_edit.setText(path)
        self._populate_neuron_list()

    def _browse_images_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Image Stack Directory")
        if not path:
            return
        self.images_dir = Path(path)
        self.images_dir_edit.setText(path)
        self._init_image_sampler()

    def _browse_skeleton_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Mesh Skeleton JSON", "", "JSON files (*.json)"
        )
        if not path:
            return
        self.skeleton_path = Path(path)
        self.skeleton_edit.setText(path)
        # Extract neuron ID from filename or directory
        skeleton_name = self.skeleton_path.stem
        if skeleton_name.startswith("neuron_"):
            self._current_neuron_name = skeleton_name
        else:
            self._current_neuron_name = f"neuron_{skeleton_name}"
        self._load_mesh_skeleton(self.skeleton_path)

    def _browse_mesh_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Mesh PLY File", "", "PLY files (*.ply)"
        )
        if not path:
            return
        self.mesh_path = Path(path)
        self.mesh_edit.setText(path)
        # Load mesh for visualization if skeleton is already loaded
        if hasattr(self, 'mesh_graph') and self.mesh_graph:
            self._load_mesh_for_visualization(self.mesh_path)

    def _init_image_sampler(self) -> None:
        if self.images_dir is None:
            return
        if self.image_sampler is not None:
            self.image_sampler.shutdown()
        self.image_sampler = ImageSampler(self.images_dir, self.data_root)
        try:
            self.image_sampler.open()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Images", f"Failed to open images: {exc}")
            self.image_sampler = None

    def _load_skeletons_from_dir(self, skeleton_dir: Path) -> None:
        """Load all skeleton data from subdirectories in the given directory."""
        self.skeleton_data.clear()
        self.use_skeletons = True
        for subdir in skeleton_dir.iterdir():
            if subdir.is_dir():
                skeleton_file = subdir / "skeleton.json"
                if skeleton_file.exists():
                    try:
                        nodes, run_stats = load_mesh_skeleton(skeleton_file)
                        skeleton_json = json.loads(skeleton_file.read_text())
                        edges = [(i, j) for i, j in skeleton_json["edges"]]
                        self.skeleton_data[subdir.name] = {
                            'nodes': nodes,
                            'run_stats': run_stats,
                            'edges': edges,
                            'paths': skeleton_json.get("paths", []),
                            'counts': skeleton_json.get("counts", {}),
                            'source_path': skeleton_file,
                        }
                    except Exception as exc:
                        print(f"Failed to load skeleton from {skeleton_file}: {exc}")
        self._populate_neuron_list()

    def _populate_neuron_list(self) -> None:
        self.neuron_list.clear()
        if self.data_root:
            neuron_dirs = sorted(p for p in self.data_root.iterdir() if p.is_dir() and p.name.startswith("neuron_"))
            for path in neuron_dirs:
                self.neuron_list.addItem(path.name)
        if self.use_skeletons:
            for neuron in self.skeleton_data:
                if self.neuron_list.findItems(neuron, QtCore.Qt.MatchExactly) == []:
                    self.neuron_list.addItem(neuron)

    def _on_neuron_selected(self) -> None:
        items = self.neuron_list.selectedItems()
        if not items:
            return
        self._current_neuron_name = items[0].text()
        if self.use_skeletons:
            self._load_skeleton_neuron(self._current_neuron_name)
        else:
            neuron_id = normalize_neuron_id(items[0].text())
            self._load_neuron(neuron_id)

    def _clear_loaded_neuron_state(self) -> None:
        self.entries = []
        self.entries_by_frame = {}
        self.run_stats = {}
        self._base_graph = None
        self.nodes = []
        self.nodes_by_frame = {}
        self.graph = None
        self.mesh_graph = None
        self._current_original_node_ids = []
        self._original_to_current_node_ids = {}
        self.mesh_segments = []
        self.mesh_segment_colors = []
        self.mesh_segment_edges = {}
        self.mesh_node_to_segments = {}
        self.mesh_kdtree = None
        self.mesh_kdtree_segment_ids = None
        self.active_segment_id = None
        self._segment_nodes = []
        self._segment_frame_order = []
        self._segment_frame_points_px = []
        self._segment_frame_points_xyz = []
        self._segment_frame_node_ids = []
        self._segment_anchor_node_id = None
        self._mesh_nav_enabled = False
        self._mesh_path = None
        self._mesh_preview_points = None
        self._mesh_preview_key = None
        self._current_focus_center_px = None
        self._current_focus_source = None
        self.frame_order = []
        self.current_frame_index = 0
        self._last_focus = None
        self._last_view_context = None
        self.flagged_masks.clear()
        self.flagged_points.clear()
        self._hillock_original_node_id = None
        self._distal_original_node_id = None
        self._hillock_forward_original_node_id = None
        self._soma_original_node_ids = []
        self._soma_segment_id = None
        self._primary_neurite_segment_id = None
        self._segment_special_paths = {}
        self._segment_entry_map = {}
        self.mask_cache.clear()
        self.component_cache.clear()
        self._flagged_points_path = self._flagged_points_file_path()
        self._hillock_cutoff_path = self._hillock_cutoff_file_path()
        self._update_hillock_status()

    def _load_neuron(self, neuron_id: str, *, raise_on_error: bool = False) -> None:
        if self.data_root is None:
            return
        self._current_neuron_name = neuron_id
        self._clear_loaded_neuron_state()
        try:
            self.entries = load_mask_entries(self.data_root, neuron_id)
        except Exception as exc:
            self._populate_run_list()
            self._refresh_navigation(reset=True)
            self._update_neuron_info_label()
            self._refresh_minimap()
            QtWidgets.QMessageBox.warning(self, "Masks", f"Failed to load masks: {exc}")
            if raise_on_error:
                raise
            return
        self.entries_by_frame = iter_mask_entries_by_frame(self.entries)
        self.run_stats = collect_run_stats(self.entries)
        self._populate_run_list()
        self._rebuild_graph()
        self._load_flagged_points()
        self._resolve_flagged_points()
        self._refresh_minimap()

    def _update_neuron_info_label(self) -> None:
        parts = []
        if self.entries:
            parts.append(f"{len(self.entries)} masks")
        if self.run_stats:
            parts.append(f"{len(self.run_stats)} runs")
        if self.entries_by_frame:
            frames = sorted(self.entries_by_frame.keys())
            parts.append(f"frames {frames[0]}\u2013{frames[-1]}")
        if self.mesh_segments:
            parts.append(f"{len(self.mesh_segments)} segments")
        if self.nodes:
            parts.append(f"{len(self.nodes)} nodes")
        if self._soma_original_node_ids:
            parts.append(f"soma branch {len(self._soma_original_node_ids)} nodes")
        self.neuron_info_label.setText(" | ".join(parts) if parts else "")

    def _populate_run_list(self) -> None:
        self._populating_runs = True
        self.run_list.clear()
        for run_id, stats in sorted(self.run_stats.items()):
            count = sum(1 for e in self.entries if e.run_id == run_id)
            item = QtWidgets.QListWidgetItem(f"{run_id}  ({count})")
            item.setData(QtCore.Qt.UserRole, run_id)
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            color = getattr(stats, "color", None)
            if color:
                item.setForeground(QtGui.QBrush(QtGui.QColor(*color)))
            self.run_list.addItem(item)
        self._populating_runs = False
        self._populate_focus_run_combo()

    def _populate_focus_run_combo(self) -> None:
        if not hasattr(self, "focus_run_combo"):
            return
        current_data = self.focus_run_combo.currentData()
        self.focus_run_combo.blockSignals(True)
        self.focus_run_combo.clear()
        self.focus_run_combo.addItem("Auto (all runs)", userData=None)
        for run_id in sorted(self.run_stats.keys()):
            self.focus_run_combo.addItem(run_id, userData=run_id)
        if current_data is not None:
            idx = self.focus_run_combo.findData(current_data)
            if idx != -1:
                self.focus_run_combo.setCurrentIndex(idx)
        self.focus_run_combo.blockSignals(False)

    def _hillock_cutoff_file_path(self) -> Optional[Path]:
        if self.data_root is not None and self._current_neuron_name:
            neuron_dir = self.data_root / self._current_neuron_name
            if neuron_dir.exists():
                return neuron_dir / "hillock_cutoff.json"
        skeleton_path = getattr(self, "skeleton_path", None)
        if isinstance(skeleton_path, Path) and skeleton_path.exists():
            return skeleton_path.with_name(f"{skeleton_path.stem}_hillock.json")
        mesh_path = getattr(self, "_mesh_path", None)
        if isinstance(mesh_path, Path):
            return mesh_path.with_suffix(".hillock.json")
        return None

    def _load_hillock_cutoff(self) -> None:
        self._hillock_cutoff_path = self._hillock_cutoff_file_path()
        self._hillock_original_node_id = None
        self._distal_original_node_id = None
        self._hillock_forward_original_node_id = None
        self._soma_original_node_ids = []
        self._soma_segment_id = None
        self._primary_neurite_segment_id = None
        self._segment_special_paths = {}
        path = self._hillock_cutoff_path
        if path is None or not path.exists():
            self._update_hillock_status()
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read hillock cutoff: %s", path)
            self._update_hillock_status()
            return
        try:
            hillock_id = payload.get("hillock_node_id")
            distal_id = payload.get("distal_node_id")
            if hillock_id is not None:
                self._hillock_original_node_id = int(hillock_id)
            if distal_id is not None:
                self._distal_original_node_id = int(distal_id)
        except Exception:
            self._hillock_original_node_id = None
            self._distal_original_node_id = None
        self._update_hillock_status()

    def _save_hillock_cutoff(self) -> None:
        self._hillock_cutoff_path = self._hillock_cutoff_file_path()
        path = self._hillock_cutoff_path
        if path is None:
            return
        if self._hillock_original_node_id is None and self._distal_original_node_id is None:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                logger.warning("Failed to remove hillock cutoff file: %s", path)
            return
        payload = {
            "version": 1,
            "neuron": self._current_neuron_name,
            "hillock_node_id": self._hillock_original_node_id,
            "distal_node_id": self._distal_original_node_id,
            "forward_neighbor_node_id": self._hillock_forward_original_node_id,
            "soma_original_node_ids": list(self._soma_original_node_ids),
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("Failed to write hillock cutoff: %s", path)

    def _set_base_graph(self, graph: GraphResult) -> None:
        self._base_graph = clone_graph(graph)

    def _set_current_graph_state(
        self,
        graph: GraphResult,
        *,
        current_to_original: Optional[List[int]] = None,
        anchor_original_node_id: Optional[int] = None,
        segment_nodes: Optional[List[List[int]]] = None,
        segment_edges: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
        preferred_active_segment_id: Optional[int] = None,
        soma_segment_id: Optional[int] = None,
        primary_neurite_segment_id: Optional[int] = None,
        segment_special_paths: Optional[Dict[int, Path]] = None,
    ) -> None:
        self.mesh_graph = graph
        self.graph = graph
        self.nodes = list(graph.nodes)
        if current_to_original is None:
            current_to_original = list(range(len(self.nodes)))
        self._current_original_node_ids = list(current_to_original)
        self._original_to_current_node_ids = {
            int(original_id): idx
            for idx, original_id in enumerate(self._current_original_node_ids)
        }
        self._mesh_nav_enabled = bool(self.nodes)
        self.nodes_by_frame = {}
        for node in self.nodes:
            self.nodes_by_frame.setdefault(node.frame, []).append(node)
        self.component_cache.clear()
        self.mask_cache.clear()
        self._soma_segment_id = soma_segment_id
        self._primary_neurite_segment_id = primary_neurite_segment_id
        self._segment_special_paths = dict(segment_special_paths or {})
        self._init_mesh_segments(
            graph,
            segment_nodes=segment_nodes,
            segment_edges=segment_edges,
            preferred_active_segment_id=preferred_active_segment_id,
        )
        if anchor_original_node_id is None:
            return
        current_anchor = self._original_to_current_node_ids.get(int(anchor_original_node_id))
        if current_anchor is None:
            return
        candidate_segments = list(self.mesh_node_to_segments.get(current_anchor, []))
        if not candidate_segments:
            return
        chosen_segment = int(candidate_segments[0])
        current_forward = None
        if self._hillock_forward_original_node_id is not None:
            current_forward = self._original_to_current_node_ids.get(
                int(self._hillock_forward_original_node_id)
            )
        if current_forward is not None:
            for seg_id in candidate_segments:
                if current_forward in self.mesh_segments[seg_id]:
                    chosen_segment = int(seg_id)
                    break
        self._set_active_segment(chosen_segment, anchor_node_id=current_anchor)

    def _apply_hillock_cutoff_to_loaded_graph(self) -> None:
        base_graph = self._base_graph
        if base_graph is None:
            return
        self._hillock_forward_original_node_id = None
        self._soma_original_node_ids = []
        self._soma_segment_id = None
        self._primary_neurite_segment_id = None
        self._segment_special_paths = {}
        anchor_original_node_id = None
        if (
            self._hillock_original_node_id is not None
            and self._distal_original_node_id is not None
        ):
            try:
                result = build_soma_aware_segments(
                    base_graph,
                    hillock_node_id=int(self._hillock_original_node_id),
                    distal_node_id=int(self._distal_original_node_id),
                )
            except ValueError as exc:
                logger.warning("Hillock cutoff invalid for %s: %s", self._current_neuron_name, exc)
                self.statusBar().showMessage(f"Hillock cutoff ignored: {exc}", 5000)
            else:
                self._hillock_forward_original_node_id = result.forward_neighbor_original_node_id
                self._soma_original_node_ids = list(result.soma_original_node_ids)
                anchor_original_node_id = result.hillock_original_node_id
                self._set_current_graph_state(
                    result.graph,
                    current_to_original=result.current_to_original,
                    anchor_original_node_id=anchor_original_node_id,
                    segment_nodes=result.segment_nodes,
                    segment_edges=result.segment_edges,
                    preferred_active_segment_id=result.primary_neurite_segment_index,
                    soma_segment_id=result.soma_segment_index,
                    primary_neurite_segment_id=result.primary_neurite_segment_index,
                    segment_special_paths={
                        int(result.soma_segment_index): Path("soma"),
                        int(result.primary_neurite_segment_index): Path("root"),
                    },
                )
                self._update_hillock_status()
                return
        self._set_current_graph_state(clone_graph(base_graph))
        self._update_hillock_status()

    def _current_original_node(self) -> Optional[int]:
        _frame, node_id = self._current_frame_and_node()
        if node_id is None:
            return None
        if 0 <= node_id < len(self._current_original_node_ids):
            return int(self._current_original_node_ids[node_id])
        return None

    def _set_hillock_original_node(self, original_node_id: int) -> None:
        self._hillock_original_node_id = int(original_node_id)
        self._save_hillock_cutoff()
        self._apply_hillock_cutoff_to_loaded_graph()
        self._save_hillock_cutoff()
        self._resolve_flagged_points()
        self._minimap_edge_median_px = self._compute_edge_median_px()
        self._refresh_navigation(reset=True)
        self._refresh_minimap()
        self._update_neuron_info_label()
        self._show_current_frame()

    def _set_distal_original_node(self, original_node_id: int) -> None:
        self._distal_original_node_id = int(original_node_id)
        self._save_hillock_cutoff()
        self._apply_hillock_cutoff_to_loaded_graph()
        self._save_hillock_cutoff()
        self._resolve_flagged_points()
        self._minimap_edge_median_px = self._compute_edge_median_px()
        self._refresh_navigation(reset=True)
        self._refresh_minimap()
        self._update_neuron_info_label()
        self._show_current_frame()

    def _set_hillock_from_current_node(self) -> None:
        original_node_id = self._current_original_node()
        if original_node_id is None:
            QtWidgets.QMessageBox.information(
                self,
                "Hillock",
                "No exact current node is selected. Use the 3D minimap or move to a non-interpolated node first.",
            )
            return
        self._set_hillock_original_node(original_node_id)

    def _set_distal_from_current_node(self) -> None:
        original_node_id = self._current_original_node()
        if original_node_id is None:
            QtWidgets.QMessageBox.information(
                self,
                "Hillock",
                "No exact current node is selected. Use the 3D minimap or move to a non-interpolated node first.",
            )
            return
        self._set_distal_original_node(original_node_id)

    def _clear_hillock_cutoff(self) -> None:
        self._hillock_original_node_id = None
        self._distal_original_node_id = None
        self._hillock_forward_original_node_id = None
        self._soma_original_node_ids = []
        self._soma_segment_id = None
        self._primary_neurite_segment_id = None
        self._segment_special_paths = {}
        self._save_hillock_cutoff()
        if self._base_graph is not None:
            self._apply_hillock_cutoff_to_loaded_graph()
            self._resolve_flagged_points()
            self._minimap_edge_median_px = self._compute_edge_median_px()
            self._refresh_navigation(reset=True)
            self._refresh_minimap()
            self._update_neuron_info_label()
            self._show_current_frame()
        else:
            self._update_hillock_status()

    def _update_hillock_status(self) -> None:
        label = getattr(self, "hillock_status_label", None)
        if label is None:
            return
        parts: List[str] = []
        if self._hillock_original_node_id is not None:
            parts.append(f"Hillock n{int(self._hillock_original_node_id)}")
        else:
            parts.append("Hillock not set")
        if self._distal_original_node_id is not None:
            parts.append(f"Distal n{int(self._distal_original_node_id)}")
        else:
            parts.append("Distal not set")
        if self._soma_original_node_ids:
            parts.append(f"Soma branch active, {len(self._soma_original_node_ids)} soma nodes")
        elif self._hillock_original_node_id is not None and self._distal_original_node_id is not None:
            parts.append("Soma branch pending or invalid")
        else:
            parts.append("Full skeleton")
        label.setText(" | ".join(parts))

    def _show_minimap_node_context_menu(self, current_node_id: int, global_pos: QtCore.QPoint) -> None:
        if current_node_id < 0 or current_node_id >= len(self.nodes):
            return
        if 0 <= current_node_id < len(self._current_original_node_ids):
            original_node_id = int(self._current_original_node_ids[current_node_id])
        else:
            original_node_id = int(current_node_id)
        node = self.nodes[current_node_id]
        menu = QtWidgets.QMenu(self)
        hillock_action = menu.addAction(
            f"Set node {original_node_id} (frame {int(node.frame)}) as axon hillock"
        )
        distal_action = menu.addAction(
            f"Set node {original_node_id} (frame {int(node.frame)}) as distal neurite node"
        )
        menu.addSeparator()
        clear_action = menu.addAction("Clear hillock cutoff")
        chosen = menu.exec_(global_pos)
        if chosen == hillock_action:
            self._set_hillock_original_node(original_node_id)
        elif chosen == distal_action:
            self._set_distal_original_node(original_node_id)
        elif chosen == clear_action:
            self._clear_hillock_cutoff()

    def _flagged_points_file_path(self) -> Optional[Path]:
        if self.data_root is None or not self._current_neuron_name:
            return None
        neuron_dir = self.data_root / self._current_neuron_name
        if neuron_dir.exists():
            return neuron_dir / "viewer_flags.json"
        return self.data_root / f"{self._current_neuron_name}_viewer_flags.json"

    def _load_flagged_points(self) -> None:
        self.flagged_points.clear()
        path = self._flagged_points_path or self._flagged_points_file_path()
        self._flagged_points_path = path
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read viewer flags: %s", path)
            return
        points = data.get("points", [])
        if not isinstance(points, list):
            return
        for item in points:
            try:
                seg_id = int(item.get("segment"))
                frame = int(item.get("frame"))
            except Exception:
                continue
            record = _flag_point_record(
                seg_id,
                frame,
                step=item.get("step"),
                x=item.get("x"),
                y=item.get("y"),
                z=item.get("z"),
            )
            self.flagged_points[(seg_id, frame)] = record

    def _save_flagged_points(self) -> None:
        path = self._flagged_points_path or self._flagged_points_file_path()
        self._flagged_points_path = path
        if path is None:
            return
        payload = {
            "version": 1,
            "neuron": self._current_neuron_name,
            "updated": datetime.now().isoformat(timespec="seconds"),
            "points": list(self.flagged_points.values()),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("Failed to write viewer flags: %s", path)

    def _resolve_flagged_points(self) -> None:
        if not self.mesh_segments or not self.flagged_points:
            return
        updated = False
        for key, record in list(self.flagged_points.items()):
            if "x" in record and "y" in record and "z" in record:
                continue
            seg_id = record.get("segment")
            frame = record.get("frame")
            if seg_id is None or frame is None:
                continue
            try:
                seg_id = int(seg_id)
                frame = int(frame)
            except Exception:
                continue
            if seg_id < 0 or seg_id >= len(self.mesh_segments):
                continue
            frames, points_xyz, _points_px, _node_ids = self._segment_frame_path(seg_id)
            if frame in frames:
                idx = frames.index(frame)
                x, y, z = points_xyz[idx]
                record["x"] = float(x)
                record["y"] = float(y)
                record["z"] = float(z)
                updated = True
        if updated:
            self._save_flagged_points()

    def _on_run_selection_changed(self, *_args: object) -> None:
        if self._populating_runs:
            return
        selected = self.run_list.selectedItems()
        self.merge_runs_btn.setEnabled(len(selected) >= 2)

    def _on_focus_run_changed(self) -> None:
        self._last_focus = None
        self._refresh_current_view()

    def _selected_runs(self) -> Optional[List[str]]:
        return None

    def _selected_neuron_name(self) -> Optional[str]:
        items = self.neuron_list.selectedItems()
        if not items:
            return None
        return items[0].text()

    def _mesh_name_tokens(self, value: str) -> List[str]:
        stem = Path(str(value)).stem.lower()
        stem = re.sub(r"[^a-z0-9]+", " ", stem)
        tokens = [tok for tok in stem.split() if tok]
        generic = {"mesh", "fixed", "largest", "surface", "mask", "ply", "neuron"}
        filtered = [tok for tok in tokens if tok not in generic]
        return filtered or tokens

    def _mesh_match_score(self, neuron_name: str, path: Path) -> int:
        base = normalize_neuron_id(neuron_name).lower()
        stem = path.stem.lower()
        if stem in {
            f"{base}_mesh",
            f"neuron_{base}_mesh",
            f"{base}_mask_surface",
            base,
        }:
            return 1000

        target_tokens = self._mesh_name_tokens(base)
        candidate_tokens = self._mesh_name_tokens(stem)
        if not target_tokens or not candidate_tokens:
            return -1
        if candidate_tokens == target_tokens:
            return 950
        if len(candidate_tokens) >= len(target_tokens) and candidate_tokens[: len(target_tokens)] == target_tokens:
            return 900
        if all(tok in candidate_tokens for tok in target_tokens):
            return 800 - max(0, len(candidate_tokens) - len(target_tokens))

        target_compact = "".join(target_tokens)
        candidate_compact = "".join(candidate_tokens)
        if target_compact and target_compact in candidate_compact:
            return 700 - max(0, len(candidate_compact) - len(target_compact))
        return -1

    def _find_best_mesh_match(self, root: Path, neuron_name: str, *, recursive: bool) -> Optional[Path]:
        if not root.exists() or not root.is_dir():
            return None
        try:
            candidates = root.rglob("*.ply") if recursive else root.glob("*.ply")
        except Exception:
            return None
        best_path: Optional[Path] = None
        best_key: Optional[Tuple[int, int, int, str]] = None
        for path in candidates:
            score = self._mesh_match_score(neuron_name, path)
            if score < 0:
                continue
            key = (score, -len(path.stem), -len(str(path)), str(path).lower())
            if best_key is None or key > best_key:
                best_key = key
                best_path = path
        return best_path

    def _find_neuron_mesh(self, neuron_name: str) -> Optional[Path]:
        if not neuron_name:
            return None
        base = neuron_name
        if base.lower().startswith("neuron_"):
            base = base[7:]

        override_roots: List[Path] = []
        if self.mesh_dir is not None:
            override_roots.append(self.mesh_dir)
        for candidate in (
            Path.cwd().parent / "fixed_meshes",
            Path.home() / "Desktop" / "fixed_meshes",
        ):
            if candidate not in override_roots:
                override_roots.append(candidate)

        logger.debug("Searching override meshes for neuron=%s roots=%s", neuron_name, override_roots)
        for root in override_roots:
            match = self._find_best_mesh_match(root, neuron_name, recursive=True)
            if match is not None:
                logger.info("Found replacement mesh for %s at %s", neuron_name, match)
                return match

        candidate_paths = [
            f"{base}_mesh.ply",
            f"{neuron_name}_mesh.ply",
            f"{base}_mask_surface.ply",
            f"{base}.ply",
        ]
        search_roots: List[Path] = []
        if self.data_root is not None:
            search_roots.append(self.data_root)
        desktop = Path.home() / "Desktop"
        if desktop.exists():
            search_roots.append(desktop)
        search_roots.append(Path.cwd())
        logger.debug("Searching default meshes for neuron=%s roots=%s candidates=%s", neuron_name, search_roots, candidate_paths)
        for root in search_roots:
            exact_paths = [root / name for name in candidate_paths]
            exact_paths.extend(
                [
                    root / neuron_name / "mesh.ply",
                    root / f"neuron_{base}" / "mesh.ply",
                    root / base / "mesh.ply",
                ]
            )
            for path in exact_paths:
                if path.exists():
                    logger.info("Found mesh for %s at %s", neuron_name, path)
                    return path
            for neuron_root in (root / neuron_name, root / f"neuron_{base}", root / base):
                match = self._find_best_mesh_match(neuron_root, neuron_name, recursive=True)
                if match is not None:
                    logger.info("Found mesh for %s in neuron folder at %s", neuron_name, match)
                    return match
            match = self._find_best_mesh_match(root, neuron_name, recursive=False)
            if match is not None:
                logger.info("Found partial-name mesh for %s at %s", neuron_name, match)
                return match
        logger.warning("No mesh found for neuron=%s", neuron_name)
        return None

    def _selected_entries_for_frame(self, frame: int) -> List[MaskEntry]:
        selected_runs = set(self._selected_runs() or [])
        entries = []
        for entry in self.entries_by_frame.get(frame, []):
            if selected_runs and entry.run_id not in selected_runs:
                continue
            entries.append(entry)
        return entries

    def _rebuild_graph(self) -> None:
        self._last_focus = None
        if not self.entries:
            return
        logger.info("Rebuild graph (mesh): start entries=%d", len(self.entries))
        start_t = time.perf_counter()
        pixel_size = self.pixel_xy_spin.value()
        slice_thickness = self.slice_z_spin.value()
        mesh_path = self._find_neuron_mesh(self._current_neuron_name) if self._current_neuron_name else None
        self._mesh_path = mesh_path
        if not (mesh_path and mesh_path.exists()):
            logger.warning("Rebuild graph (mesh): no mesh found for neuron=%s", self._current_neuron_name)
            QtWidgets.QMessageBox.warning(
                self,
                "Mesh Required",
                f"No mesh PLY file found for {self._current_neuron_name}.\n"
                "A mesh is required to load a neuron.",
            )
            self.entries = []
            self.entries_by_frame = {}
            return
        mesh_graph = _mesh_skeleton_graph(
            mesh_path,
            pixel_size_xy=pixel_size,
            slice_thickness_z=slice_thickness,
        )
        if mesh_graph is None:
            logger.warning("Rebuild graph (mesh): skeleton build failed for %s", mesh_path)
            QtWidgets.QMessageBox.warning(
                self,
                "Skeleton Failed",
                f"Failed to build skeleton from mesh:\n{mesh_path}\n"
                "Check that trimesh and scikit-image are installed.",
            )
            self.entries = []
            self.entries_by_frame = {}
            return
        self._set_base_graph(mesh_graph)
        self._load_hillock_cutoff()
        self._apply_hillock_cutoff_to_loaded_graph()
        self._save_hillock_cutoff()
        self._load_mesh_preview_points(mesh_path)
        self._resolve_flagged_points()
        self._minimap_edge_median_px = self._compute_edge_median_px()
        self._refresh_navigation(reset=True)
        self._refresh_minimap()
        self._update_neuron_info_label()
        self._show_current_frame()
        logger.info(
            "Rebuild graph (mesh): nodes=%d edges=%d segments=%d total=%.2fs",
            len(self.nodes),
            len(self.graph.edges) if self.graph else 0,
            len(self.mesh_segments),
            time.perf_counter() - start_t,
        )

    def _load_mesh_preview_points(self, mesh_path: Path) -> Optional[np.ndarray]:
        try:
            mesh_mtime = float(mesh_path.stat().st_mtime)
        except OSError:
            mesh_mtime = 0.0
        cache_key = (str(mesh_path), mesh_mtime)
        if self._mesh_preview_key == cache_key and self._mesh_preview_points is not None:
            return self._mesh_preview_points
        points: Optional[np.ndarray] = None
        try:
            import trimesh

            mesh = trimesh.load(mesh_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                if not mesh.geometry:
                    mesh = None
                else:
                    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            if mesh is not None and hasattr(mesh, "vertices"):
                verts = np.asarray(mesh.vertices)
                if verts.size:
                    points = verts
        except Exception:
            points = None
        if points is None:
            mesh_data = _load_ascii_ply_mesh(mesh_path, max_faces=0)
            if mesh_data is not None:
                points = mesh_data[0]
        if points is None or points.size == 0:
            self._mesh_preview_points = None
            self._mesh_preview_key = cache_key
            return None
        if points.shape[0] > MESH_PREVIEW_MAX_POINTS:
            try:
                rng = np.random.default_rng()
                idx = rng.choice(points.shape[0], size=MESH_PREVIEW_MAX_POINTS, replace=False)
                points = points[idx]
            except Exception:
                stride = max(1, points.shape[0] // MESH_PREVIEW_MAX_POINTS)
                points = points[::stride]
        points = points.astype(np.float32, copy=False)
        self._mesh_preview_points = points
        self._mesh_preview_key = cache_key
        return points

    def _merge_selected_runs(self) -> None:
        import uuid
        from datetime import datetime as _dt

        selected_items = self.run_list.selectedItems()
        selected_run_ids = [item.data(QtCore.Qt.UserRole) for item in selected_items]
        selected_run_ids = [r for r in selected_run_ids if r]
        if len(selected_run_ids) < 2:
            QtWidgets.QMessageBox.information(self, "Merge", "Select at least 2 runs to merge.")
            return
        if self.data_root is None or not self._current_neuron_name:
            return
        neuron_id = normalize_neuron_id(self._current_neuron_name)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Merge Runs",
            f"Merge {len(selected_run_ids)} runs into a single run?\n\n"
            "The original runs will be deleted.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self.statusBar().showMessage("Merging runs...")
        QtWidgets.QApplication.processEvents()

        store_dir = self.data_root / f"neuron_{neuron_id}" / "masks"
        index_path = store_dir / "index.json"
        try:
            with index_path.open("r", encoding="utf-8") as fh:
                index_data = json.load(fh)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Merge", f"Failed to load index: {exc}")
            return

        selected_set = set(selected_run_ids)
        entries_by_frame: Dict[int, List[dict]] = {}
        for entry in index_data:
            if entry.get("run_id") in selected_set:
                entries_by_frame.setdefault(int(entry.get("frame", 0)), []).append(entry)

        first_color = None
        for entry in index_data:
            if entry.get("run_id") in selected_set:
                c = entry.get("color")
                if isinstance(c, (list, tuple)) and len(c) >= 3:
                    first_color = list(c[:3])
                    break
        if first_color is None:
            first_color = [255, 128, 0]

        new_run_id = f"merged_{uuid.uuid4().hex[:8]}"
        new_run_started = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entries: List[dict] = []
        frame_count = 0

        for frame, frame_entries in sorted(entries_by_frame.items()):
            if not frame_entries:
                continue
            masks_and_offsets = []
            fw = int(frame_entries[0].get("full_width") or 0)
            fh_val = int(frame_entries[0].get("full_height") or 0)
            for fe in frame_entries:
                mask_path = store_dir / str(fe.get("path", ""))
                if not mask_path.exists():
                    continue
                try:
                    with np.load(mask_path) as data:
                        mask = np.asarray(data["mask"]).astype(np.uint8)
                except Exception:
                    continue
                if mask.size == 0:
                    continue
                masks_and_offsets.append((mask, int(fe.get("x", 0)), int(fe.get("y", 0))))

            if not masks_and_offsets:
                continue

            min_x = min(ox for _, ox, _ in masks_and_offsets)
            min_y = min(oy for _, _, oy in masks_and_offsets)
            max_x = max(ox + m.shape[1] for m, ox, _ in masks_and_offsets)
            max_y = max(oy + m.shape[0] for m, _, oy in masks_and_offsets)
            merged_w = max_x - min_x
            merged_h = max_y - min_y
            merged = np.zeros((merged_h, merged_w), dtype=np.uint8)
            for mask, ox, oy in masks_and_offsets:
                lx = ox - min_x
                ly = oy - min_y
                h, w = mask.shape
                merged[ly : ly + h, lx : lx + w] |= mask

            if merged.sum() == 0:
                continue
            ys, xs = np.where(merged > 0)
            ty0, ty1 = int(ys.min()), int(ys.max()) + 1
            tx0, tx1 = int(xs.min()), int(xs.max()) + 1
            trimmed = merged[ty0:ty1, tx0:tx1]
            final_x = min_x + tx0
            final_y = min_y + ty0

            filename = f"{new_run_id}_{frame:05d}.npz"
            np.savez(store_dir / filename, mask=trimmed)
            new_entries.append(
                {
                    "path": filename,
                    "id": f"{new_run_id}_{frame:05d}",
                    "frame": frame,
                    "x": final_x,
                    "y": final_y,
                    "width": int(trimmed.shape[1]),
                    "height": int(trimmed.shape[0]),
                    "full_width": fw,
                    "full_height": fh_val,
                    "run_id": new_run_id,
                    "run_started": new_run_started,
                    "color": first_color,
                    "direction": "merged",
                }
            )
            frame_count += 1

        old_files = set()
        kept = []
        for entry in index_data:
            if entry.get("run_id") in selected_set:
                old_files.add(store_dir / str(entry.get("path", "")))
            else:
                kept.append(entry)
        for f in old_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        kept.extend(new_entries)
        with index_path.open("w", encoding="utf-8") as fh:
            json.dump(kept, fh, indent=2)

        self._load_neuron(neuron_id)
        self.statusBar().showMessage(
            f"Merged {len(selected_run_ids)} runs \u2192 {new_run_id} ({frame_count} frames). Skeleton rebuilt.", 5000
        )

    def _open_neuron_properties(self) -> None:
        from neurochecker.gui.neuron_properties import NeuronPropertiesDialog

        if not self._current_neuron_name:
            QtWidgets.QMessageBox.information(self, "Properties", "No neuron loaded.")
            return
        dlg = NeuronPropertiesDialog(self)
        dlg.exec_()

    def _open_reconcile_dialog(self) -> None:
        from neurochecker.gui.reconcile_dialog import ReconcileDialog

        if not self.nodes or self.graph is None:
            QtWidgets.QMessageBox.information(self, "Reconcile", "No skeleton loaded.")
            return
        dlg = ReconcileDialog(self)
        dlg.exec_()

    def _focus_goto_frame(self) -> None:
        self.goto_frame_spin.setFocus()
        self.goto_frame_spin.selectAll()

    def _load_mesh_skeleton(self, skeleton_path: Path) -> None:
        """Load skeleton data from mesh skeleton JSON file."""
        try:
            nodes, run_stats = load_mesh_skeleton(skeleton_path)
            # Load skeleton data
            skeleton_data = json.loads(skeleton_path.read_text())
            edges = [(i, j) for i, j in skeleton_data["edges"]]
            paths = skeleton_data.get("paths", [])
            counts = skeleton_data.get("counts", {})
            self._set_skeleton_data(nodes, run_stats, edges, paths, counts)

            mesh_path = self._find_neuron_mesh(self._current_neuron_name or skeleton_path.stem)
            if mesh_path is None:
                mesh_path = skeleton_path.parent / f"{skeleton_path.stem.replace('_skeleton', '')}.ply"
            if mesh_path.exists():
                self._load_mesh_for_visualization(mesh_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Skeleton", f"Failed to load mesh skeleton: {exc}")

    def _load_skeleton_neuron(self, neuron_name: str) -> None:
        """Load skeleton data for the selected neuron from skeleton_data."""
        if neuron_name not in self.skeleton_data:
            return
        data = self.skeleton_data[neuron_name]
        self.skeleton_path = data.get("source_path")
        self._set_skeleton_data(
            data['nodes'], data['run_stats'], data['edges'], data['paths'], data['counts']
        )
        mesh_path = self._find_neuron_mesh(neuron_name)
        if mesh_path is not None and mesh_path.exists():
            self._load_mesh_for_visualization(mesh_path)

    def _set_skeleton_data(self, nodes, run_stats, edges, paths, counts) -> None:
        self.entries = []  # No mask entries for mesh skeletons
        self.entries_by_frame = {}
        self.run_stats = run_stats
        self._last_focus = None
        self.flagged_masks.clear()
        self._populate_run_list()
        self.mask_cache.clear()
        self.component_cache.clear()
        self._mesh_nav_enabled = True

        graph = GraphResult(
            nodes=nodes,
            edges=edges,
            order=list(range(len(nodes))),
            paths=paths,
            counts=counts
        )
        self._set_base_graph(graph)
        self._load_hillock_cutoff()
        self._apply_hillock_cutoff_to_loaded_graph()
        self.flagged_points.clear()
        self._flagged_points_path = self._flagged_points_file_path()
        self._save_hillock_cutoff()
        self._load_flagged_points()
        self._resolve_flagged_points()
        self._refresh_minimap()

    def _load_mesh_for_visualization(self, mesh_path: Path) -> None:
        """Load mesh for 3D visualization."""
        try:
            verts, faces = _load_ascii_ply_mesh(mesh_path)
            if verts is not None and faces is not None:
                # Store mesh data for visualization
                self._mesh_vertices = verts
                self._mesh_faces = faces
                self._mesh_path = mesh_path
                logger.info("Loaded mesh for visualization: %s (%d verts, %d faces)",
                          mesh_path, len(verts), len(faces))
            else:
                logger.warning("Failed to load mesh: %s", mesh_path)
        except Exception as exc:
            logger.exception("Failed to load mesh: %s", mesh_path)
