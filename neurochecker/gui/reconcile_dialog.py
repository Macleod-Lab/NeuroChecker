from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from neurochecker.graph import (
    GraphResult,
    Node,
    find_bridge_candidates,
    find_connected_components,
    label_graph,
)
from neurochecker.hillock import clone_graph
from neurochecker.gui.constants import logger


class _ComponentRow:
    __slots__ = (
        "component_id",
        "node_ids",
        "disc_node",
        "main_node",
        "distance",
        "frame_gap",
        "frame_range",
        "action",
    )

    def __init__(
        self,
        component_id: int,
        node_ids: List[int],
        disc_node: int,
        main_node: int,
        distance: float,
        frame_gap: int,
        frame_range: str,
    ) -> None:
        self.component_id = component_id
        self.node_ids = node_ids
        self.disc_node = disc_node
        self.main_node = main_node
        self.distance = distance
        self.frame_gap = frame_gap
        self.frame_range = frame_range
        self.action: Optional[str] = None  # "bridge", "delete", or None


class ReconcileDialog(QtWidgets.QDialog):
    """Identify disconnected skeleton components and bridge or remove them."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reconcile Disconnected Branches")
        self.resize(820, 500)
        self._parent = parent
        self._rows: List[_ComponentRow] = []
        self._build_ui()
        self._analyze()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.info_label = QtWidgets.QLabel("Analyzing...")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        threshold_row = QtWidgets.QHBoxLayout()
        threshold_row.addWidget(QtWidgets.QLabel("Auto-bridge max distance:"))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setRange(0.0, 100.0)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setToolTip(
            "Bridge all disconnected components whose nearest distance "
            "to the main tree is below this threshold (in physical units)."
        )
        threshold_row.addWidget(self.threshold_spin)
        auto_bridge_btn = QtWidgets.QPushButton("Auto-Bridge All Within Threshold")
        auto_bridge_btn.clicked.connect(self._auto_bridge)
        threshold_row.addWidget(auto_bridge_btn)
        threshold_row.addStretch()
        layout.addLayout(threshold_row)

        action_row = QtWidgets.QHBoxLayout()
        bridge_sel_btn = QtWidgets.QPushButton("Bridge Selected")
        bridge_sel_btn.clicked.connect(self._bridge_selected)
        delete_sel_btn = QtWidgets.QPushButton("Delete Selected")
        delete_sel_btn.clicked.connect(self._delete_selected)
        clear_sel_btn = QtWidgets.QPushButton("Clear Selected Actions")
        clear_sel_btn.clicked.connect(self._clear_selected)
        action_row.addWidget(bridge_sel_btn)
        action_row.addWidget(delete_sel_btn)
        action_row.addWidget(clear_sel_btn)
        action_row.addStretch()
        layout.addLayout(action_row)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        apply_btn = QtWidgets.QPushButton("Apply && Rebuild")
        apply_btn.clicked.connect(self._apply)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _analyze(self) -> None:
        p = self._parent
        nodes: List[Node] = getattr(p, "nodes", [])
        graph: Optional[GraphResult] = getattr(p, "graph", None)
        if not nodes or graph is None:
            self.info_label.setText("No skeleton loaded.")
            return

        components = find_connected_components(nodes, graph.edges)
        if len(components) <= 1:
            self.info_label.setText("Skeleton is fully connected. Nothing to reconcile.")
            return

        main_ids = components[0]
        main_set = set(main_ids)
        disc_components = components[1:]
        self.info_label.setText(
            f"Main tree: {len(main_ids)} nodes. "
            f"{len(disc_components)} disconnected component(s) found."
        )

        self._rows = []
        for idx, comp_ids in enumerate(disc_components):
            disc_node, main_node, distance, frame_gap = find_bridge_candidates(
                nodes, main_ids, comp_ids
            )
            frames = [nodes[nid].frame for nid in comp_ids]
            fr = f"{min(frames)}\u2013{max(frames)}" if frames else ""
            self._rows.append(
                _ComponentRow(
                    component_id=idx + 1,
                    node_ids=comp_ids,
                    disc_node=disc_node,
                    main_node=main_node,
                    distance=distance,
                    frame_gap=frame_gap,
                    frame_range=fr,
                )
            )

        self._refresh_table()

    def _refresh_table(self) -> None:
        cols = [
            "Comp", "Nodes", "Frame Range",
            "Nearest Dist", "Frame Gap",
            "Bridge: Disc Node", "Bridge: Main Node",
            "Action",
        ]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(self._rows))

        for r, row in enumerate(self._rows):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(row.component_id)))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(len(row.node_ids))))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(row.frame_range))
            dist_item = QtWidgets.QTableWidgetItem(f"{row.distance:.5f}")
            self.table.setItem(r, 3, dist_item)
            self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(str(row.frame_gap)))
            self.table.setItem(r, 5, QtWidgets.QTableWidgetItem(str(row.disc_node)))
            self.table.setItem(r, 6, QtWidgets.QTableWidgetItem(str(row.main_node)))
            action_item = QtWidgets.QTableWidgetItem(row.action or "")
            if row.action == "bridge":
                action_item.setForeground(QtGui.QBrush(QtGui.QColor(80, 200, 80)))
            elif row.action == "delete":
                action_item.setForeground(QtGui.QBrush(QtGui.QColor(220, 80, 80)))
            self.table.setItem(r, 7, action_item)

        self.table.resizeColumnsToContents()

    def _selected_row_indices(self) -> List[int]:
        return sorted(set(idx.row() for idx in self.table.selectedIndexes()))

    def _bridge_selected(self) -> None:
        for r in self._selected_row_indices():
            if 0 <= r < len(self._rows):
                self._rows[r].action = "bridge"
        self._refresh_table()

    def _delete_selected(self) -> None:
        for r in self._selected_row_indices():
            if 0 <= r < len(self._rows):
                self._rows[r].action = "delete"
        self._refresh_table()

    def _clear_selected(self) -> None:
        for r in self._selected_row_indices():
            if 0 <= r < len(self._rows):
                self._rows[r].action = None
        self._refresh_table()

    def _auto_bridge(self) -> None:
        threshold = self.threshold_spin.value()
        count = 0
        for row in self._rows:
            if row.distance <= threshold:
                row.action = "bridge"
                count += 1
        self._refresh_table()
        self.info_label.setText(
            self.info_label.text().split(".")[0]
            + f". Auto-bridge marked {count} component(s) within {threshold:.4f}."
        )

    def _apply(self) -> None:
        actions = [(r.action, r) for r in self._rows if r.action]
        if not actions:
            QtWidgets.QMessageBox.information(
                self, "Reconcile", "No actions selected. Mark components as bridge or delete first."
            )
            return

        p = self._parent
        nodes: List[Node] = list(getattr(p, "nodes", []))
        graph: Optional[GraphResult] = getattr(p, "graph", None)
        if not nodes or graph is None:
            return
        current_to_original = list(
            getattr(p, "_current_original_node_ids", list(range(len(nodes))))
        )
        base_graph: Optional[GraphResult] = getattr(p, "_base_graph", None)
        if base_graph is not None:
            working_graph = clone_graph(base_graph)
            nodes = list(working_graph.nodes)
            edges = list(working_graph.edges)
        else:
            working_graph = None
            edges = list(graph.edges)

        bridge_count = 0
        delete_count = 0
        delete_node_ids: Set[int] = set()

        for action, row in actions:
            if action == "bridge":
                disc_node = (
                    current_to_original[row.disc_node]
                    if 0 <= row.disc_node < len(current_to_original)
                    else row.disc_node
                )
                main_node = (
                    current_to_original[row.main_node]
                    if 0 <= row.main_node < len(current_to_original)
                    else row.main_node
                )
                edges.append((disc_node, main_node))
                bridge_count += 1
                logger.info(
                    "Reconcile bridge: node %d <-> node %d (dist=%.5f gap=%d)",
                    disc_node, main_node, row.distance, row.frame_gap,
                )
            elif action == "delete":
                for node_id in row.node_ids:
                    if 0 <= node_id < len(current_to_original):
                        delete_node_ids.add(int(current_to_original[node_id]))
                delete_count += 1
                logger.info(
                    "Reconcile delete: component %d (%d nodes)",
                    row.component_id, len(row.node_ids),
                )

        old_to_new: Dict[int, int] = {}
        if delete_node_ids:
            edges = [
                (i, j) for i, j in edges
                if i not in delete_node_ids and j not in delete_node_ids
            ]
            new_nodes: List[Node] = []
            for node in nodes:
                if node.id in delete_node_ids:
                    continue
                new_id = len(new_nodes)
                old_to_new[node.id] = new_id
                node.id = new_id
                new_nodes.append(node)
            edges = [
                (old_to_new[i], old_to_new[j])
                for i, j in edges
                if i in old_to_new and j in old_to_new
            ]
            nodes = new_nodes
            if getattr(p, "_hillock_original_node_id", None) is not None:
                p._hillock_original_node_id = old_to_new.get(p._hillock_original_node_id)
            if getattr(p, "_distal_original_node_id", None) is not None:
                p._distal_original_node_id = old_to_new.get(p._distal_original_node_id)
            if getattr(p, "_hillock_forward_original_node_id", None) is not None:
                p._hillock_forward_original_node_id = old_to_new.get(p._hillock_forward_original_node_id)

        new_graph = label_graph(nodes, edges)
        if hasattr(p, "_set_base_graph"):
            p._set_base_graph(new_graph)
        else:
            p._base_graph = new_graph
        if hasattr(p, "_apply_hillock_cutoff_to_loaded_graph"):
            p._apply_hillock_cutoff_to_loaded_graph()
        else:
            p.nodes = list(new_graph.nodes)
            p.graph = new_graph
            p.mesh_graph = new_graph
            p.nodes_by_frame = {}
            for node in p.nodes:
                p.nodes_by_frame.setdefault(node.frame, []).append(node)
            p._init_mesh_segments(new_graph)
        p._resolve_flagged_points()
        if hasattr(p, "_save_hillock_cutoff"):
            p._save_hillock_cutoff()
        p._minimap_edge_median_px = p._compute_edge_median_px()
        p._refresh_navigation(reset=True)
        p._refresh_minimap()
        p._update_neuron_info_label()
        p._show_current_frame()

        self._save_to_skeleton_cache(p, list(new_graph.nodes), list(new_graph.edges))

        msg = []
        if bridge_count:
            msg.append(f"Bridged {bridge_count} component(s)")
        if delete_count:
            msg.append(f"Deleted {delete_count} component(s) ({len(delete_node_ids)} nodes)")
        logger.info("Reconcile applied: %s", ", ".join(msg))
        p.statusBar().showMessage("Reconciled: " + ", ".join(msg), 5000)
        self.accept()

    def _save_to_skeleton_cache(
        self,
        parent: QtWidgets.QWidget,
        nodes: List[Node],
        edges: List[Tuple[int, int]],
    ) -> None:
        mesh_path = getattr(parent, "_mesh_path", None)
        if mesh_path is None:
            return
        from neurochecker.gui.constants import _MESH_SKELETON_CACHE
        from neurochecker.gui.mesh import _mesh_skeleton_cache_path

        cache_path = _mesh_skeleton_cache_path(mesh_path)
        sz = float(parent.slice_z_spin.value()) if hasattr(parent, "slice_z_spin") else 0.059
        px = float(parent.pixel_xy_spin.value()) if hasattr(parent, "pixel_xy_spin") else 0.00495

        points = np.asarray(
            [[n.x, n.y, n.z] for n in nodes], dtype=np.float32
        )
        edges_arr = (
            np.asarray(edges, dtype=np.int32)
            if edges
            else np.zeros((0, 2), dtype=np.int32)
        )
        try:
            old = np.load(cache_path, allow_pickle=False)
            pitch_xy = float(old["pitch_xy"]) if "pitch_xy" in old else px
        except Exception:
            pitch_xy = px

        try:
            np.savez_compressed(
                cache_path,
                points=points,
                edges=edges_arr,
                pitch_xy=np.float64(pitch_xy),
                slice_thickness_z=np.float64(sz),
            )
            logger.info("Skeleton cache updated after reconcile: %s", cache_path)
        except Exception:
            logger.warning("Failed to update skeleton cache: %s", cache_path)

        _MESH_SKELETON_CACHE.clear()
