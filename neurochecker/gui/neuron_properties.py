from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets


class NeuronPropertiesDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Neuron Properties")
        self.resize(780, 580)
        self._parent = parent
        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.summary_text = QtWidgets.QTextEdit()
        self.summary_text.setReadOnly(True)
        self.tabs.addTab(self.summary_text, "Summary")

        self.runs_table = QtWidgets.QTableWidget()
        self.runs_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.runs_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.runs_table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.runs_table, "Runs")

        self.segments_table = QtWidgets.QTableWidget()
        self.segments_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.segments_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.segments_table.horizontalHeader().setStretchLastSection(True)
        self.segments_table.cellDoubleClicked.connect(self._on_segment_double_click)
        self.tabs.addTab(self.segments_table, "Segments")

        btn_layout = QtWidgets.QHBoxLayout()
        reconcile_btn = QtWidgets.QPushButton("Reconcile Disconnected...")
        reconcile_btn.setToolTip("Bridge or remove disconnected skeleton components")
        reconcile_btn.clicked.connect(self._open_reconcile)
        btn_layout.addWidget(reconcile_btn)
        btn_layout.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _populate(self) -> None:
        p = self._parent
        neuron_name = getattr(p, "_current_neuron_name", None) or "?"
        entries = getattr(p, "entries", [])
        entries_by_frame = getattr(p, "entries_by_frame", {})
        run_stats = getattr(p, "run_stats", {})
        nodes = getattr(p, "nodes", [])
        graph = getattr(p, "graph", None)
        mesh_segments = getattr(p, "mesh_segments", [])

        seg_tree: Dict[int, Path] = {}
        builder = getattr(p, "_build_segment_tree_paths", None)
        if builder:
            seg_tree = builder()
        disconnected_ids: Set[int] = set()
        for seg_id, tree_path in seg_tree.items():
            if str(tree_path).startswith("disconnected"):
                disconnected_ids.add(seg_id)

        all_frames = sorted(entries_by_frame.keys()) if entries_by_frame else []
        frame_range = f"{all_frames[0]}\u2013{all_frames[-1]}" if all_frames else "N/A"

        edge_count = len(graph.edges) if graph else 0
        branch_count = sum(1 for n in nodes if n.label == "branch")
        endpoint_count = sum(1 for n in nodes if n.label == "endpoint")
        isolated_count = sum(1 for n in nodes if n.label == "isolated")
        connected_count = len(mesh_segments) - len(disconnected_ids)

        summary_lines = [
            f"<h3>{neuron_name}</h3>",
            f"<b>Total masks:</b> {len(entries)}",
            f"<b>Runs:</b> {len(run_stats)}",
            f"<b>Frame range:</b> {frame_range} ({len(all_frames)} unique frames)",
            "",
            f"<b>Skeleton nodes:</b> {len(nodes)}",
            f"<b>Skeleton edges:</b> {edge_count}",
            f"<b>Branch points:</b> {branch_count}",
            f"<b>Endpoints:</b> {endpoint_count}",
            f"<b>Isolated nodes:</b> {isolated_count}",
            "",
            f"<b>Segments (paths):</b> {len(mesh_segments)}",
            f"<b>Connected segments:</b> {connected_count}",
            f"<b>Disconnected segments:</b> {len(disconnected_ids)}",
        ]
        if disconnected_ids:
            summary_lines.append(
                f"<br><span style='color:#e8a040'><b>Disconnected IDs:</b> "
                f"{', '.join(str(s) for s in sorted(disconnected_ids))}</span>"
            )
        self.summary_text.setHtml("<br>".join(summary_lines))

        self._populate_runs_table(entries, run_stats, entries_by_frame)
        self._populate_segments_table(nodes, mesh_segments, seg_tree, disconnected_ids)

    def _populate_runs_table(
        self,
        entries: list,
        run_stats: dict,
        entries_by_frame: dict,
    ) -> None:
        cols = ["Run ID", "Masks", "Frame Range", "Max W", "Max H", "Color"]
        self.runs_table.setColumnCount(len(cols))
        self.runs_table.setHorizontalHeaderLabels(cols)

        run_entries: Dict[str, List] = {}
        for e in entries:
            run_entries.setdefault(e.run_id, []).append(e)

        rows = sorted(run_entries.keys())
        self.runs_table.setRowCount(len(rows))
        for r, run_id in enumerate(rows):
            ents = run_entries[run_id]
            frames = sorted(set(e.frame for e in ents))
            stats = run_stats.get(run_id)
            self.runs_table.setItem(r, 0, QtWidgets.QTableWidgetItem(run_id))
            self.runs_table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(len(ents))))
            fr = f"{frames[0]}\u2013{frames[-1]}" if frames else ""
            self.runs_table.setItem(r, 2, QtWidgets.QTableWidgetItem(fr))
            self.runs_table.setItem(
                r, 3, QtWidgets.QTableWidgetItem(str(stats.max_width) if stats else "")
            )
            self.runs_table.setItem(
                r, 4, QtWidgets.QTableWidgetItem(str(stats.max_height) if stats else "")
            )
            color_item = QtWidgets.QTableWidgetItem("")
            color = getattr(stats, "color", None) if stats else None
            if color:
                color_item.setBackground(QtGui.QBrush(QtGui.QColor(*color)))
            self.runs_table.setItem(r, 5, color_item)

        self.runs_table.resizeColumnsToContents()

    def _populate_segments_table(
        self,
        nodes: list,
        mesh_segments: list,
        seg_tree: Dict[int, Path],
        disconnected_ids: Set[int],
    ) -> None:
        cols = [
            "Seg ID", "Tree Path", "Status", "Nodes",
            "Frame Range", "Start Node", "End Node",
            "Label Start", "Label End",
        ]
        self.segments_table.setColumnCount(len(cols))
        self.segments_table.setHorizontalHeaderLabels(cols)
        self.segments_table.setRowCount(len(mesh_segments))
        self.segments_table.setSortingEnabled(False)

        warn_brush = QtGui.QBrush(QtGui.QColor(180, 100, 30))

        for r, path in enumerate(mesh_segments):
            if not path:
                continue
            seg_nodes = [nodes[nid] for nid in path if 0 <= nid < len(nodes)]
            frames = [n.frame for n in seg_nodes]
            fr = f"{min(frames)}\u2013{max(frames)}" if frames else ""
            start_node = path[0] if path else ""
            end_node = path[-1] if path else ""
            label_start = seg_nodes[0].label if seg_nodes else ""
            label_end = seg_nodes[-1].label if seg_nodes else ""
            tree_path = str(seg_tree.get(r, "?"))
            is_disconnected = r in disconnected_ids

            self.segments_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(r)))
            self.segments_table.setItem(r, 1, QtWidgets.QTableWidgetItem(tree_path))
            status_item = QtWidgets.QTableWidgetItem(
                "DISCONNECTED" if is_disconnected else "connected"
            )
            if is_disconnected:
                status_item.setForeground(warn_brush)
            self.segments_table.setItem(r, 2, status_item)
            self.segments_table.setItem(r, 3, QtWidgets.QTableWidgetItem(str(len(path))))
            self.segments_table.setItem(r, 4, QtWidgets.QTableWidgetItem(fr))
            self.segments_table.setItem(r, 5, QtWidgets.QTableWidgetItem(str(start_node)))
            self.segments_table.setItem(r, 6, QtWidgets.QTableWidgetItem(str(end_node)))
            self.segments_table.setItem(r, 7, QtWidgets.QTableWidgetItem(label_start))
            self.segments_table.setItem(r, 8, QtWidgets.QTableWidgetItem(label_end))

        self.segments_table.setSortingEnabled(True)
        self.segments_table.resizeColumnsToContents()

    def _open_reconcile(self) -> None:
        from neurochecker.gui.reconcile_dialog import ReconcileDialog

        dlg = ReconcileDialog(self._parent)
        result = dlg.exec_()
        if result == QtWidgets.QDialog.Accepted:
            self._populate()

    def _on_segment_double_click(self, row: int, _col: int) -> None:
        item = self.segments_table.item(row, 0)
        if item is None:
            return
        try:
            seg_id = int(item.text())
        except ValueError:
            return
        p = self._parent
        if hasattr(p, "_set_active_segment") and hasattr(p, "mesh_segments"):
            if 0 <= seg_id < len(p.mesh_segments):
                p._set_active_segment(seg_id)
                p._refresh_navigation(reset=True)
                p._show_current_frame()
                self.accept()
