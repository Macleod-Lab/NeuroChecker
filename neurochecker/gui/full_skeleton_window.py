from typing import List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

from neurochecker.gui.widgets import MiniMap3DWidget


class FullSkeleton3DWindow(QtWidgets.QMainWindow):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NeuroChecker Full Skeleton 3D")
        self.resize(1200, 900)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.help_label = QtWidgets.QLabel(
            "Full skeleton picker. Left-drag rotates. Shift+left-drag or middle-drag pans. "
            "Mouse wheel zooms. Right-click a node to set hillock or distal neurite. "
            "Double-click resets the view."
        )
        self.help_label.setWordWrap(True)
        layout.addWidget(self.help_label)

        self.viewer = MiniMap3DWidget(self)
        self.viewer.setStyleSheet("background: rgb(14, 16, 20); border: 1px solid #333;")
        layout.addWidget(self.viewer, 1)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def clear(self) -> None:
        self.viewer.clear()
        self.status_label.setText("")

    def update_graph(
        self,
        *,
        positions: np.ndarray,
        edges: List[Tuple[int, int]],
        current_index: Optional[int],
        node_ids: List[int],
        node_colors,
        edge_colors,
        legend_items,
        ghost_positions: Optional[np.ndarray],
        flagged_positions: Optional[np.ndarray],
        hillock_positions: Optional[np.ndarray],
        distal_positions: Optional[np.ndarray],
        prev_dir,
        next_dir,
        title: str,
        status_text: str,
    ) -> None:
        self.setWindowTitle(title)
        self.viewer.set_data(
            positions,
            edges,
            current_index,
            node_ids=node_ids,
            node_colors=node_colors,
            edge_colors=edge_colors,
            legend_items=legend_items,
            ghost_positions=ghost_positions,
            flagged_positions=flagged_positions,
            hillock_positions=hillock_positions,
            distal_positions=distal_positions,
        )
        self.viewer.set_arrows(prev_dir, next_dir)
        self.viewer.set_status(status_text)
        self.status_label.setText(status_text)
