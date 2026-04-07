from pathlib import Path
from typing import List

from PyQt5 import QtCore, QtWidgets


class MassExportDialog(QtWidgets.QDialog):
    def __init__(self, neuron_dirs: List[Path], parent: QtWidgets.QWidget = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mass Export — Select Neurons")
        self.resize(400, 450)
        self._neuron_dirs = neuron_dirs

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"{len(neuron_dirs)} neurons found. Select which to export:"))

        btn_row = QtWidgets.QHBoxLayout()
        select_all = QtWidgets.QPushButton("Select All")
        select_all.clicked.connect(self._select_all)
        deselect_all = QtWidgets.QPushButton("Deselect All")
        deselect_all.clicked.connect(self._deselect_all)
        btn_row.addWidget(select_all)
        btn_row.addWidget(deselect_all)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for d in neuron_dirs:
            item = QtWidgets.QListWidgetItem(d.name)
            item.setSelected(True)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_all(self) -> None:
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setSelected(True)

    def _deselect_all(self) -> None:
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setSelected(False)

    def selected_neurons(self) -> List[str]:
        return [item.text() for item in self.list_widget.selectedItems()]
