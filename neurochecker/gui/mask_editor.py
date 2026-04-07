"""Mask Editor window for multi-reviewer consensus workflows."""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, label as nd_label

from neurochecker.gui.consensus import (
    FrameKey,
    assignments_for_user,
    compute_consensus,
    frame_key_str,
    generate_assignments,
    load_delta_mask,
    load_verdicts,
    read_assignments,
    read_volunteers,
    save_delta_mask,
    save_verdicts,
    write_assignments,
)
from neurochecker.gui.helpers import _mask_outline, numpy_to_qimage

logger = logging.getLogger("neurochecker")

VERDICT_COLORS = {
    "good": "#2e7d32",
    "bad": "#c62828",
    "pending": "#757575",
}


class ReviewGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, scene: QtWidgets.QGraphicsScene, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(scene, parent)
        self._min_scale = 0.05
        self._max_scale = 40.0
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.SmartViewportUpdate)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def fit_scene_down_only(self) -> None:
        rect = self.sceneRect()
        viewport = self.viewport().rect()
        self.resetTransform()
        if rect.isNull() or viewport.isNull():
            return
        scale = min(
            viewport.width() / max(1.0, rect.width()),
            viewport.height() / max(1.0, rect.height()),
        )
        if scale < 1.0:
            self.scale(scale, scale)
        self.centerOn(rect.center())

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        current_scale = self.transform().m11()
        if current_scale <= 0:
            current_scale = 1.0
        factor = 1.0 + (delta / 1200.0)
        factor = max(0.2, min(5.0, factor))
        target_scale = current_scale * factor
        if target_scale < self._min_scale:
            factor = self._min_scale / current_scale
        elif target_scale > self._max_scale:
            factor = self._max_scale / current_scale
        self.scale(factor, factor)
        event.accept()


class MaskEditorWindow(QtWidgets.QMainWindow):

    def __init__(self, export_root: Optional[Path] = None, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NeuroChecker \u2014 Mask Editor")
        self.resize(1400, 900)

        self._export_root: Optional[Path] = None
        self._volunteers: List[str] = []
        self._assignments: List[Tuple[str, int, int, str]] = []
        self._user_frames: List[FrameKey] = []
        self._idx: int = 0
        self._verdicts: Dict[str, str] = {}
        self._sample_lookup: Dict[FrameKey, Dict] = {}

        self._edit_mode = False
        self._brush_add = True
        self._brush_radius = 8
        self._mask: Optional[np.ndarray] = None
        self._dirty = False
        self._painting = False
        self._last_pt: Optional[Tuple[int, int]] = None
        self._brush_cache: Dict[int, np.ndarray] = {}
        self._alpha = 80
        self._brush_cursor_item: Optional[QtWidgets.QGraphicsEllipseItem] = None
        self._rendered_fk: Optional[FrameKey] = None

        self._build_ui()
        self._build_menu()
        if export_root is not None:
            self._load_export(export_root)

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        left = QtWidgets.QWidget()
        left.setFixedWidth(280)
        col = QtWidgets.QVBoxLayout(left)
        col.setContentsMargins(0, 0, 0, 0)

        # folder
        g = QtWidgets.QGroupBox("Export Folder")
        r = QtWidgets.QHBoxLayout(g)
        self._folder_edit = QtWidgets.QLineEdit()
        self._folder_edit.setReadOnly(True)
        btn = QtWidgets.QPushButton("...")
        btn.setFixedWidth(30)
        btn.clicked.connect(self._browse_export)
        r.addWidget(self._folder_edit)
        r.addWidget(btn)
        col.addWidget(g)

        # reviewer
        g = QtWidgets.QGroupBox("Reviewer")
        r = QtWidgets.QVBoxLayout(g)
        self._user_combo = QtWidgets.QComboBox()
        self._user_combo.currentIndexChanged.connect(self._on_user_changed)
        r.addWidget(self._user_combo)
        col.addWidget(g)

        # overlay + edit
        g = QtWidgets.QGroupBox("Mask Overlay / Edit")
        f = QtWidgets.QFormLayout(g)
        self._alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._alpha_slider.setRange(0, 255)
        self._alpha_slider.setValue(self._alpha)
        self._alpha_slider.valueChanged.connect(self._on_alpha)
        f.addRow("Alpha:", self._alpha_slider)
        self._edit_check = QtWidgets.QCheckBox("Edit mode (E)")
        self._edit_check.stateChanged.connect(self._toggle_edit)
        f.addRow(self._edit_check)
        bm = QtWidgets.QHBoxLayout()
        self._btn_add = QtWidgets.QRadioButton("Add")
        self._btn_erase = QtWidgets.QRadioButton("Erase")
        self._btn_add.setChecked(True)
        self._btn_add.toggled.connect(lambda: setattr(self, "_brush_add", self._btn_add.isChecked()))
        bm.addWidget(self._btn_add)
        bm.addWidget(self._btn_erase)
        f.addRow("Brush:", bm)
        self._size_spin = QtWidgets.QSpinBox()
        self._size_spin.setRange(1, 100)
        self._size_spin.setValue(self._brush_radius)
        self._size_spin.valueChanged.connect(lambda v: setattr(self, "_brush_radius", max(1, v)))
        f.addRow("Size:", self._size_spin)
        for label, slot in [("Fill holes", self._fill_holes), ("Remove dust", self._remove_dust), ("Smooth", self._smooth)]:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(slot)
            f.addRow(b)
        col.addWidget(g)

        # verdict
        g = QtWidgets.QGroupBox("Verdict")
        v = QtWidgets.QVBoxLayout(g)
        for text, slot, color in [
            ("Good  (G)", self._mark_good, VERDICT_COLORS["good"]),
            ("Bad + Save Delta  (B)", self._mark_bad, VERDICT_COLORS["bad"]),
            ("Skip  (S)", self._skip, None),
        ]:
            b = QtWidgets.QPushButton(text)
            if color:
                b.setStyleSheet(f"QPushButton {{ color: {color}; font-weight: bold; }}")
            b.clicked.connect(slot)
            v.addWidget(b)
        self._verdict_label = QtWidgets.QLabel("")
        self._verdict_label.setAlignment(QtCore.Qt.AlignCenter)
        v.addWidget(self._verdict_label)
        col.addWidget(g)

        col.addStretch()
        self._progress_label = QtWidgets.QLabel("No frames loaded")
        col.addWidget(self._progress_label)

        root.addWidget(left)

        # view
        right = QtWidgets.QVBoxLayout()
        self._scene = QtWidgets.QGraphicsScene()
        self._view = ReviewGraphicsView(self._scene)
        self._view.setMouseTracking(True)
        self._view.viewport().installEventFilter(self)
        right.addWidget(self._view, stretch=1)
        self._info_label = QtWidgets.QLabel("")
        self._info_label.setAlignment(QtCore.Qt.AlignCenter)
        right.addWidget(self._info_label)
        root.addLayout(right, stretch=1)

        # hidden spin for reviewers-per-frame (used by auto-generation)
        self._rpf = 2

    def _build_menu(self) -> None:
        m = self.menuBar()
        fm = m.addMenu("File")
        fm.addAction("Open Export Folder...", self._browse_export)
        cm = m.addMenu("Consensus")
        cm.addAction("Compute Scores", self._compute_consensus)

    # ── events ────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        k = event.key()
        if k == QtCore.Qt.Key_Right:
            self._go(1)
        elif k == QtCore.Qt.Key_Left:
            self._go(-1)
        elif k == QtCore.Qt.Key_G:
            self._mark_good()
        elif k == QtCore.Qt.Key_B:
            self._mark_bad()
        elif k == QtCore.Qt.Key_S:
            self._skip()
        elif k == QtCore.Qt.Key_E:
            self._edit_check.setChecked(not self._edit_check.isChecked())
        elif k == QtCore.Qt.Key_F:
            self._view.fit_scene_down_only()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is not self._view.viewport():
            return False
        t = event.type()
        if t == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton and self._edit_mode:
            self._painting = True
            self._last_pt = None
            self._brush_at(event.pos(), event.modifiers())
            return True
        if t == QtCore.QEvent.MouseMove:
            self._update_brush_cursor(event.pos())
            if self._painting and self._edit_mode:
                self._brush_at(event.pos(), event.modifiers())
                return True
        if t == QtCore.QEvent.MouseButtonRelease and self._painting:
            self._painting = False
            self._last_pt = None
            return True
        return False

    # ── loading ───────────────────────────────────────────────────────

    def _browse_export(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder")
        if d:
            self._load_export(Path(d))

    def _load_export(self, root: Path) -> None:
        sp = root / "sample_points.csv"
        if not sp.exists():
            QtWidgets.QMessageBox.warning(self, "Error", f"No sample_points.csv in\n{root}")
            return
        self._export_root = root
        self._folder_edit.setText(str(root))

        self._sample_lookup = {}
        with sp.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["segment"], int(row["frame"]), int(row["step"]))
                self._sample_lookup[key] = row

        vol_path = root / "volunteers.csv"
        self._volunteers = read_volunteers(vol_path)
        self._user_combo.blockSignals(True)
        self._user_combo.clear()
        self._user_combo.addItem("-- select reviewer --")
        for v in self._volunteers:
            self._user_combo.addItem(v)
        self._user_combo.blockSignals(False)

        if not self._volunteers:
            QtWidgets.QMessageBox.information(self, "Setup", "Create volunteers.csv with a 'name' column in the export folder.")
            return

        ap = root / "assignments.csv"
        if ap.exists():
            self._assignments = read_assignments(ap)
        else:
            self._assignments = []
        if not self._assignments and self._volunteers:
            self._assignments = generate_assignments(sp, vol_path, self._rpf)
            if self._assignments:
                write_assignments(self._assignments, ap)
                logger.info("Auto-generated %d assignments (%d reviewers/frame)", len(self._assignments), self._rpf)

        self._user_combo.setCurrentIndex(0)
        self._user_frames = []
        self._idx = 0
        self._verdicts = {}
        self._refresh()

    def _on_user_changed(self, idx: int) -> None:
        if idx <= 0 or not self._export_root:
            self._user_frames = []
            self._verdicts = {}
        else:
            name = self._user_combo.currentText()
            self._user_frames = assignments_for_user(self._assignments, name)
            self._verdicts = load_verdicts(self._export_root, name)
        self._idx = 0
        self._refresh()

    # ── navigation ────────────────────────────────────────────────────

    def _go(self, delta: int) -> None:
        if not self._user_frames:
            return
        self._idx = max(0, min(self._idx + delta, len(self._user_frames) - 1))
        self._refresh()

    def _fk(self) -> Optional[FrameKey]:
        if not self._user_frames or self._idx >= len(self._user_frames):
            return None
        return self._user_frames[self._idx]

    def _capture_view_state(self) -> Optional[Tuple[float, float, float, float]]:
        rect = self._view.viewport().rect()
        if rect.isNull():
            return None
        center = self._view.mapToScene(rect.center())
        transform = self._view.transform()
        return transform.m11(), transform.m22(), center.x(), center.y()

    def _restore_view_state(self, state: Optional[Tuple[float, float, float, float]]) -> bool:
        if state is None:
            return False
        sx, sy, cx, cy = state
        if sx <= 0 or sy <= 0:
            return False
        self._view.resetTransform()
        self._view.scale(sx, sy)
        self._view.centerOn(cx, cy)
        return True

    # ── rendering ─────────────────────────────────────────────────────

    def _refresh(self) -> None:
        fk = self._fk()
        preserve_view = fk is not None and fk == self._rendered_fk
        view_state = self._capture_view_state() if preserve_view else None
        self._scene.clear()
        self._brush_cursor_item = None
        self._update_progress()
        if fk is None or not self._export_root:
            self._info_label.setText("No frame selected")
            self._verdict_label.setText("")
            self._rendered_fk = None
            return

        seg, frame, step = fk
        sample = self._sample_lookup.get(fk, {})
        img_path = sample.get("image_path", "")
        msk_path = sample.get("mask_path", "")

        if not img_path or not Path(img_path).exists():
            self._info_label.setText(f"Image missing: {Path(img_path).name if img_path else '?'}")
            self._rendered_fk = None
            return

        img = np.array(Image.open(img_path).convert("RGB"))

        user = self._user_combo.currentText() if self._user_combo.currentIndex() > 0 else None
        delta = load_delta_mask(self._export_root, user, seg, frame) if user else None

        if delta is not None:
            mask = delta
        elif msk_path and Path(msk_path).exists():
            mask = (np.array(Image.open(msk_path).convert("L")) > 127).astype(np.uint8)
        else:
            mask = None

        self._mask = mask.copy() if mask is not None else np.zeros(img.shape[:2], dtype=np.uint8)

        comp = self._composite(img, self._mask)
        qimg = numpy_to_qimage(comp)
        pixmap_item = self._scene.addPixmap(QtGui.QPixmap.fromImage(qimg))
        self._scene.setSceneRect(pixmap_item.boundingRect())
        if not self._restore_view_state(view_state):
            self._view.fit_scene_down_only()
        self._rendered_fk = fk

        key_s = frame_key_str(seg, frame, step)
        v = self._verdicts.get(key_s, "pending")
        color = VERDICT_COLORS.get(v, "#757575")
        self._verdict_label.setText(f"<b style='color:{color}'>{v.upper()}</b>")
        self._info_label.setText(f"{seg}  |  frame {frame}  step {step}  |  [{self._idx + 1}/{len(self._user_frames)}]")

    def _composite(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = img.copy()
        if mask is None or mask.shape[:2] != img.shape[:2] or not mask.any():
            return out
        a = self._alpha / 255.0
        green = np.array([0, 200, 0], dtype=np.float32)
        yellow = np.array([255, 255, 0], dtype=np.float32)
        region = mask > 0
        out_f = out.astype(np.float32)
        out_f[region] = out_f[region] * (1 - a) + green * a
        outline = _mask_outline(mask)
        if outline is not None and outline.any():
            out_f[outline > 0] = yellow
        return out_f.astype(np.uint8)

    # ── brush cursor ──────────────────────────────────────────────────

    def _update_brush_cursor(self, viewport_pos: QtCore.QPoint) -> None:
        if not self._edit_mode:
            if self._brush_cursor_item:
                self._scene.removeItem(self._brush_cursor_item)
                self._brush_cursor_item = None
            return
        sp = self._view.mapToScene(viewport_pos)
        r = self._brush_radius
        if self._brush_cursor_item is None:
            pen = QtGui.QPen(QtGui.QColor(255, 255, 0, 180), 1.5)
            self._brush_cursor_item = self._scene.addEllipse(-r, -r, r * 2, r * 2, pen)
            self._brush_cursor_item.setZValue(100)
        self._brush_cursor_item.setRect(sp.x() - r, sp.y() - r, r * 2, r * 2)

    # ── edit mode ─────────────────────────────────────────────────────

    def _toggle_edit(self, state: int) -> None:
        self._edit_mode = state == QtCore.Qt.Checked
        self._view.setCursor(QtCore.Qt.CrossCursor if self._edit_mode else QtCore.Qt.ArrowCursor)
        self._view.setDragMode(
            QtWidgets.QGraphicsView.NoDrag
            if self._edit_mode
            else QtWidgets.QGraphicsView.ScrollHandDrag
        )
        if not self._edit_mode and self._brush_cursor_item:
            self._scene.removeItem(self._brush_cursor_item)
            self._brush_cursor_item = None

    def _brush_at(self, vp_pos: QtCore.QPoint, mods: QtCore.Qt.KeyboardModifiers) -> None:
        if self._mask is None:
            return
        sp = self._view.mapToScene(vp_pos)
        x, y = int(round(sp.x())), int(round(sp.y()))
        add = self._brush_add if not (mods & QtCore.Qt.ShiftModifier) else not self._brush_add

        if self._last_pt is not None:
            x0, y0 = self._last_pt
            dx, dy = x - x0, y - y0
            steps = max(1, int(max(abs(dx), abs(dy))))
            for s in range(steps + 1):
                self._stamp(int(round(x0 + dx * s / steps)), int(round(y0 + dy * s / steps)), add)
        else:
            self._stamp(x, y, add)
        self._last_pt = (x, y)
        self._dirty = True
        self._refresh()

    def _stamp(self, cx: int, cy: int, add: bool) -> None:
        if self._mask is None:
            return
        h, w = self._mask.shape
        r = self._brush_radius
        y0, y1 = max(0, cy - r), min(h - 1, cy + r)
        x0, x1 = max(0, cx - r), min(w - 1, cx + r)
        if y0 > y1 or x0 > x1:
            return
        cached = self._brush_cache.get(r)
        if cached is None:
            sz = r * 2 + 1
            yy, xx = np.ogrid[:sz, :sz]
            cached = (yy - r) ** 2 + (xx - r) ** 2 <= r * r
            self._brush_cache[r] = cached
        by0, bx0 = y0 - (cy - r), x0 - (cx - r)
        sub = cached[by0: by0 + (y1 - y0 + 1), bx0: bx0 + (x1 - x0 + 1)]
        if add:
            self._mask[y0: y1 + 1, x0: x1 + 1][sub] = 1
        else:
            self._mask[y0: y1 + 1, x0: x1 + 1][sub] = 0

    # ── mask ops ──────────────────────────────────────────────────────

    def _fill_holes(self) -> None:
        if self._mask is None:
            return
        self._mask[:] = binary_fill_holes(self._mask > 0).astype(np.uint8)
        self._dirty = True
        self._refresh()

    def _remove_dust(self) -> None:
        if self._mask is None:
            return
        labeled, n = nd_label(self._mask > 0)
        if n <= 0:
            return
        counts = np.bincount(labeled.ravel())
        keep = counts >= 25
        if keep.size:
            keep[0] = False
        self._mask[:] = keep[labeled].astype(np.uint8)
        self._dirty = True
        self._refresh()

    def _smooth(self) -> None:
        if self._mask is None:
            return
        s = np.ones((3, 3), dtype=bool)
        self._mask[:] = binary_closing(binary_opening(self._mask > 0, structure=s), structure=s).astype(np.uint8)
        self._dirty = True
        self._refresh()

    # ── overlay ───────────────────────────────────────────────────────

    def _on_alpha(self, v: int) -> None:
        self._alpha = v
        self._refresh()

    # ── verdicts ──────────────────────────────────────────────────────

    def _username(self) -> Optional[str]:
        return self._user_combo.currentText() if self._user_combo.currentIndex() > 0 else None

    def _mark_good(self) -> None:
        self._set_verdict("good")

    def _mark_bad(self) -> None:
        user = self._username()
        fk = self._fk()
        if user and fk and self._export_root and self._mask is not None:
            save_delta_mask(self._export_root, user, fk[0], fk[1], self._mask)
        self._set_verdict("bad")

    def _skip(self) -> None:
        self._go(1)

    def _set_verdict(self, verdict: str) -> None:
        user = self._username()
        fk = self._fk()
        if not user or not fk or not self._export_root:
            QtWidgets.QMessageBox.warning(self, "Error", "Select a reviewer first.")
            return
        self._verdicts[frame_key_str(*fk)] = verdict
        save_verdicts(self._export_root, user, self._verdicts)
        self._go(1)

    def _update_progress(self) -> None:
        n = len(self._user_frames)
        done = sum(1 for fk in self._user_frames if frame_key_str(*fk) in self._verdicts)
        g = sum(1 for v in self._verdicts.values() if v == "good")
        b = sum(1 for v in self._verdicts.values() if v == "bad")
        self._progress_label.setText(f"{done}/{n} reviewed  \u2022  {g} good  {b} bad" if n else "No frames loaded")

    # ── consensus ─────────────────────────────────────────────────────

    def _compute_consensus(self) -> None:
        if not self._export_root:
            return
        rows = compute_consensus(self._export_root)
        if not rows:
            QtWidgets.QMessageBox.warning(self, "Consensus", "No data.")
            return
        complete = sum(1 for r in rows if r["num_pending"] == 0)
        scores = [r["consensus_score"] for r in rows if r["consensus_score"] is not None]
        avg = np.mean(scores) if scores else 0
        QtWidgets.QMessageBox.information(
            self, "Consensus",
            f"{len(rows)} frames  |  {complete} fully reviewed  |  avg score {avg:.0%}\n\n"
            f"Written to {self._export_root / 'consensus.csv'}",
        )
