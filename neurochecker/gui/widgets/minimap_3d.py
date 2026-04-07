import math
from typing import List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


class MiniMap3DWidget(QtWidgets.QWidget):
    nodeContextRequested = QtCore.pyqtSignal(int, QtCore.QPoint)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self._positions = np.zeros((0, 3), dtype=np.float32)
        self._node_ids: List[int] = []
        self._ghost_positions = np.zeros((0, 3), dtype=np.float32)
        self._hillock_positions = np.zeros((0, 3), dtype=np.float32)
        self._distal_positions = np.zeros((0, 3), dtype=np.float32)
        self._edges: List[Tuple[int, int]] = []
        self._bbox_edges: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
        self._bbox_corners: List[Tuple[float, float, float]] = []
        self._flagged_positions = np.zeros((0, 3), dtype=np.float32)
        self._current_index: Optional[int] = None
        self._status_text: Optional[str] = None
        self._node_colors: Optional[List[QtGui.QColor]] = None
        self._edge_colors: Optional[List[QtGui.QColor]] = None
        self._legend_items: List[Tuple[str, QtGui.QColor]] = []
        self._arrow_prev_dir: Optional[Tuple[float, float, float]] = None
        self._arrow_next_dir: Optional[Tuple[float, float, float]] = None
        self._yaw = 0.7
        self._pitch = -0.6
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_mode: Optional[str] = None
        self._last_pos: Optional[QtCore.QPoint] = None

    def clear(self) -> None:
        self._positions = np.zeros((0, 3), dtype=np.float32)
        self._node_ids = []
        self._ghost_positions = np.zeros((0, 3), dtype=np.float32)
        self._hillock_positions = np.zeros((0, 3), dtype=np.float32)
        self._distal_positions = np.zeros((0, 3), dtype=np.float32)
        self._edges = []
        self._bbox_edges = []
        self._bbox_corners = []
        self._flagged_positions = np.zeros((0, 3), dtype=np.float32)
        self._current_index = None
        self._status_text = None
        self._node_colors = None
        self._edge_colors = None
        self._legend_items = []
        self._arrow_prev_dir = None
        self._arrow_next_dir = None
        self.update()

    def reset_view(self) -> None:
        self._yaw = 0.7
        self._pitch = -0.6
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_mode = None
        self._last_pos = None
        self.update()

    def set_data(
        self,
        positions: Optional[np.ndarray],
        edges: List[Tuple[int, int]],
        current_index: Optional[int],
        *,
        node_ids: Optional[List[int]] = None,
        node_colors: Optional[List[QtGui.QColor]] = None,
        edge_colors: Optional[List[QtGui.QColor]] = None,
        legend_items: Optional[List[Tuple[str, QtGui.QColor]]] = None,
        ghost_positions: Optional[np.ndarray] = None,
        bbox_edges: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = None,
        bbox_corners: Optional[List[Tuple[float, float, float]]] = None,
        flagged_positions: Optional[np.ndarray] = None,
        hillock_positions: Optional[np.ndarray] = None,
        distal_positions: Optional[np.ndarray] = None,
    ) -> None:
        if positions is None:
            self._positions = np.zeros((0, 3), dtype=np.float32)
        else:
            self._positions = positions.astype(np.float32, copy=False)
        self._node_ids = list(node_ids) if node_ids is not None else list(range(len(self._positions)))
        if ghost_positions is None:
            self._ghost_positions = np.zeros((0, 3), dtype=np.float32)
        else:
            self._ghost_positions = ghost_positions.astype(np.float32, copy=False)
        if hillock_positions is None:
            self._hillock_positions = np.zeros((0, 3), dtype=np.float32)
        else:
            self._hillock_positions = hillock_positions.astype(np.float32, copy=False)
        if distal_positions is None:
            self._distal_positions = np.zeros((0, 3), dtype=np.float32)
        else:
            self._distal_positions = distal_positions.astype(np.float32, copy=False)
        self._edges = edges
        self._bbox_edges = bbox_edges or []
        self._bbox_corners = list(bbox_corners) if bbox_corners else []
        if flagged_positions is None:
            self._flagged_positions = np.zeros((0, 3), dtype=np.float32)
        else:
            self._flagged_positions = flagged_positions.astype(np.float32, copy=False)
        self._current_index = current_index
        self._node_colors = node_colors
        self._edge_colors = edge_colors
        self._legend_items = legend_items or []
        self.update()

    def set_arrows(
        self,
        prev_dir: Optional[Tuple[float, float, float]],
        next_dir: Optional[Tuple[float, float, float]],
    ) -> None:
        self._arrow_prev_dir = prev_dir
        self._arrow_next_dir = next_dir
        self.update()

    def set_status(self, text: Optional[str]) -> None:
        self._status_text = text if text else None
        self.update()

    def _project_points(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if pts.size == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        cy = math.cos(self._yaw)
        sy = math.sin(self._yaw)
        cp = math.cos(self._pitch)
        sp = math.sin(self._pitch)
        x1 = pts[:, 0] * cy - pts[:, 1] * sy
        y1 = pts[:, 0] * sy + pts[:, 1] * cy
        z1 = pts[:, 2]
        y2 = y1 * cp - z1 * sp
        z2 = y1 * sp + z1 * cp
        x2 = x1
        return x2, y2, z2

    def _project(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._project_points(self._positions)

    def _screen_projection(self) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
        xs, ys, _zs = self._project()
        if xs.size == 0:
            return np.zeros(0), np.zeros(0), 1.0, max(1, self.width()), max(1, self.height())
        max_x = float(np.max(np.abs(xs)))
        max_y = float(np.max(np.abs(ys)))
        width = max(1e-6, max_x * 2.0)
        height = max(1e-6, max_y * 2.0)
        w = max(1, self.width())
        h = max(1, self.height())
        scale = 0.85 * min((w - 6) / width, (h - 6) / height)
        scale *= self._zoom
        sx = xs * scale + w / 2.0 + self._pan_x
        sy = h / 2.0 - ys * scale + self._pan_y
        return sx, sy, scale, w, h

    def _screen_points_for(
        self,
        pts: np.ndarray,
        *,
        scale: float,
        w: int,
        h: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if pts.size == 0:
            return np.zeros(0), np.zeros(0)
        xs, ys, _zs = self._project_points(pts)
        return xs * scale + w / 2.0 + self._pan_x, h / 2.0 - ys * scale + self._pan_y

    def _nearest_node_index(self, pos: QtCore.QPoint) -> Optional[int]:
        if self._positions.size == 0:
            return None
        sx, sy, _scale, _w, _h = self._screen_projection()
        if sx.size == 0:
            return None
        dx = sx - float(pos.x())
        dy = sy - float(pos.y())
        dist2 = dx * dx + dy * dy
        best_idx = int(np.argmin(dist2))
        if float(dist2[best_idx]) > 14.0 * 14.0:
            return None
        return best_idx

    def _rotate_dir(self, dx: float, dy: float, dz: float) -> Tuple[float, float, float]:
        cy = math.cos(self._yaw)
        sy = math.sin(self._yaw)
        cp = math.cos(self._pitch)
        sp = math.sin(self._pitch)
        x1 = dx * cy - dy * sy
        y1 = dx * sy + dy * cy
        z1 = dz
        y2 = y1 * cp - z1 * sp
        z2 = y1 * sp + z1 * cp
        x2 = x1
        return x2, y2, z2

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtCore.Qt.transparent)
        if self._positions.size == 0:
            return
        sx, sy, scale, w, h = self._screen_projection()
        if sx.size == 0:
            return

        if self._ghost_positions.size:
            sgx, sgy = self._screen_points_for(self._ghost_positions, scale=scale, w=w, h=h)
            pen = QtGui.QPen(QtGui.QColor(80, 255, 140, 150), 2.0)
            painter.setPen(pen)
            points = QtGui.QPolygonF()
            for x, y in zip(sgx, sgy):
                points.append(QtCore.QPointF(x, y))
            painter.drawPoints(points)

        if self._bbox_edges:
            if len(self._bbox_corners) >= 4:
                corners = np.array(self._bbox_corners, dtype=np.float32)
                bx, by = self._screen_points_for(corners, scale=scale, w=w, h=h)
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(80, 160, 255, 50)))
                poly = QtGui.QPolygonF()
                for x, y in zip(bx, by):
                    poly.append(QtCore.QPointF(x, y))
                painter.drawPolygon(poly)
            pts = np.array([p for edge in self._bbox_edges for p in edge], dtype=np.float32)
            bx, by = self._screen_points_for(pts, scale=scale, w=w, h=h)
            pen = QtGui.QPen(QtGui.QColor(80, 160, 255, 210), 1.6)
            painter.setPen(pen)
            for i in range(0, len(bx), 2):
                painter.drawLine(QtCore.QPointF(bx[i], by[i]), QtCore.QPointF(bx[i + 1], by[i + 1]))

        if self._flagged_positions.size:
            sfx, sfy = self._screen_points_for(self._flagged_positions, scale=scale, w=w, h=h)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 210, 40, 240), 1.8))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 210, 40, 200)))
            for x, y in zip(sfx, sfy):
                painter.drawEllipse(QtCore.QPointF(x, y), 4.2, 4.2)

        if self._hillock_positions.size:
            hx, hy = self._screen_points_for(self._hillock_positions, scale=scale, w=w, h=h)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 160, 40, 245), 2.2))
            painter.setBrush(QtCore.Qt.NoBrush)
            for x, y in zip(hx, hy):
                painter.drawEllipse(QtCore.QPointF(x, y), 6.6, 6.6)

        if self._distal_positions.size:
            dx, dy = self._screen_points_for(self._distal_positions, scale=scale, w=w, h=h)
            painter.setPen(QtGui.QPen(QtGui.QColor(80, 220, 255, 245), 2.2))
            painter.setBrush(QtCore.Qt.NoBrush)
            for x, y in zip(dx, dy):
                painter.drawEllipse(QtCore.QPointF(x, y), 5.6, 5.6)

        if self._edges:
            default_pen = QtGui.QPen(QtGui.QColor(170, 210, 235, 190), 1.4)
            for edge_idx, (i, j) in enumerate(self._edges):
                if self._edge_colors and edge_idx < len(self._edge_colors):
                    pen = QtGui.QPen(self._edge_colors[edge_idx], 1.6)
                else:
                    pen = default_pen
                painter.setPen(pen)
                painter.drawLine(QtCore.QPointF(sx[i], sy[i]), QtCore.QPointF(sx[j], sy[j]))

        painter.setPen(QtCore.Qt.NoPen)
        default_brush = QtGui.QBrush(QtGui.QColor(235, 235, 235, 220))
        for idx in range(len(sx)):
            if self._current_index is not None and idx == self._current_index:
                continue
            if self._node_colors and idx < len(self._node_colors):
                painter.setBrush(self._node_colors[idx])
            else:
                painter.setBrush(default_brush)
            painter.drawEllipse(QtCore.QPointF(sx[idx], sy[idx]), 2.4, 2.4)

        if self._current_index is not None and 0 <= self._current_index < len(sx):
            painter.setBrush(QtGui.QColor(255, 0, 255, 240))
            painter.drawEllipse(
                QtCore.QPointF(sx[self._current_index], sy[self._current_index]), 4.2, 4.2
            )

        if (
            self._current_index is not None
            and 0 <= self._current_index < len(sx)
            and (self._arrow_prev_dir is not None or self._arrow_next_dir is not None)
        ):
            cx = float(sx[self._current_index])
            cy = float(sy[self._current_index])
            perp_offset = 16.0
            arrow_len = 32.0
            head_len = 10.0
            head_angle = 0.6

            def draw_arrow(dir_vec: Optional[Tuple[float, float, float]], color: QtGui.QColor) -> None:
                if dir_vec is None:
                    return
                dx, dy, dz = self._rotate_dir(dir_vec[0], dir_vec[1], dir_vec[2])
                dxs = dx * scale
                dys = -dy * scale
                norm = (dxs * dxs + dys * dys) ** 0.5
                if norm < 1e-6:
                    return
                ux = dxs / norm
                uy = dys / norm
                px = -uy
                py = ux
                ox = cx + px * perp_offset
                oy = cy + py * perp_offset
                ex = ox + ux * arrow_len
                ey = oy + uy * arrow_len
                angle = math.degrees(math.atan2(uy, ux))
                pen = QtGui.QPen(color, 3.6)
                painter.save()
                painter.translate(ox, oy)
                painter.rotate(angle)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawLine(QtCore.QPointF(0.0, 0.0), QtCore.QPointF(arrow_len, 0.0))
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(color))
                head = QtGui.QPolygonF(
                    [
                        QtCore.QPointF(arrow_len, 0.0),
                        QtCore.QPointF(arrow_len - head_len, head_len * head_angle),
                        QtCore.QPointF(arrow_len - head_len, -head_len * head_angle),
                    ]
                )
                painter.drawPolygon(head)
                painter.restore()

            draw_arrow(self._arrow_prev_dir, QtGui.QColor(220, 80, 80, 230))
            draw_arrow(self._arrow_next_dir, QtGui.QColor(80, 160, 255, 230))

        if self._legend_items:
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            metrics = QtGui.QFontMetrics(font)
            y = 6
            for label, color in self._legend_items:
                text_w = metrics.horizontalAdvance(label) + 18
                text_h = metrics.height() + 4
                rect = QtCore.QRect(6, y, text_w, text_h)
                painter.setBrush(QtGui.QColor(10, 10, 10, 170))
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawRoundedRect(rect, 4, 4)
                painter.setBrush(color)
                painter.drawRect(rect.left() + 4, rect.top() + 3, 6, rect.height() - 6)
                painter.setPen(QtGui.QColor(240, 240, 240, 230))
                painter.drawText(rect.adjusted(14, 0, -2, -1), QtCore.Qt.AlignVCenter, label)
                y += rect.height() + 4

        if self._arrow_prev_dir is not None or self._arrow_next_dir is not None:
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            metrics = QtGui.QFontMetrics(font)
            labels = [("Left", QtGui.QColor(220, 80, 80, 230)), ("Right", QtGui.QColor(80, 160, 255, 230))]
            total_h = sum(metrics.height() + 4 for _ in labels) + (len(labels) - 1) * 4
            y = max(6, self.height() - total_h - 6)
            for label, color in labels:
                text_w = metrics.horizontalAdvance(label) + 18
                text_h = metrics.height() + 4
                rect = QtCore.QRect(6, y, text_w, text_h)
                painter.setBrush(QtGui.QColor(10, 10, 10, 170))
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawRoundedRect(rect, 4, 4)
                painter.setBrush(color)
                painter.drawRect(rect.left() + 4, rect.top() + 3, 6, rect.height() - 6)
                painter.setPen(QtGui.QColor(240, 240, 240, 230))
                painter.drawText(rect.adjusted(14, 0, -2, -1), QtCore.Qt.AlignVCenter, label)
                y += rect.height() + 4

        if self._status_text:
            painter.setPen(QtGui.QColor(240, 240, 240, 230))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            metrics = QtGui.QFontMetrics(font)
            text = self._status_text
            text_w = metrics.horizontalAdvance(text) + 8
            text_h = metrics.height() + 4
            rect = QtCore.QRect(6, 6, text_w, text_h)
            painter.setBrush(QtGui.QColor(10, 10, 10, 170))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(QtGui.QColor(240, 240, 240, 230))
            painter.drawText(rect.adjusted(4, 0, -4, -1), QtCore.Qt.AlignVCenter, text)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton:
            self._drag_mode = "pan"
            self._last_pos = event.pos()
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton:
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                self._drag_mode = "pan"
            else:
                self._drag_mode = "rotate"
            self._last_pos = event.pos()
            event.accept()
            return
        if event.button() == QtCore.Qt.RightButton:
            node_index = self._nearest_node_index(event.pos())
            if node_index is not None:
                node_id = int(self._node_ids[node_index]) if node_index < len(self._node_ids) else int(node_index)
                self.nodeContextRequested.emit(node_id, event.globalPos())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_mode is None or self._last_pos is None:
            return
        delta = event.pos() - self._last_pos
        self._last_pos = event.pos()
        if self._drag_mode == "pan":
            self._pan_x += float(delta.x())
            self._pan_y += float(delta.y())
        else:
            self._yaw += delta.x() * 0.01
            self._pitch += delta.y() * 0.01
            max_pitch = math.radians(88.0)
            if self._pitch > max_pitch:
                self._pitch = max_pitch
            elif self._pitch < -max_pitch:
                self._pitch = -max_pitch
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.MiddleButton):
            self._drag_mode = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.MiddleButton):
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
            self._zoom *= math.pow(1.0015, float(delta))
            self._zoom = max(0.1, min(25.0, self._zoom))
            self.update()
