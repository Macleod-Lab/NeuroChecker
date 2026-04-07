from typing import List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets


class SegmentBarWidget(QtWidgets.QWidget):
    segmentClicked = QtCore.pyqtSignal(int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._segments: List[Tuple[int, int, int, QtGui.QColor]] = []
        self._rows: List[List[Tuple[int, int, int, QtGui.QColor]]] = []
        self._active_segment: Optional[int] = None
        self._current_frame: Optional[int] = None
        self._rects: List[Tuple[int, int, int, QtCore.QRect]] = []
        self._min_frame = 0
        self._max_frame = 0
        self._row_height = 10
        self._row_gap = 4
        self._pad = 6
        self.setMinimumHeight(18)

    def set_segments(self, segments: List[Tuple[int, int, int, QtGui.QColor]]) -> None:
        self._segments = segments
        if segments:
            self._min_frame = min(s for _, s, _, _ in segments)
            self._max_frame = max(e for _, _, e, _ in segments)
        else:
            self._min_frame = 0
            self._max_frame = 0
        rows: List[List[Tuple[int, int, int, QtGui.QColor]]] = []
        row_ends: List[int] = []
        for seg in sorted(segments, key=lambda item: (item[1], item[2], item[0])):
            seg_id, start, end, color = seg
            placed = False
            for idx, last_end in enumerate(row_ends):
                if start > last_end:
                    rows[idx].append(seg)
                    row_ends[idx] = end
                    placed = True
                    break
            if not placed:
                rows.append([seg])
                row_ends.append(end)
        self._rows = rows
        total_h = self._pad * 2
        if rows:
            total_h += len(rows) * self._row_height + (len(rows) - 1) * self._row_gap
        self.setMinimumHeight(max(18, total_h))
        self.setMaximumHeight(max(18, total_h))
        self.update()

    def set_active(self, segment_id: Optional[int]) -> None:
        self._active_segment = segment_id
        self.update()

    def set_current_frame(self, frame: Optional[int]) -> None:
        self._current_frame = int(frame) if frame is not None else None
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor(18, 18, 18, 255))
        self._rects = []
        if not self._rows:
            return
        frame_span = max(1, self._max_frame - self._min_frame + 1)
        for row_idx, row in enumerate(self._rows):
            y = rect.top() + self._pad + row_idx * (self._row_height + self._row_gap)
            for seg_id, start, end, color in row:
                x0 = int(
                    rect.left()
                    + self._pad
                    + (start - self._min_frame) / float(frame_span) * (rect.width() - 2 * self._pad)
                )
                x1 = int(
                    rect.left()
                    + self._pad
                    + (end - self._min_frame + 1) / float(frame_span) * (rect.width() - 2 * self._pad)
                )
                w = max(2, x1 - x0)
                segment_rect = QtCore.QRect(x0, y, w, self._row_height)
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(color)
                painter.drawRoundedRect(segment_rect, 3, 3)
                if self._active_segment == seg_id:
                    pen = QtGui.QPen(QtGui.QColor(80, 220, 120, 255), 2)
                    painter.setPen(pen)
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawRoundedRect(segment_rect.adjusted(1, 1, -2, -2), 3, 3)
                self._rects.append((seg_id, start, end, segment_rect))
        if (
            self._current_frame is not None
            and self._min_frame <= self._current_frame <= self._max_frame
            and frame_span > 0
        ):
            x = int(
                rect.left()
                + self._pad
                + (self._current_frame - self._min_frame) / float(frame_span) * (rect.width() - 2 * self._pad)
            )
            pen = QtGui.QPen(QtGui.QColor(80, 160, 255, 230), 2)
            painter.setPen(pen)
            painter.drawLine(x, rect.top() + 2, x, rect.bottom() - 2)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = event.pos()
        for seg_id, start, end, rect in self._rects:
            if rect.contains(pos):
                if end <= start:
                    frame = start
                else:
                    rel = (pos.x() - rect.left()) / float(max(1, rect.width() - 1))
                    rel = max(0.0, min(1.0, rel))
                    frame = int(round(start + rel * (end - start)))
                self.segmentClicked.emit(seg_id, frame)
                break
        super().mousePressEvent(event)
