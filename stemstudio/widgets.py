"""Widget de forma de onda: playhead, clique-para-buscar e seleção de loop A–B."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .theme import ACCENT, PANEL, TEXT_DIM


def compute_peaks(stems: dict[str, np.ndarray], frames: int, n_bins: int = 1600) -> np.ndarray:
    """Resume a mixagem (todas as faixas, ganho 1) em n_bins picos para desenho."""
    if not stems or frames <= 0:
        return np.zeros(n_bins, dtype=np.float32)
    mix = np.zeros(frames, dtype=np.float32)
    for data in stems.values():
        mix += np.abs(data[:frames]).mean(axis=1)
    step = max(1, frames // n_bins)
    usable = (frames // step) * step
    peaks = mix[:usable].reshape(-1, step).max(axis=1)
    m = peaks.max()
    return (peaks / m).astype(np.float32) if m > 0 else peaks.astype(np.float32)


class WaveformWidget(QWidget):
    seek_requested = Signal(float)            # fração 0–1
    region_selected = Signal(float, float)    # frações A, B

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(96)
        self.setCursor(Qt.PointingHandCursor)
        self._peaks = np.zeros(1600, dtype=np.float32)
        self._playhead = 0.0
        self._loop: tuple[float, float] | None = None
        self._loop_enabled = False
        self._press_x: int | None = None
        self._dragging = False
        self._drag_a = 0.0
        self._drag_b = 0.0
        self.setToolTip(
            "Clique para buscar uma posição.\n"
            "Arraste para selecionar um trecho de loop A–B."
        )

    # ---------- estado ----------
    def set_peaks(self, peaks: np.ndarray):
        self._peaks = peaks
        self.update()

    def set_playhead(self, frac: float):
        if abs(frac - self._playhead) > 1e-4:
            self._playhead = frac
            self.update()

    def set_loop(self, region: tuple[float, float] | None, enabled: bool):
        self._loop = region
        self._loop_enabled = enabled
        self.update()

    # ---------- mouse ----------
    def _frac(self, x: int) -> float:
        return min(1.0, max(0.0, x / max(1, self.width())))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._press_x = ev.position().x()
            self._dragging = False

    def mouseMoveEvent(self, ev):
        if self._press_x is None:
            return
        x = ev.position().x()
        if abs(x - self._press_x) > 5:
            self._dragging = True
            self._drag_a = self._frac(min(self._press_x, x))
            self._drag_b = self._frac(max(self._press_x, x))
            self.update()

    def mouseReleaseEvent(self, ev):
        if self._press_x is None:
            return
        if self._dragging and (self._drag_b - self._drag_a) > 0.005:
            self.region_selected.emit(self._drag_a, self._drag_b)
        else:
            self.seek_requested.emit(self._frac(ev.position().x()))
        self._press_x = None
        self._dragging = False
        self.update()

    # ---------- desenho ----------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setBrush(QColor(PANEL))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(0, 0, w, h), 10, 10)

        # região de loop (selecionada ou em arrasto)
        region = (self._drag_a, self._drag_b) if self._dragging else self._loop
        if region:
            a, b = region
            color = QColor(ACCENT)
            color.setAlpha(70 if self._loop_enabled or self._dragging else 36)
            p.setBrush(color)
            p.drawRect(QRectF(a * w, 0, (b - a) * w, h))

        # picos
        n = len(self._peaks)
        if n:
            mid = h / 2
            bar_w = w / n
            p.setPen(Qt.NoPen)
            played_x = self._playhead * w
            base = QColor("#4a5160")
            played = QColor(ACCENT)
            for i, v in enumerate(self._peaks):
                x = i * bar_w
                bh = max(1.5, v * (h * 0.85))
                p.setBrush(played if x <= played_x else base)
                p.drawRect(QRectF(x, mid - bh / 2, max(1.0, bar_w * 0.7), bh))

        # marcadores A/B
        if region:
            a, b = region
            pen = QPen(QColor(ACCENT), 2)
            p.setPen(pen)
            p.drawLine(int(a * w), 0, int(a * w), h)
            p.drawLine(int(b * w), 0, int(b * w), h)

        # playhead
        pen = QPen(QColor("#ffffff"), 1.5)
        p.setPen(pen)
        x = int(self._playhead * w)
        p.drawLine(x, 4, x, h - 4)

        if not np.any(self._peaks):
            p.setPen(QColor(TEXT_DIM))
            p.drawText(self.rect(), Qt.AlignCenter, "Abra uma música e separe os instrumentos")
        p.end()
