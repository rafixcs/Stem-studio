"""Janela principal — PySide6, tema escuro estilo DAW."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSlider,
    QStatusBar, QVBoxLayout, QWidget,
)

from . import guitar_split, metronome, separator, tempo
from .player import StemPlayer
from .theme import stem_color
from .widgets import WaveformWidget, compute_peaks

STEM_LABELS = {
    "vocals": "Vocais", "drums": "Bateria", "bass": "Baixo",
    "guitar": "Guitarras", "piano": "Piano", "other": "Outros",
    "click": "Metrônomo",
}
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
CACHE_ROOT = Path.home() / ".stemstudio" / "cache"


# ---------------------------------------------------------------- workers
class SeparationWorker(QThread):
    progress = Signal(str)
    percent = Signal(int)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, audio_path: str, cache_root: str):
        super().__init__()
        self.audio_path, self.cache_root = audio_path, cache_root

    def run(self):
        try:
            stems = separator.separate(
                self.audio_path, self.cache_root,
                progress_cb=self.progress.emit, percent_cb=self.percent.emit,
            )
            self.done.emit({k: str(v) for k, v in stems.items()})
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class GuitarWorker(QThread):
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, guitar_array: np.ndarray):
        super().__init__()
        self.y = guitar_array

    def run(self):
        try:
            centers = guitar_split.estimate_pan_centers(self.y)
            if len(centers) <= 1:
                self.done.emit({})
                return
            tracks = guitar_split.split_by_pan(self.y, centers)
            parts = {
                f"Guitarra {i + 1} ({guitar_split.pan_label(c)})": t
                for i, (c, t) in enumerate(zip(centers, tracks))
            }
            self.done.emit(parts)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class StretchWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(dict, int)
    failed = Signal(str)

    def __init__(self, arrays: dict[str, np.ndarray], sr: int, rate: float):
        super().__init__()
        self.arrays, self.sr, self.rate = arrays, sr, rate

    def run(self):
        try:
            out = tempo.stretch_many(self.arrays, self.sr, self.rate,
                                     progress_cb=self.progress.emit)
            self.done.emit(out, self.sr)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class BpmWorker(QThread):
    done = Signal(float, object, object)
    failed = Signal(str)

    def __init__(self, path: str, cache_dir: Path):
        super().__init__()
        self.path, self.cache_dir = path, cache_dir

    def run(self):
        try:
            npz = self.cache_dir / "beats.npz"
            if npz.exists():
                d = np.load(npz, allow_pickle=False)
                segs = [tuple(row) for row in d["segments"]]
                self.done.emit(float(d["bpm"]), d["beats"], segs)
                return
            bpm, beats, segments = metronome.detect(self.path)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(npz, bpm=bpm, beats=beats,
                     segments=np.array(segments, dtype=np.float64).reshape(-1, 3))
            self.done.emit(bpm, beats, segments)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


# ---------------------------------------------------------------- janela
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stem Studio")
        self.resize(860, 720)
        self.setAcceptDrops(True)

        self.player = StemPlayer()
        self.original_arrays: dict[str, np.ndarray] | None = None
        self.original_sr = 44100
        self.original_bpm: float | None = None
        self.beat_times: np.ndarray | None = None
        self.tempo_segments: list = []
        self.current_rate: float = 1.0
        self.audio_path: str | None = None
        self.settings = QSettings("StemStudio", "StemStudio")
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._shortcuts()

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 10)
        root.setSpacing(10)

        # arquivo
        file_row = QHBoxLayout()
        self.btn_open = QPushButton("Abrir música…")
        self.btn_open.setToolTip("Ou arraste um arquivo de áudio para a janela")
        self.btn_open.clicked.connect(self.open_file)
        self.lbl_file = QLabel("Nenhum arquivo — arraste uma música para cá")
        self.lbl_file.setProperty("kind", "dim")
        self.btn_separate = QPushButton("Separar instrumentos")
        self.btn_separate.setProperty("kind", "accent")
        self.btn_separate.setEnabled(False)
        self.btn_separate.clicked.connect(self.run_separation)
        file_row.addWidget(self.btn_open)
        file_row.addWidget(self.lbl_file, 1)
        file_row.addWidget(self.btn_separate)
        root.addLayout(file_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.hide()
        root.addWidget(self.progress)

        # forma de onda
        self.wave = WaveformWidget()
        self.wave.seek_requested.connect(self._seek)
        self.wave.region_selected.connect(self._region_selected)
        root.addWidget(self.wave)

        # transporte
        tr = QHBoxLayout()
        tr.setSpacing(8)
        self.btn_play = QPushButton("▶")
        self.btn_play.setProperty("kind", "accent")
        self.btn_play.setFixedWidth(52)
        self.btn_play.setToolTip("Tocar/Pausar (Espaço)")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setFixedWidth(40)
        self.btn_stop.setToolTip("Parar e voltar ao início (Home)")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)
        self.lbl_time = QLabel("0:00.0 / 0:00.0")
        self.lbl_time.setProperty("kind", "time")
        self.btn_loop = QPushButton("Loop A–B")
        self.btn_loop.setProperty("kind", "loop")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setEnabled(False)
        self.btn_loop.setToolTip("Repete o trecho selecionado na forma de onda (L)\nArraste na forma de onda para selecionar")
        self.btn_loop.toggled.connect(self._loop_toggled)
        self.btn_clear_loop = QPushButton("✕")
        self.btn_clear_loop.setProperty("kind", "tiny")
        self.btn_clear_loop.setToolTip("Limpar seleção de loop")
        self.btn_clear_loop.setEnabled(False)
        self.btn_clear_loop.clicked.connect(self._clear_loop)
        lbl_master = QLabel("Master")
        lbl_master.setProperty("kind", "dim")
        self.master_slider = QSlider(Qt.Horizontal)
        self.master_slider.setRange(0, 150)
        self.master_slider.setValue(100)
        self.master_slider.setFixedWidth(140)
        self.master_slider.setToolTip("Volume geral")
        self.master_slider.valueChanged.connect(lambda v: self.player.set_master(v / 100))
        tr.addWidget(self.btn_play)
        tr.addWidget(self.btn_stop)
        tr.addWidget(self.lbl_time)
        tr.addSpacing(10)
        tr.addWidget(self.btn_loop)
        tr.addWidget(self.btn_clear_loop)
        tr.addStretch(1)
        tr.addWidget(lbl_master)
        tr.addWidget(self.master_slider)
        root.addLayout(tr)

        # mixer
        self.mixer_box = QGroupBox("Instrumentos")
        self.mixer_grid = QGridLayout(self.mixer_box)
        self.mixer_grid.setHorizontalSpacing(10)
        self.mixer_grid.setVerticalSpacing(6)
        self.mixer_box.setEnabled(False)
        self.sliders: dict[str, QSlider] = {}
        self.gain_labels: dict[str, QLabel] = {}
        root.addWidget(self.mixer_box)

        self.lbl_guitars = QLabel("")
        self.lbl_guitars.setProperty("kind", "dim")
        root.addWidget(self.lbl_guitars)

        # andamento
        tempo_box = QGroupBox("Andamento")
        trow = QHBoxLayout(tempo_box)
        self.lbl_bpm = QLabel("BPM: —")
        self.btn_half = QPushButton("÷2")
        self.btn_half.setProperty("kind", "tiny")
        self.btn_half.setToolTip("O metrônomo está no dobro do tempo real? Divide pela metade.")
        self.btn_half.setEnabled(False)
        self.btn_half.clicked.connect(lambda: self._octave(0.5))
        self.btn_double = QPushButton("×2")
        self.btn_double.setProperty("kind", "tiny")
        self.btn_double.setToolTip("O metrônomo está na metade do tempo real? Dobra.")
        self.btn_double.setEnabled(False)
        self.btn_double.clicked.connect(lambda: self._octave(2.0))
        self.spin_bpm = QDoubleSpinBox()
        self.spin_bpm.setRange(30, 300)
        self.spin_bpm.setDecimals(1)
        self.spin_bpm.setValue(120.0)
        self.spin_bpm.setToolTip("Novo andamento desejado")
        self.lbl_pct = QLabel("")
        self.lbl_pct.setProperty("kind", "dim")
        self.btn_apply_bpm = QPushButton("Aplicar")
        self.btn_apply_bpm.setProperty("kind", "accent")
        self.btn_apply_bpm.setEnabled(False)
        self.btn_apply_bpm.clicked.connect(self.apply_bpm)
        self.btn_reset_bpm = QPushButton("Original")
        self.btn_reset_bpm.setEnabled(False)
        self.btn_reset_bpm.clicked.connect(self.reset_bpm)
        trow.addWidget(self.lbl_bpm)
        trow.addWidget(self.btn_half)
        trow.addWidget(self.btn_double)
        trow.addStretch(1)
        trow.addWidget(QLabel("Novo BPM:"))
        trow.addWidget(self.spin_bpm)
        trow.addWidget(self.lbl_pct)
        trow.addWidget(self.btn_apply_bpm)
        trow.addWidget(self.btn_reset_bpm)
        self.spin_bpm.valueChanged.connect(self._update_pct)
        root.addWidget(tempo_box)

        # exportação
        ex = QHBoxLayout()
        ex.addStretch(1)
        self.btn_export = QPushButton("Exportar mixagem…")
        self.btn_export.setToolTip("Salva a mixagem atual (volumes, solos e andamento) em WAV")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_mix)
        self.btn_export_stems = QPushButton("Exportar faixas…")
        self.btn_export_stems.setToolTip("Salva cada faixa separada como WAV (no andamento atual)")
        self.btn_export_stems.setEnabled(False)
        self.btn_export_stems.clicked.connect(self.export_stems)
        ex.addWidget(self.btn_export)
        ex.addWidget(self.btn_export_stems)
        root.addLayout(ex)

        root.addStretch(1)
        self.setCentralWidget(central)
        for btn in central.findChildren(QPushButton):
            btn.setFocusPolicy(Qt.NoFocus)
        hint = QLabel("Espaço: tocar/pausar   ·   ←/→: ±5 s   ·   L: loop   ·   Home: início")
        hint.setProperty("kind", "dim")
        root.addWidget(hint)
        self.setStatusBar(QStatusBar())
        if not tempo.rubberband_available():
            self.statusBar().showMessage(
                "Aviso: 'rubberband' não encontrado — o stretch usará o librosa (qualidade menor)."
            )

    def _shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence(Qt.Key_Home), self, activated=self.stop)
        QShortcut(QKeySequence(Qt.Key_L), self, activated=lambda: self.btn_loop.toggle())
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self._nudge(-5))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self._nudge(5))

    def _rebuild_mixer(self):
        while self.mixer_grid.count():
            item = self.mixer_grid.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self.sliders.clear()
        self.gain_labels.clear()

        for row, name in enumerate(self.player.stems.keys()):
            chip = QFrame()
            chip.setFixedSize(10, 10)
            chip.setStyleSheet(f"background: {stem_color(name)}; border-radius: 5px;")
            self.mixer_grid.addWidget(chip, row, 0)

            self.mixer_grid.addWidget(QLabel(STEM_LABELS.get(name, name)), row, 1)

            b_solo = QPushButton("S")
            b_solo.setProperty("kind", "solo")
            b_solo.setCheckable(True)
            b_solo.setFixedWidth(28)
            b_solo.setToolTip("Solo — ouve apenas as faixas em solo")
            b_solo.setChecked(self.player.solos.get(name, False))
            b_solo.toggled.connect(lambda v, n=name: self.player.set_solo(n, v))
            self.mixer_grid.addWidget(b_solo, row, 2)

            b_mute = QPushButton("M")
            b_mute.setProperty("kind", "mute")
            b_mute.setCheckable(True)
            b_mute.setFixedWidth(28)
            b_mute.setToolTip("Mudo")
            b_mute.setChecked(self.player.mutes.get(name, False))
            b_mute.toggled.connect(lambda v, n=name: self.player.set_mute(n, v))
            self.mixer_grid.addWidget(b_mute, row, 3)

            s = QSlider(Qt.Horizontal)
            s.setRange(0, 150)
            s.setValue(int(self.player.gains.get(name, 1.0) * 100))
            s.valueChanged.connect(lambda v, n=name: self._gain_changed(n, v))
            self.mixer_grid.addWidget(s, row, 4)

            pct = QLabel(f"{s.value()}%")
            pct.setProperty("kind", "dim")
            pct.setFixedWidth(44)
            self.mixer_grid.addWidget(pct, row, 5)
            self.sliders[name], self.gain_labels[name] = s, pct
        self.mixer_grid.setColumnStretch(4, 1)
        self.mixer_box.setEnabled(True)

    # ------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, ev):
        urls = ev.mimeData().urls()
        if urls and urls[0].toLocalFile().lower().endswith(AUDIO_EXTS):
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        path = ev.mimeData().urls()[0].toLocalFile()
        self._load_path(path)

    # ------------------------------------------------------------ ações
    def open_file(self):
        last = self.settings.value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir música", last, f"Áudio (*{' *'.join(AUDIO_EXTS)})"
        )
        if path:
            self._load_path(path)

    def _load_path(self, path: str):
        self.audio_path = path
        self.settings.setValue("last_dir", str(Path(path).parent))
        self.lbl_file.setText(Path(path).name)
        self.btn_separate.setEnabled(True)
        cached = separator.cached_stems(path, CACHE_ROOT)
        if cached:
            self.statusBar().showMessage("Esta música já foi separada antes — o cache será usado.", 6000)
        self.run_bpm_detect()

    def _cache_dir(self) -> Path:
        return CACHE_ROOT / separator.file_key(self.audio_path)

    def run_bpm_detect(self):
        if not self.audio_path:
            return
        self.statusBar().showMessage("Analisando andamento e batidas…")
        self.bpm_worker = BpmWorker(self.audio_path, self._cache_dir())
        self.bpm_worker.done.connect(self._bpm_done)
        self.bpm_worker.failed.connect(lambda e: self.statusBar().showMessage(f"Falha na análise: {e}"))
        self.bpm_worker.start()

    @staticmethod
    def _fmt_t(s: float) -> str:
        return f"{int(s) // 60}:{int(s) % 60:02d}"

    def _bpm_label_text(self) -> str:
        bpm = self.original_bpm or 0.0
        if len(self.tempo_segments) > 1:
            return f"BPM: {bpm:.1f} (varia em {len(self.tempo_segments)} trechos)"
        return f"BPM: {bpm:.1f}"

    def _bpm_done(self, bpm: float, beats, segments):
        self.original_bpm = bpm
        self.beat_times = beats
        self.tempo_segments = list(segments or [])
        self.lbl_bpm.setText(self._bpm_label_text())
        self.spin_bpm.setValue(bpm)
        self.btn_half.setEnabled(True)
        self.btn_double.setEnabled(True)
        if len(self.tempo_segments) > 1:
            det = "\n".join(f"{self._fmt_t(s)}–{self._fmt_t(e)}  ≈ {b:.0f} BPM"
                            for s, e, b in self.tempo_segments)
            self.lbl_bpm.setToolTip("Trechos de andamento:\n" + det)
        else:
            self.lbl_bpm.setToolTip("")
        self.statusBar().showMessage(f"BPM detectado: {bpm:.1f}", 5000)
        if self.player.stems and "click" not in self.player.stems:
            self._add_click_track(self.current_rate)
            self._rebuild_mixer()

    def run_separation(self):
        if not self.audio_path:
            return
        self.btn_separate.setEnabled(False)
        self.progress.setValue(0)
        self.progress.show()
        self.statusBar().showMessage("Separando instrumentos…")
        self.sep_worker = SeparationWorker(self.audio_path, str(CACHE_ROOT))
        self.sep_worker.progress.connect(lambda s: self.statusBar().showMessage(s[:120]))
        self.sep_worker.percent.connect(self.progress.setValue)
        self.sep_worker.done.connect(self._separation_done)
        self.sep_worker.failed.connect(self._separation_failed)
        self.sep_worker.start()

    def _separation_done(self, stems: dict):
        self.player.load_stems({k: Path(v) for k, v in stems.items()})
        guitar = self.player.stems.get("guitar")
        if guitar is not None:
            self.statusBar().showMessage("Analisando quantidade de guitarras…")
            self.guitar_worker = GuitarWorker(guitar.copy())
            self.guitar_worker.done.connect(self._guitars_done)
            self.guitar_worker.failed.connect(lambda e: self._finish_load())
            self.guitar_worker.start()
        else:
            self._finish_load()

    def _guitars_done(self, parts: dict):
        if parts:
            self.player.split_stem("guitar", parts)
            self.lbl_guitars.setText(
                f"🎸 {len(parts)} guitarras identificadas pela imagem estéreo: " + ", ".join(parts.keys())
            )
        else:
            self.lbl_guitars.setText("🎸 1 guitarra identificada (ou imagem estéreo única).")
        self._finish_load()

    def _finish_load(self):
        self.progress.hide()
        self.original_arrays = {k: v.copy() for k, v in self.player.stems.items() if k != "click"}
        self.original_sr = self.player.sr
        self.current_rate = 1.0
        self._add_click_track(1.0)
        self._rebuild_mixer()
        self._refresh_wave()
        for b in (self.btn_play, self.btn_stop, self.btn_export, self.btn_export_stems,
                  self.btn_apply_bpm, self.btn_reset_bpm, self.btn_separate, self.btn_loop,
                  self.btn_clear_loop):
            b.setEnabled(True)
        self.statusBar().showMessage("Pronto! Arraste na forma de onda para criar um loop A–B.", 8000)

    def _separation_failed(self, err: str):
        self.progress.hide()
        self.btn_separate.setEnabled(True)
        QMessageBox.critical(self, "Erro na separação", err)

    def _refresh_wave(self):
        audio = {k: v for k, v in self.player.stems.items() if k != "click"}
        self.wave.set_peaks(compute_peaks(audio, self.player.frames))
        self.wave.set_loop(self.player.loop_fractions(), self.player.loop_enabled)

    # ------------------------------------------------------------ metrônomo
    def _add_click_track(self, rate: float):
        if self.beat_times is None or not self.player.stems:
            return
        times = self.beat_times / rate
        click = metronome.render_click_track(times, self.player.frames, self.player.sr)
        gain = self.player.gains.get("click", 1.0)
        mute = self.player.mutes.get("click", True)
        self.player.add_stem("click", click, gain=gain, mute=mute)

    def _octave(self, factor: float):
        if self.beat_times is None or not self.original_bpm:
            return
        if factor == 2.0:
            self.beat_times = metronome.double_beats(self.beat_times)
        else:
            if len(self.beat_times) < 4:
                return
            self.beat_times = metronome.halve_beats(self.beat_times)
        self.original_bpm = round(self.original_bpm * factor, 1)
        self.spin_bpm.setValue(self.spin_bpm.value() * factor)
        self.tempo_segments = [(s, e, round(b * factor, 1)) for s, e, b in self.tempo_segments]
        self.lbl_bpm.setText(self._bpm_label_text())
        self._add_click_track(self.current_rate)
        self.statusBar().showMessage(f"Metrônomo ajustado para {self.original_bpm:.1f} BPM.", 4000)

    def _gain_changed(self, name: str, value: int):
        self.player.set_gain(name, value / 100.0)
        if name in self.gain_labels:
            self.gain_labels[name].setText(f"{value}%")

    # ------------------------------------------------------------ andamento
    def _update_pct(self):
        if self.original_bpm:
            pct = self.spin_bpm.value() / self.original_bpm * 100
            self.lbl_pct.setText(f"({pct:.0f}% da velocidade)")

    def apply_bpm(self):
        if not self.original_arrays or not self.original_bpm:
            QMessageBox.information(self, "Andamento", "Separe os instrumentos primeiro.")
            return
        rate = self.spin_bpm.value() / self.original_bpm
        self.pending_rate = rate
        self.statusBar().showMessage(f"Aplicando time-stretch ({rate * 100:.0f}%)…")
        self.btn_apply_bpm.setEnabled(False)
        self.progress.setValue(0)
        self.progress.show()
        self.stretch_worker = StretchWorker(dict(self.original_arrays), self.original_sr, rate)
        self.stretch_worker.progress.connect(
            lambda d, t, n: (self.progress.setValue(int(d / t * 100)),
                             self.statusBar().showMessage(f"Esticando: {STEM_LABELS.get(n, n)} ({d}/{t})"))
        )
        self.stretch_worker.done.connect(self._stretch_done)
        self.stretch_worker.failed.connect(self._stretch_failed)
        self.stretch_worker.start()

    def _stretch_done(self, arrays: dict, sr: int):
        gains, mutes, solos = dict(self.player.gains), dict(self.player.mutes), dict(self.player.solos)
        self.player.replace_arrays(arrays, sr)
        self.player.gains.update(gains)
        self.player.mutes.update(mutes)
        self.player.solos.update(solos)
        self.current_rate = self.pending_rate
        self._add_click_track(self.current_rate)
        self._rebuild_mixer()
        self._refresh_wave()
        self.progress.hide()
        self.btn_apply_bpm.setEnabled(True)
        self.statusBar().showMessage("Novo andamento aplicado!", 5000)

    def _stretch_failed(self, err: str):
        self.progress.hide()
        self.btn_apply_bpm.setEnabled(True)
        QMessageBox.critical(self, "Erro no time-stretch", err)

    def reset_bpm(self):
        if not self.original_arrays:
            return
        gains, mutes, solos = dict(self.player.gains), dict(self.player.mutes), dict(self.player.solos)
        self.player.replace_arrays({k: v.copy() for k, v in self.original_arrays.items()}, self.original_sr)
        self.player.gains.update(gains)
        self.player.mutes.update(mutes)
        self.player.solos.update(solos)
        self.current_rate = 1.0
        self._add_click_track(1.0)
        self._rebuild_mixer()
        self._refresh_wave()
        if self.original_bpm:
            self.spin_bpm.setValue(self.original_bpm)
        self.statusBar().showMessage("Andamento original restaurado.", 4000)

    # ------------------------------------------------------------ transporte
    def toggle_play(self):
        if not self.player.stems:
            return
        if self.player.playing:
            self.player.pause()
            self.btn_play.setText("▶")
        else:
            self.player.play()
            self.btn_play.setText("⏸")

    def stop(self):
        self.player.stop()
        self.btn_play.setText("▶")

    def _seek(self, frac: float):
        self.player.seek_fraction(frac)

    def _nudge(self, seconds: float):
        dur = self.player.duration_seconds()
        if dur:
            self.player.seek_fraction(self.player.position_fraction() + seconds / dur)

    def _region_selected(self, a: float, b: float):
        self.player.set_loop(a, b)
        self.btn_loop.setChecked(True)
        self.wave.set_loop(self.player.loop_fractions(), True)

    def _loop_toggled(self, on: bool):
        self.player.set_loop_enabled(on)
        self.wave.set_loop(self.player.loop_fractions(), self.player.loop_enabled)
        if on and not self.player.loop:
            self.statusBar().showMessage("Arraste na forma de onda para selecionar o trecho do loop.", 5000)

    def _clear_loop(self):
        self.player.clear_loop()
        self.btn_loop.setChecked(False)
        self.wave.set_loop(None, False)

    # ------------------------------------------------------------ exportação
    def export_mix(self):
        path, _ = QFileDialog.getSaveFileName(self, "Exportar mixagem", "mixagem.wav", "WAV (*.wav)")
        if not path:
            return
        sf.write(path, self.player.render_mix(), self.player.sr)
        self.statusBar().showMessage(f"Exportado: {path}", 6000)

    def export_stems(self):
        folder = QFileDialog.getExistingDirectory(self, "Pasta para as faixas")
        if not folder:
            return
        for name, data in self.player.stems.items():
            label = STEM_LABELS.get(name, name).replace(" ", "_").replace("(", "").replace(")", "")
            sf.write(str(Path(folder) / f"{label}.wav"), data, self.player.sr)
        self.statusBar().showMessage(f"{len(self.player.stems)} faixas exportadas em {folder}", 6000)

    # ------------------------------------------------------------ relógio
    def _tick(self):
        frac = self.player.position_fraction()
        self.wave.set_playhead(frac)
        dur = self.player.duration_seconds()
        cur = frac * dur
        self.lbl_time.setText(
            f"{int(cur) // 60}:{int(cur) % 60:02d}.{int(cur * 10) % 10} / "
            f"{int(dur) // 60}:{int(dur) % 60:02d}.{int(dur * 10) % 10}"
        )
        if self.player.playing and not self.player.loop_enabled and self.player.pos >= self.player.frames:
            self.stop()

    def closeEvent(self, event):
        self.player.stop()
        super().closeEvent(event)
