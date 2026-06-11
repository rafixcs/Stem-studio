"""Player com mixagem em tempo real: ganho/mute/solo por faixa, master e loop A-B."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


class StemPlayer:
    def __init__(self):
        self._lock = threading.Lock()
        self.stems: dict[str, np.ndarray] = {}
        self.gains: dict[str, float] = {}
        self.mutes: dict[str, bool] = {}
        self.solos: dict[str, bool] = {}
        self.master: float = 1.0
        self.sr: int = 44100
        self.frames: int = 0
        self.pos: int = 0
        self.loop: tuple[int, int] | None = None
        self.loop_enabled: bool = False
        self._stream: sd.OutputStream | None = None
        self.playing = False

    # ---------- carregamento ----------
    def load_stems(self, paths: dict[str, Path]):
        self.stop()
        with self._lock:
            self.stems.clear()
            frames = None
            for name, p in paths.items():
                data, sr = sf.read(str(p), dtype="float32", always_2d=True)
                if data.shape[1] == 1:
                    data = np.repeat(data, 2, axis=1)
                self.sr = sr
                self.stems[name] = data
                frames = data.shape[0] if frames is None else min(frames, data.shape[0])
                self.gains.setdefault(name, 1.0)
                self.mutes.setdefault(name, False)
                self.solos.setdefault(name, False)
            self.frames = frames or 0
            self.pos = 0
            self.loop = None
            self.loop_enabled = False

    def replace_arrays(self, arrays: dict[str, np.ndarray], sr: int):
        self.stop()
        with self._lock:
            old_frames = self.frames or 1
            ratio = None
            self.stems = {k: v.astype(np.float32) for k, v in arrays.items()}
            self.sr = sr
            new_frames = min(v.shape[0] for v in self.stems.values()) if self.stems else 0
            if self.loop and new_frames:
                ratio = new_frames / old_frames
                a, b = self.loop
                self.loop = (int(a * ratio), int(b * ratio))
            self.frames = new_frames
            self.pos = 0
            for name in self.stems:
                self.gains.setdefault(name, 1.0)
                self.mutes.setdefault(name, False)
                self.solos.setdefault(name, False)

    def add_stem(self, name: str, data: np.ndarray, gain: float = 1.0, mute: bool = False):
        with self._lock:
            self.stems[name] = data.astype(np.float32)
            self.gains[name] = gain
            self.mutes[name] = mute
            self.solos.setdefault(name, False)
            self.frames = min(v.shape[0] for v in self.stems.values())

    def split_stem(self, name: str, parts: dict[str, np.ndarray]):
        with self._lock:
            if name not in self.stems:
                return
            base_gain = self.gains.pop(name, 1.0)
            self.mutes.pop(name, None)
            self.solos.pop(name, None)
            del self.stems[name]
            for k, v in parts.items():
                self.stems[k] = v.astype(np.float32)
                self.gains[k] = base_gain
                self.mutes[k] = False
                self.solos[k] = False
            self.frames = min(v.shape[0] for v in self.stems.values()) if self.stems else 0

    # ---------- controles ----------
    def set_gain(self, name: str, gain: float):
        self.gains[name] = float(gain)

    def set_mute(self, name: str, mute: bool):
        self.mutes[name] = bool(mute)

    def set_solo(self, name: str, solo: bool):
        self.solos[name] = bool(solo)

    def set_master(self, gain: float):
        self.master = float(gain)

    def set_loop(self, a_frac: float, b_frac: float):
        with self._lock:
            a = int(max(0.0, min(a_frac, b_frac)) * self.frames)
            b = int(min(1.0, max(a_frac, b_frac)) * self.frames)
            self.loop = (a, b) if b - a > 256 else None

    def clear_loop(self):
        with self._lock:
            self.loop = None
            self.loop_enabled = False

    def set_loop_enabled(self, enabled: bool):
        self.loop_enabled = bool(enabled) and self.loop is not None
        if self.loop_enabled and self.loop:
            with self._lock:
                a, b = self.loop
                if not (a <= self.pos < b):
                    self.pos = a

    def loop_fractions(self) -> tuple[float, float] | None:
        if self.loop and self.frames:
            return self.loop[0] / self.frames, self.loop[1] / self.frames
        return None

    def seek_fraction(self, frac: float):
        with self._lock:
            self.pos = int(max(0.0, min(1.0, frac)) * self.frames)

    def position_fraction(self) -> float:
        return self.pos / self.frames if self.frames else 0.0

    def duration_seconds(self) -> float:
        return self.frames / self.sr if self.sr else 0.0

    def _audible(self) -> list[str]:
        solo_set = [n for n, s in self.solos.items() if s]
        if solo_set:
            return solo_set
        return [n for n in self.stems if not self.mutes.get(n)]

    # ---------- reprodução ----------
    def play(self):
        if self.playing or not self.stems:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sr, channels=2, dtype="float32",
            callback=self._callback, blocksize=1024, latency="low",
        )
        self._stream.start()
        self.playing = True

    def pause(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.playing = False

    def stop(self):
        self.pause()
        with self._lock:
            self.pos = 0

    def _callback(self, outdata, nframes, time_info, status):
        mix = np.zeros((nframes, 2), dtype=np.float32)
        finished = False
        with self._lock:
            audible = self._audible()
            loop = self.loop if (self.loop_enabled and self.loop) else None
            filled = 0
            while filled < nframes:
                if loop and self.pos >= loop[1]:
                    self.pos = loop[0]
                limit = loop[1] if loop else self.frames
                n = min(nframes - filled, limit - self.pos)
                if n <= 0:
                    if loop:
                        self.pos = loop[0]
                        continue
                    finished = True
                    break
                s, e = self.pos, self.pos + n
                for name in audible:
                    g = self.gains.get(name, 1.0)
                    if g > 0:
                        mix[filled : filled + n] += self.stems[name][s:e] * g
                self.pos = e
                filled += n
        mix *= self.master
        np.clip(mix, -1.0, 1.0, out=mix)
        outdata[:] = mix
        if finished:
            raise sd.CallbackStop()

    # ---------- exportação ----------
    def render_mix(self) -> np.ndarray:
        with self._lock:
            audible = self._audible()
            mix = np.zeros((self.frames, 2), dtype=np.float32)
            for name in audible:
                mix += self.stems[name][: self.frames] * self.gains.get(name, 1.0)
        mix *= self.master
        peak = float(np.max(np.abs(mix))) or 1.0
        if peak > 1.0:
            mix /= peak
        return mix
