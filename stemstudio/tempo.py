"""Detecção de BPM (librosa) e time-stretch sem alteração de pitch.

Prioriza o Rubber Band (qualidade superior, formant-preserving); se o binário
`rubberband` não estiver instalado no sistema, cai para o phase vocoder do librosa.
"""
from __future__ import annotations

import shutil

import numpy as np


def detect_bpm(path: str) -> float:
    """Detecta o andamento (BPM) de um arquivo de áudio."""
    import librosa

    y, sr = librosa.load(path, sr=None, mono=True, duration=120)
    # Separa a parte percussiva para uma estimativa mais estável
    y_perc = librosa.effects.percussive(y)
    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    # Normaliza para uma faixa musicalmente comum
    while tempo < 60:
        tempo *= 2
    while tempo > 200:
        tempo /= 2
    return round(tempo, 1)


def rubberband_available() -> bool:
    return shutil.which("rubberband") is not None or shutil.which("rubberband-r3") is not None


def stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Estica/comprime o áudio no tempo mantendo o pitch.

    y: array (frames,) ou (frames, canais), float32/float64.
    rate > 1.0 acelera; rate < 1.0 desacelera (rate = bpm_alvo / bpm_original).
    """
    if abs(rate - 1.0) < 1e-4:
        return y

    if rubberband_available():
        import pyrubberband as pyrb

        # Opções de alta qualidade do Rubber Band
        return pyrb.time_stretch(y, sr, rate, rbargs={"--fine": "", "-c": "6"}).astype(np.float32)

    # Fallback: phase vocoder do librosa (qualidade boa, inferior ao Rubber Band)
    import librosa

    if y.ndim == 1:
        return librosa.effects.time_stretch(y.astype(np.float32), rate=rate)
    chans = [librosa.effects.time_stretch(y[:, c].astype(np.float32), rate=rate) for c in range(y.shape[1])]
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans], axis=1)


def stretch_many(arrays: dict, sr: int, rate: float, progress_cb=None) -> dict:
    """Aplica o time-stretch em todas as faixas em paralelo (1 worker por stem,
    até 4 simultâneos — o Rubber Band roda como subprocesso, então paraleliza bem)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out: dict = {}
    done = 0
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(arrays)))) as ex:
        futures = {ex.submit(stretch, y, sr, rate): name for name, y in arrays.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            out[name] = fut.result()
            done += 1
            if progress_cb:
                progress_cb(done, len(arrays), name)
    return out
