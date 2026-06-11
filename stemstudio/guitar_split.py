"""Detecção da quantidade de guitarras e separação por posição estéreo (pan).

O htdemucs_6s entrega um stem único de "guitar". Para dividi-lo em múltiplas
faixas, analisamos a imagem estéreo: em mixagens típicas de rock/metal, cada
guitarra ocupa uma posição de pan distinta (base dobrada esq./dir., solo ao
centro). Algoritmo:

1. STFT dos canais L e R;
2. índice de pan por bin tempo-frequência: (|R| - |L|) / (|L| + |R|);
3. histograma de pan ponderado por energia -> picos = guitarras distintas;
4. máscara gaussiana em torno de cada pico -> ISTFT -> uma faixa por guitarra.

Limitação: guitarras mixadas na MESMA posição estéreo saem juntas numa faixa.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, istft, stft

N_FFT = 4096
HOP = 1024
EPS = 1e-10


def _stft_stereo(y: np.ndarray):
    f, t, ZL = stft(y[:, 0], nperseg=N_FFT, noverlap=N_FFT - HOP)
    _, _, ZR = stft(y[:, 1], nperseg=N_FFT, noverlap=N_FFT - HOP)
    return ZL, ZR


def estimate_pan_centers(y: np.ndarray, max_sources: int = 4) -> list[float]:
    """Estima as posições de pan das guitarras. Retorna [] se o stem for ~silêncio."""
    if y.ndim != 2 or y.shape[1] < 2:
        return [0.0]  # mono: impossível dividir por pan
    if float(np.sqrt(np.mean(y**2))) < 1e-4:
        return []

    ZL, ZR = _stft_stereo(y)
    magL, magR = np.abs(ZL), np.abs(ZR)
    energy = magL**2 + magR**2
    pan = (magR - magL) / (magL + magR + EPS)

    # Considera apenas bins com energia relevante
    thresh = np.percentile(energy, 75)
    mask = energy > thresh
    if not mask.any():
        return [0.0]

    hist, edges = np.histogram(pan[mask], bins=121, range=(-1, 1), weights=energy[mask])
    centers_axis = (edges[:-1] + edges[1:]) / 2

    # Suavização gaussiana
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    hist_s = np.convolve(hist, k / k.sum(), mode="same")

    peaks, props = find_peaks(
        hist_s,
        height=0.15 * hist_s.max(),
        prominence=0.12 * hist_s.max(),
        distance=12,
    )
    if len(peaks) == 0:
        avg = float(np.average(pan[mask], weights=energy[mask]))
        return [avg]

    # Ordena por altura, funde picos a menos de 0.22 de distância (mantém o mais forte)
    order = np.argsort(props["peak_heights"])[::-1]
    sel: list[float] = []
    for idx in order:
        c = float(centers_axis[peaks[idx]])
        if all(abs(c - s) >= 0.22 for s in sel):
            sel.append(c)
        if len(sel) >= max_sources:
            break
    return sorted(sel)


def split_by_pan(y: np.ndarray, centers: list[float]) -> list[np.ndarray]:
    """Separa o áudio estéreo em len(centers) faixas via máscaras de pan."""
    n = y.shape[0]
    ZL, ZR = _stft_stereo(y)
    magL, magR = np.abs(ZL), np.abs(ZR)
    pan = (magR - magL) / (magL + magR + EPS)

    if len(centers) > 1:
        spacing = min(b - a for a, b in zip(centers, centers[1:]))
        sigma = float(np.clip(spacing / 2.5, 0.07, 0.25))
    else:
        sigma = 0.25

    weights = [np.exp(-0.5 * ((pan - c) / sigma) ** 2) for c in centers]
    total = sum(weights) + EPS

    out: list[np.ndarray] = []
    for w in weights:
        m = w / total
        _, xl = istft(ZL * m, nperseg=N_FFT, noverlap=N_FFT - HOP)
        _, xr = istft(ZR * m, nperseg=N_FFT, noverlap=N_FFT - HOP)
        track = np.stack([xl[:n], xr[:n]], axis=1).astype(np.float32)
        if track.shape[0] < n:
            track = np.pad(track, ((0, n - track.shape[0]), (0, 0)))
        out.append(track)
    return out


def pan_label(c: float) -> str:
    if c < -0.15:
        return "esquerda"
    if c > 0.15:
        return "direita"
    return "centro"
