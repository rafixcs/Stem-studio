"""Metrônomo sincronizado com a música, com suporte a mudanças de andamento.

Pipeline:
1. Tempograma de autocorrelação do envelope de onsets;
2. Fronteiras de trecho por NOVIDADE do tempograma (mudança sustentada do
   perfil de periodicidade), robusta a aliases de oitava/tercina;
3. Em cada trecho, tempos candidatos = picos do tempograma (± oitavas);
   cada candidato roda o beat tracker e é avaliado pela força dos onsets
   nos beats previstos (média alta + poucos beats "fracos");
4. Correções por trecho: se beats alternados são muito mais fracos, o
   tracker travou numa subdivisão -> divide por 2; se há ataques fortes nos
   pontos médios, travou na metade -> dobra;
5. Costura dos beats de todos os trechos com deduplicação nas fronteiras.
"""
from __future__ import annotations

import numpy as np

HOP = 512


def detect(path: str) -> tuple[float, np.ndarray, list[tuple[float, float, float]]]:
    """Retorna (bpm_principal, beat_times, trechos[(início_s, fim_s, bpm)])."""
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    frame_dur = HOP / sr
    env_n = oenv / (oenv.max() + 1e-9)
    env_t = librosa.times_like(oenv, sr=sr, hop_length=HOP)

    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=HOP, win_length=512)
    freqs = librosa.tempo_frequencies(tg.shape[0], sr=sr, hop_length=HOP)

    bounds = _novelty_bounds(tg, frame_dur)

    all_beats: list[np.ndarray] = []
    seg_info: list[tuple[float, float, float]] = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if (e - s) * frame_dur < 3.0 or oenv[s:e].sum() <= 0:
            continue
        res = _track_segment(s, e, oenv, tg, freqs, env_t, env_n, sr, frame_dur)
        if res is None:
            continue
        bt, bpm_seg = res
        seg_info.append((s * frame_dur, e * frame_dur, round(bpm_seg, 1)))
        all_beats.append(bt)

    if not all_beats:
        return 120.0, np.array([], dtype=np.float64), []

    beats = np.sort(np.concatenate(all_beats))
    med = float(np.median(np.diff(beats))) if len(beats) > 2 else 0.5
    keep = [beats[0]]
    for b in beats[1:]:
        if b - keep[-1] >= 0.5 * med:
            keep.append(b)
    beats = np.asarray(keep)

    merged: list[tuple[float, float, float]] = []
    for s0, e0, b0 in seg_info:
        if merged and abs(merged[-1][2] - b0) < 3.0:
            merged[-1] = (merged[-1][0], e0, merged[-1][2])
        else:
            merged.append((s0, e0, b0))

    bpm_main = round(60.0 / float(np.median(np.diff(beats))), 1) if len(beats) > 2 else merged[0][2]
    return bpm_main, beats, merged


def _novelty_bounds(tg: np.ndarray, frame_dur: float) -> list[int]:
    """Fronteiras onde o perfil de periodicidade muda de forma sustentada."""
    from scipy.signal import find_peaks, medfilt

    n = tg.shape[1]
    tgn = tg / (np.linalg.norm(tg, axis=0, keepdims=True) + 1e-9)
    w = max(2, int(4.0 / frame_dur))
    nov = np.zeros(n)
    for i in range(w, n - w):
        a = tgn[:, i - w : i].mean(axis=1)
        b = tgn[:, i : i + w].mean(axis=1)
        nov[i] = 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    if nov.max() <= 1e-6:
        return [0, n]
    nov = medfilt(nov, kernel_size=int(1.0 / frame_dur) | 1)
    pk, _ = find_peaks(
        nov, height=0.25 * nov.max(), prominence=0.2 * nov.max(),
        distance=max(1, int(8.0 / frame_dur)),
    )
    bounds = [0] + list(int(p) for p in pk) + [n]
    # funde segmentos menores que 8 s no vizinho (artefatos de borda)
    min_f = int(8.0 / frame_dur)
    out = [bounds[0]]
    for b in bounds[1:-1]:
        if b - out[-1] >= min_f and n - b >= min_f:
            out.append(b)
    out.append(n)
    return out


def _track_segment(s, e, oenv, tg, freqs, env_t, env_n, sr, frame_dur):
    """Escolhe o tempo do trecho por candidatos e devolve (beat_times, bpm)."""
    import librosa
    from scipy.signal import find_peaks

    agg = tg[:, s:e].mean(axis=1)
    valid = (freqs >= 60) & (freqs <= 240) & np.isfinite(freqs)
    p, _ = find_peaks(agg[valid], height=0.2 * (agg[valid].max() or 1))
    cands = set(np.round(freqs[valid][p], 1)) or {120.0}
    for c in list(cands):
        for m in (0.5, 2.0):
            if 60 <= c * m <= 240:
                cands.add(round(c * m, 1))

    t0 = s * frame_dur
    results: dict[int, tuple[np.ndarray, float, float]] = {}
    for c in sorted(cands):
        _, bt = librosa.beat.beat_track(
            onset_envelope=oenv[s:e], sr=sr, hop_length=HOP,
            start_bpm=float(c), tightness=400, units="time", trim=False,
        )
        bt = np.asarray(bt, dtype=np.float64) + t0
        if len(bt) < 4:
            continue
        bpm = 60.0 / float(np.median(np.diff(bt)))
        key = int(round(bpm))
        if key in results:
            continue
        st = np.interp(bt, env_t, env_n)
        mean_s = float(st.mean())
        weak = float(np.mean(st < 0.3 * np.percentile(st, 90)))
        results[key] = (bt, mean_s, weak)

    if not results:
        return None

    # Beats devem cair consistentemente em ataques: poucos beats fracos
    strong = {k: v for k, v in results.items() if v[2] <= 0.18}
    pool = strong or results
    bt, _, _ = max(pool.values(), key=lambda v: v[1])

    bt = _fix_octave(bt, env_t, env_n)
    bpm = 60.0 / float(np.median(np.diff(bt)))
    return bt, bpm


def _fix_octave(bt: np.ndarray, env_t, env_n) -> np.ndarray:
    """Dobra se houver ataques fortes nos pontos médios (meia velocidade).
    (Não dividimos automaticamente: a alternância bumbo/caixa nos beats reais
    se confunde com subdivisão — divisão fica nos botões manuais ÷2/x2.)"""
    # x2: travou na metade?
    while len(bt) > 8:
        bpm = 60.0 / float(np.median(np.diff(bt)))
        if bpm >= 120 or bpm * 2 > 250:
            break
        st_b = float(np.median(np.interp(bt, env_t, env_n)))
        mids = (bt[:-1] + bt[1:]) / 2
        st_m = float(np.median(np.interp(mids, env_t, env_n)))
        if st_b > 0 and st_m > 0.40 * st_b:
            bt = double_beats(bt)
        else:
            break
    return bt


def double_beats(beat_times: np.ndarray) -> np.ndarray:
    """Insere um beat no ponto médio entre cada par consecutivo (tempo x2)."""
    if len(beat_times) < 2:
        return beat_times
    mids = (beat_times[:-1] + beat_times[1:]) / 2
    out = np.empty(len(beat_times) + len(mids), dtype=np.float64)
    out[0::2] = beat_times
    out[1::2] = mids
    return out


def halve_beats(beat_times: np.ndarray) -> np.ndarray:
    """Mantém beats alternados (tempo /2)."""
    return beat_times[::2]


def _click(sr: int, freq: float = 1000.0, dur: float = 0.045, amp: float = 0.85) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t) * np.exp(-t * 70.0)).astype(np.float32)


def render_click_track(beat_times: np.ndarray, n_frames: int, sr: int) -> np.ndarray:
    """Gera uma faixa estéreo (n_frames, 2) com um click em cada beat."""
    mono = np.zeros(n_frames, dtype=np.float32)
    c = _click(sr)
    for t in beat_times:
        i = int(round(t * sr))
        if i >= n_frames:
            break
        seg = min(len(c), n_frames - i)
        mono[i : i + seg] += c[:seg]
    np.clip(mono, -1.0, 1.0, out=mono)
    return np.stack([mono, mono], axis=1)
