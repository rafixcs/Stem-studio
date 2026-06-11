"""Separação de stems usando Demucs (htdemucs_6s), com cache por arquivo e
progresso determinado (percentual extraído da saída do Demucs)."""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

STEM_NAMES = ["vocals", "drums", "bass", "guitar", "piano", "other"]
MODEL = "htdemucs_6s"
_PCT = re.compile(r"(\d{1,3})%")


class SeparationError(RuntimeError):
    pass


def file_key(audio_path: str | Path) -> str:
    """Hash rápido do arquivo (primeiro 1 MB + tamanho) para chavear o cache."""
    p = Path(audio_path)
    h = hashlib.md5()
    h.update(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


def cached_stems(audio_path: str | Path, cache_root: str | Path) -> dict[str, Path] | None:
    """Retorna os stems do cache se a música já foi separada antes."""
    stem_dir = Path(cache_root) / file_key(audio_path) / MODEL / Path(audio_path).stem
    result = {n: stem_dir / f"{n}.wav" for n in STEM_NAMES}
    return result if all(p.exists() for p in result.values()) else None


def separate(audio_path: str, cache_root: str, progress_cb=None,
             percent_cb=None, device: str | None = None) -> dict[str, Path]:
    """Roda o Demucs (ou usa o cache) e retorna {nome_do_stem: caminho_wav}."""
    cached = cached_stems(audio_path, cache_root)
    if cached:
        if progress_cb:
            progress_cb("Separação encontrada no cache — reutilizando.")
        if percent_cb:
            percent_cb(100)
        return cached

    audio_path = Path(audio_path)
    out_dir = Path(cache_root) / file_key(audio_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "demucs", "-n", MODEL, "-o", str(out_dir)]
    if device:
        cmd += ["-d", device]
    cmd.append(str(audio_path))

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        lines.append(line)
        if percent_cb:
            m = _PCT.findall(line)
            if m:
                percent_cb(min(100, int(m[-1])))
        if progress_cb:
            progress_cb(line)
    proc.wait()
    if proc.returncode != 0:
        tail = "\n".join(lines[-10:])
        raise SeparationError(f"Demucs falhou (código {proc.returncode}):\n{tail}")

    stem_dir = out_dir / MODEL / audio_path.stem
    result: dict[str, Path] = {}
    for name in STEM_NAMES:
        wav = stem_dir / f"{name}.wav"
        if not wav.exists():
            raise SeparationError(f"Stem não encontrado: {wav}")
        result[name] = wav
    return result
