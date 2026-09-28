"""Audio loading, resampling and segmentation shared by the engines."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

GIGAAM_SR = 16000
TONE_SR = 8000


def load(path: Path) -> tuple[np.ndarray, int, dict]:
    """Mono float32 in [-1, 1], plus what had to be done to get there."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    info = {"source_sr": sr, "channels": audio.shape[1]}

    if audio.shape[1] > 1:
        left, right = audio[:, 0], audio[:, 1]
        corr = float(np.corrcoef(left, right)[0, 1]) if len(left) > 1 else 1.0
        info["channel_correlation"] = round(corr, 3)
        # Two genuinely different channels means two speakers, and mixing them
        # creates overlap the model never saw. Say so rather than hide it.
        info["two_real_channels"] = corr < 0.99
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    return np.ascontiguousarray(audio, dtype=np.float32), sr, info


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Polyphase, once per segment -- not linear interpolation and not per frame
    (spec §11.5)."""
    if src_sr == dst_sr:
        return audio
    g = gcd(int(src_sr), int(dst_sr))
    return np.ascontiguousarray(
        resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
    )


def segment(
    audio: np.ndarray,
    sr: int,
    max_s: float = 20.0,
    min_silence_s: float = 0.30,
    frame_ms: int = 20,
) -> list[tuple[int, int]]:
    """Split at silence into chunks no longer than max_s.

    GigaAM is offline and full-context with a 25 s ceiling (spec §11.1), so long
    recordings have to be cut somewhere. Cutting at the quietest available point
    keeps words intact; a hard cut is the fallback when speech simply runs longer
    than the limit.

    Energy-based rather than a neural VAD on purpose: GigaAM's own longform path
    pulls in pyannote and transformers, which is a lot of dependency for a
    segmentation step whose job here is only to respect a length limit.
    """
    n = audio.shape[0]
    max_len = int(max_s * sr)
    if n <= max_len:
        return [(0, n)]

    hop = int(frame_ms * sr / 1000)
    frames = n // hop
    energy = np.array(
        [float(np.sqrt(np.mean(audio[i * hop : (i + 1) * hop] ** 2))) for i in range(frames)]
    )
    # Threshold relative to the recording, so it survives level differences.
    speech_level = np.percentile(energy, 90)
    threshold = max(speech_level * 0.08, float(np.percentile(energy, 10)) * 1.5, 1e-4)
    silent = energy < threshold

    # Midpoints of silences long enough to be a real pause.
    candidates: list[int] = []
    run_start = None
    min_run = max(1, int(min_silence_s * 1000 / frame_ms))
    for i, is_silent in enumerate(silent):
        if is_silent and run_start is None:
            run_start = i
        elif not is_silent and run_start is not None:
            if i - run_start >= min_run:
                candidates.append((run_start + i) // 2 * hop)
            run_start = None
    if run_start is not None and frames - run_start >= min_run:
        candidates.append((run_start + frames) // 2 * hop)

    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        limit = start + max_len
        if limit >= n:
            spans.append((start, n))
            break
        usable = [c for c in candidates if start + sr * 1.0 < c <= limit]
        cut = max(usable) if usable else limit      # hard cut only if no pause fits
        spans.append((start, cut))
        start = cut

    return [(a, b) for a, b in spans if b > a]
