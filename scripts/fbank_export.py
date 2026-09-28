"""Export the mel filterbank and window that torchaudio actually built.

Runs in the builder (needs torch). The runtime replays these arrays with numpy
(``backends/common/logmel.py``) so it can stay torch-free.

We export the tensors rather than the parameters that produced them on purpose:
mel filterbank construction has enough knobs (htk vs slaney scale, norm, f_min,
f_max rounding) that a reimplementation agreeing on the common case can still
diverge on an edge one, and the failure is silent.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


def _as_padded_window(window: torch.Tensor, n_fft: int) -> np.ndarray:
    """torch.stft centres a shorter win_length inside n_fft; do it once here so
    the runtime never has to know about the distinction."""
    win_length = window.shape[0]
    if win_length == n_fft:
        return window.detach().cpu().numpy().astype(np.float64)
    if win_length > n_fft:
        raise ValueError(f"win_length {win_length} > n_fft {n_fft}")
    left = (n_fft - win_length) // 2
    padded = torch.zeros(n_fft, dtype=window.dtype)
    padded[left : left + win_length] = window
    return padded.detach().cpu().numpy().astype(np.float64)


def extract(feature_extractor: Any) -> Dict[str, np.ndarray]:
    """Pull the arrays out of a ``gigaam.preprocess.FeatureExtractor``.

    Raises if the transform is configured in a way our numpy replay does not
    reproduce -- better a build failure than a quietly different front end.
    """
    mel = feature_extractor.featurizer[0]     # torchaudio MelSpectrogram
    spec = mel.spectrogram                    # torchaudio Spectrogram

    if float(spec.power) != 2.0:
        raise ValueError(f"expected power=2.0, got {spec.power}")
    if bool(spec.normalized):
        raise ValueError("normalized spectrogram is not replayed by logmel.py")
    if int(spec.pad) != 0:
        raise ValueError(f"expected pad=0, got {spec.pad}")
    if not bool(spec.onesided):
        raise ValueError("two-sided spectrogram is not replayed by logmel.py")

    fb = mel.mel_scale.fb.detach().cpu().numpy().astype(np.float64)  # (n_freq, n_mels)
    n_fft = int(spec.n_fft)
    expected_freq = n_fft // 2 + 1
    if fb.shape[0] != expected_freq:
        raise ValueError(f"filterbank n_freq {fb.shape[0]} != {expected_freq}")

    return {
        "filterbank": fb,
        "window": _as_padded_window(spec.window, n_fft),
        "n_fft": np.array(n_fft),
        "hop_length": np.array(int(spec.hop_length)),
        "win_length": np.array(int(spec.win_length)),
        "center": np.array(bool(spec.center)),
        "pad_mode": np.array(str(spec.pad_mode)),
        "n_mels": np.array(int(fb.shape[1])),
        "sample_rate": np.array(int(mel.sample_rate)),
    }


def save(feature_extractor: Any, out_path: str) -> Dict[str, np.ndarray]:
    arrays = extract(feature_extractor)
    np.savez(out_path, **arrays)
    return arrays
