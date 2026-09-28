"""Torch-free log-mel front end.

The runtime image ships without torch (spec §4.2), but GigaAM's feature
extractor is a ``torchaudio.transforms.MelSpectrogram`` followed by
``log(clamp(x, 1e-9, 1e9))``. Reimplementing the mel filterbank formula by hand
is how you get a front end that is subtly wrong and shows up months later as
unexplained WER -- so we do not reimplement it. The builder exports the *actual*
filterbank matrix and window tensors that torchaudio built (see
``scripts/export_filterbank.py``) and this module only replays the arithmetic
around them.

Everything here must stay importable with numpy alone.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

LOG_CLAMP_MIN = 1e-9
LOG_CLAMP_MAX = 1e9


class LogMel:
    """Replays torchaudio's MelSpectrogram + log scaling on numpy arrays.

    Parameters come from ``filterbank.npz`` written by the builder, so they are
    whatever the model's own ``cfg.preprocessor`` asked for rather than values
    guessed here.
    """

    def __init__(
        self,
        filterbank: np.ndarray,
        window: np.ndarray,
        n_fft: int,
        hop_length: int,
        center: bool = True,
        pad_mode: str = "reflect",
    ) -> None:
        if window.shape[0] != n_fft:
            # torchaudio centres a shorter win_length inside n_fft; the builder
            # exports the already-padded window so this stays a pure replay.
            raise ValueError(
                f"window must be pre-padded to n_fft ({n_fft}), got {window.shape[0]}"
            )
        if filterbank.shape[0] != n_fft // 2 + 1:
            raise ValueError(
                f"filterbank first axis must be n_freq ({n_fft // 2 + 1}), "
                f"got {filterbank.shape[0]}"
            )

        # float64 internally: the reference is torch's float32, and computing the
        # STFT in float64 keeps our own rounding well below the tolerance we are
        # being compared against instead of adding to it.
        self.filterbank = np.asarray(filterbank, dtype=np.float64)
        self.window = np.asarray(window, dtype=np.float64)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.center = bool(center)
        self.pad_mode = pad_mode
        self.n_mels = int(filterbank.shape[1])

    @classmethod
    def from_npz(cls, path: str) -> "LogMel":
        data = np.load(path)
        return cls(
            filterbank=data["filterbank"],
            window=data["window"],
            n_fft=int(data["n_fft"]),
            hop_length=int(data["hop_length"]),
            center=bool(data["center"]),
            pad_mode=str(data["pad_mode"]) if "pad_mode" in data else "reflect",
        )

    def out_len(self, n_samples: np.ndarray) -> np.ndarray:
        """Frame count for a given sample count.

        Mirrors ``gigaam.preprocess.FeatureExtractor.out_len``: with centring the
        signal is padded by n_fft//2 on both sides, so the count depends only on
        the hop.
        """
        n_samples = np.asarray(n_samples, dtype=np.int64)
        if self.center:
            return n_samples // self.hop_length + 1
        win = self.window.shape[0]
        return (n_samples - win) // self.hop_length + 1

    def __call__(
        self, audio: np.ndarray, dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """audio: (batch, samples) or (samples,) -> (batch, n_mels, frames)."""
        single = audio.ndim == 1
        if single:
            audio = audio[None, :]
        if audio.ndim != 2:
            raise ValueError(f"expected (batch, samples), got shape {audio.shape}")

        x = np.asarray(audio, dtype=np.float64)

        if self.center:
            pad = self.n_fft // 2
            x = np.pad(x, ((0, 0), (pad, pad)), mode=self.pad_mode)

        n_frames = 1 + (x.shape[1] - self.n_fft) // self.hop_length
        if n_frames < 1:
            raise ValueError(
                f"audio too short for n_fft={self.n_fft}: {audio.shape[1]} samples"
            )

        # (batch, frames, n_fft) view over the padded signal, no copy until *=.
        frames = np.lib.stride_tricks.sliding_window_view(x, self.n_fft, axis=1)
        frames = frames[:, :: self.hop_length, :][:, :n_frames, :]
        frames = frames * self.window

        spec = np.fft.rfft(frames, n=self.n_fft, axis=-1)
        power = spec.real**2 + spec.imag**2          # power=2.0, as torchaudio defaults
        mel = power @ self.filterbank                # (batch, frames, n_mels)
        mel = np.clip(mel, LOG_CLAMP_MIN, LOG_CLAMP_MAX)
        out = np.log(mel).transpose(0, 2, 1)         # (batch, n_mels, frames)

        out = out.astype(dtype)
        return out[0] if single else out


def load(path: str) -> Tuple[LogMel, dict]:
    """Convenience for the Triton backend: extractor plus the raw metadata."""
    data = np.load(path)
    meta = {k: data[k].tolist() for k in data.files if data[k].ndim == 0}
    return LogMel.from_npz(path), meta
