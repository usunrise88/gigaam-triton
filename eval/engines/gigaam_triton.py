"""GigaAM through our Triton ensembles.

A thin client: pad each segment to its bucket, pick the matching ensemble by
name, concatenate. Everything else -- features, decoding, confidence -- is behind
the contract from spec §7.2, which is the point of that contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GigaAMTritonEngine:
    def __init__(self, url: str = "localhost:18001", manifest: str | dict | None = None,
                 label: str | None = None):
        import tritonclient.grpc as grpcclient

        self.client = grpcclient.InferenceServerClient(url=url)
        if isinstance(manifest, dict):
            self.manifest = manifest
        else:
            self.manifest = json.loads(Path(manifest).read_text())
        self.buckets = sorted(self.manifest["ensembles"], key=lambda e: e["pad_to_samples"])
        self.label = label or f"gigaam/{self.manifest['variant']}"

    @property
    def name(self) -> str:
        return self.label

    @property
    def max_segment_samples(self) -> int:
        return self.buckets[-1]["pad_to_samples"]

    def _bucket_for(self, n_samples: int) -> dict:
        for entry in self.buckets:
            if n_samples <= entry["pad_to_samples"]:
                return entry
        raise ValueError(
            f"{n_samples} samples exceeds the largest bucket "
            f"({self.buckets[-1]['pad_to_samples']}); segment first (spec §11.1)"
        )

    def transcribe_segment(self, audio_16k: np.ndarray) -> str:
        import tritonclient.grpc as grpcclient

        entry = self._bucket_for(audio_16k.shape[0])
        padded = np.zeros(entry["pad_to_samples"], dtype=np.float32)
        padded[: audio_16k.shape[0]] = np.clip(audio_16k, -1.0, 1.0)

        audio_in = grpcclient.InferInput("audio", [1, padded.shape[0]], "FP32")
        audio_in.set_data_from_numpy(padded[None, :])
        len_in = grpcclient.InferInput("audio_len", [1, 1], "INT32")
        len_in.set_data_from_numpy(np.array([[audio_16k.shape[0]]], dtype=np.int32))

        result = self.client.infer(
            entry["name"], [audio_in, len_in],
            outputs=[grpcclient.InferRequestedOutput("text")],
        )
        return result.as_numpy("text")[0][0].decode("utf-8").strip()

    def transcribe(self, audio_16k: np.ndarray, spans: list[tuple[int, int]]) -> str:
        parts = []
        for start, end in spans:
            text = self.transcribe_segment(audio_16k[start:end])
            if text:
                parts.append(text)
        return " ".join(parts)
