"""T-one through the running production Triton, decoded by the official client.

Spec §10.4 is explicit that the decoder must be T-one's own: splitting and CTC
decoding happen client-side in the ``tone`` package, and a reimplementation would
be measuring our reconstruction of T-one rather than T-one. So this mirrors what
SecondChain actually runs (``src/pipeline/asr/tone_triton.py``): the acoustic
model on Triton, ``StreamingLogprobSplitter`` and the decoder from the package.

The production config uses beam_search with KenLM, so that is what a comparison
has to face; greedy is measured alongside it to show what the LM is worth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FRAME_SIZE = 2400        # 300 ms at 8 kHz
STATE_SIZE = 219729
LOGPROB_FRAMES = 10
LOGPROB_CLASSES = 35


class ToneTritonEngine:
    def __init__(
        self,
        url: str = "localhost:17001",
        model: str = "streaming_acoustic",
        decoder: str = "beam_search",
        kenlm_path: str | None = None,
    ):
        import tritonclient.grpc as grpcclient
        from tone import BeamSearchCTCDecoder, GreedyCTCDecoder, StreamingLogprobSplitter

        self.client = grpcclient.InferenceServerClient(url=url)
        self.model = model
        self.decoder_type = decoder
        self.splitter_cls = StreamingLogprobSplitter

        if decoder == "beam_search":
            if kenlm_path and Path(kenlm_path).exists():
                self.decoder = BeamSearchCTCDecoder.from_local(kenlm_path)
            else:
                self.decoder = BeamSearchCTCDecoder.from_hugging_face()
        else:
            self.decoder = GreedyCTCDecoder()

    @property
    def name(self) -> str:
        return f"tone/{self.decoder_type}"

    def _logprobs(self, audio_8k_i32: np.ndarray) -> np.ndarray:
        import tritonclient.grpc as grpcclient

        state = np.zeros((STATE_SIZE,), dtype=np.float16)
        out = []
        offset, total = 0, audio_8k_i32.shape[0]
        while offset < total:
            frame = np.zeros((FRAME_SIZE,), dtype=np.int32)
            chunk = audio_8k_i32[offset : offset + FRAME_SIZE]
            frame[: chunk.shape[0]] = chunk

            signal = grpcclient.InferInput("signal", [1, FRAME_SIZE, 1], "INT32")
            signal.set_data_from_numpy(frame.reshape(1, FRAME_SIZE, 1))
            state_in = grpcclient.InferInput("state", [1, STATE_SIZE], "FP16")
            state_in.set_data_from_numpy(state.reshape(1, STATE_SIZE))

            result = self.client.infer(
                self.model, [signal, state_in],
                outputs=[grpcclient.InferRequestedOutput("logprobs"),
                         grpcclient.InferRequestedOutput("state_next")],
            )
            out.append(
                result.as_numpy("logprobs")
                .reshape(LOGPROB_FRAMES, LOGPROB_CLASSES)
                .astype(np.float32)
            )
            state = result.as_numpy("state_next").reshape(STATE_SIZE)
            offset += FRAME_SIZE

        if not out:
            return np.zeros((0, LOGPROB_CLASSES), dtype=np.float32)
        return np.concatenate(out, axis=0)

    def transcribe(self, audio_8k: np.ndarray) -> str:
        """audio_8k: mono float32 in [-1, 1] at 8 kHz."""
        pcm = np.clip(audio_8k, -1.0, 1.0)
        samples = (pcm * 32767.0).astype(np.int16).astype(np.int32)

        logprobs = self._logprobs(samples)
        phrases, _ = self.splitter_cls().forward(logprobs, None, is_last=True)

        texts = []
        for phrase in phrases:
            text = self.decoder.forward(phrase.logprobs).strip()
            if text:
                texts.append(text)
        return " ".join(texts)
