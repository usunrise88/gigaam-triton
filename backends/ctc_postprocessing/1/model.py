"""Triton python backend: CTC greedy collapse -> text, tokens, scores.

Takes the graph's per-frame argmax rather than the log-prob tensor. The argmax
is done on the GPU inside the exported model precisely so this step stays cheap:
for a 20 s bucket the log-probs are frames x vocab, and moving them into Python
just to take a max would dominate the latency budget of spec §9.2.

Full log-probs are still exposed to the client -- they leave the graph as their
own output and are mapped straight to the ensemble output, never passing through
here (spec §7.2).

    in   token_ids       INT32 [-1]      per-frame argmax
         token_logprobs  FP32  [-1]      log-prob of that argmax
         encoded_lengths INT32 [1]       valid frames
    out  text            BYTES [1]
         tokens          INT32 [-1]      padded to frame count
         token_scores    FP32  [-1]      padded to frame count
         tokens_len      INT32 [1]       valid entries in the two above
         logprobs_len    INT32 [1]       valid frames, mirrors encoded_lengths
"""

import json
import os
import sys

import numpy as np
import triton_python_backend_utils as pb_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import ctc  # noqa: E402
from common.tokenizer import Tokenizer  # noqa: E402


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.join(args["model_repository"], args["model_version"])

        try:
            self.tokenizer = Tokenizer.from_dir(model_dir)
        except FileNotFoundError as exc:
            raise pb_utils.TritonModelException(
                f"{exc}. Tokenizer artefacts come from scripts/export_onnx.py -- "
                "regenerate the model repository."
            )

        self.blank_id = self.tokenizer.blank_id

        meta_path = os.path.join(model_dir, "meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                responses.append(self._handle(request))
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"ctc postprocessing failed: {exc}")
                    )
                )

        return responses

    def _handle(self, request):
        token_ids = pb_utils.get_input_tensor_by_name(
            request, "token_ids"
        ).as_numpy().reshape(-1)
        frame_scores = pb_utils.get_input_tensor_by_name(
            request, "token_logprobs"
        ).as_numpy().reshape(-1)
        n_frames = int(
            pb_utils.get_input_tensor_by_name(request, "encoded_lengths")
            .as_numpy()
            .reshape(-1)[0]
        )

        text, tokens, scores = ctc.decode(
            token_ids, frame_scores, n_frames, self.blank_id, self.tokenizer
        )

        # Padded to the frame count so every response in a batch keeps the same
        # shape; tokens_len says how much of it is real.
        width = int(token_ids.shape[0])
        tokens_out = np.full(width, self.blank_id, dtype=np.int32)
        scores_out = np.zeros(width, dtype=np.float32)
        if tokens:
            tokens_out[: len(tokens)] = np.asarray(tokens, dtype=np.int32)
            scores_out[: len(scores)] = np.asarray(scores, dtype=np.float32)

        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "text", np.array([[text.encode("utf-8")]], dtype=object)
                ),
                pb_utils.Tensor("tokens", tokens_out[None, :]),
                pb_utils.Tensor("token_scores", scores_out[None, :]),
                pb_utils.Tensor(
                    "tokens_len", np.array([[len(tokens)]], dtype=np.int32)
                ),
                pb_utils.Tensor(
                    "logprobs_len", np.array([[n_frames]], dtype=np.int32)
                ),
            ]
        )
