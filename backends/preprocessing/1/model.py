"""Triton python backend: raw audio -> log-mel features.

Torch-free by design (spec §4.2): replays the filterbank the builder exported
(``filterbank.npz``) with numpy. See ``common/logmel.py`` for why the arrays are
exported rather than recomputed.

Contract is the one from spec §7.2 and is identical for every variant and
runtime, so the provider never learns what is behind it:

    in   audio      FP32  [-1]   16 kHz mono, range [-1, 1]
         audio_len  INT32 [1]    true length before padding
    out  features   FP16|FP32 [n_mels, -1]
         feature_lengths INT32 [1]

Input validation is not decoration. GigaAM fed int16-scaled samples returns an
empty string with no error and no warning (spec §11.2) -- exactly the failure
mode that survives a smoke test and ruins a corpus run. So we fail loudly here.
"""

import json
import os
import sys

import numpy as np
import triton_python_backend_utils as pb_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.logmel import LogMel  # noqa: E402

# Float audio lives in [-1, 1]. A little headroom for clipping artefacts, then
# anything above this is not float audio -- almost always int16 samples that were
# reinterpreted rather than converted.
MAX_PLAUSIBLE_AMPLITUDE = 1.5


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.join(args["model_repository"], args["model_version"])

        fb_path = os.path.join(model_dir, "filterbank.npz")
        if not os.path.exists(fb_path):
            raise pb_utils.TritonModelException(
                f"filterbank.npz missing in {model_dir}. It is written by "
                "scripts/export_onnx.py -- regenerate the model repository."
            )
        self.logmel = LogMel.from_npz(fb_path)

        meta_path = os.path.join(model_dir, "meta.json")
        self.meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)

        # Both output dtypes come from the config rather than being hardcoded:
        # features follow the export precision (fp16 on GPU, fp32 on the CPU
        # profile), and feature_lengths must match whatever the exported graph
        # traced for its length input -- int64 for the stock encoder.
        config = json.loads(args["model_config"])
        self.feature_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(config, "features")["data_type"]
        )
        self.length_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(config, "feature_lengths")["data_type"]
        )

        self.sample_rate = int(self.meta.get("sample_rate", 16000))
        self.max_samples = int(self.meta.get("max_audio_s", 25)) * self.sample_rate

        # Ceiling of the largest bucket, in samples. Without it an oversize
        # segment sails through here and dies inside TensorRT with a shape error
        # about a tensor the caller never named -- technically loud, practically
        # unreadable. Matters most in a single-bucket deployment, where every
        # too-long segment hits it.
        params = {k: v["string_value"] for k, v in config.get("parameters", {}).items()}
        largest = int(params.get("largest_bucket_samples", 0))
        self.largest_bucket_samples = largest or None

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                responses.append(self._handle(request))
            except pb_utils.TritonModelException:
                raise
            except Exception as exc:  # surfaced to the client, not swallowed
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"preprocessing failed: {exc}")
                    )
                )

        return responses

    def _handle(self, request):
        audio = pb_utils.get_input_tensor_by_name(request, "audio").as_numpy()
        audio_len = pb_utils.get_input_tensor_by_name(request, "audio_len").as_numpy()

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        n_valid = int(np.asarray(audio_len).reshape(-1)[0])

        error = self._validate(audio, n_valid)
        if error is not None:
            return pb_utils.InferenceResponse(error=pb_utils.TritonError(error))

        # Features are computed over the padded audio -- the encoder masks the
        # tail using feature_lengths, and recomputing per true length would
        # break the fixed shape that lets Triton batch the bucket at all.
        features = self.logmel(audio, dtype=self.feature_dtype)
        n_frames = int(self.logmel.out_len(np.array([n_valid]))[0])
        n_frames = max(1, min(n_frames, features.shape[-1]))

        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("features", features[None, ...]),
                # Shape [batch], not [batch, 1]: config.pbtxt declares
                # `reshape { shape: [] }` on this output, which means the API
                # exposes [B, 1] while the model is expected to produce [B].
                # The encoder's traced length input is rank-1, and this is the
                # seam that reconciles the two.
                pb_utils.Tensor(
                    "feature_lengths", np.array([n_frames], dtype=self.length_dtype)
                ),
            ]
        )

    def _validate(self, audio, n_valid):
        if audio.size == 0:
            return "empty audio"

        if n_valid <= 0 or n_valid > audio.size:
            return (
                f"audio_len {n_valid} outside the supplied audio "
                f"({audio.size} samples)"
            )

        if self.largest_bucket_samples and audio.size > self.largest_bucket_samples:
            return (
                f"audio is {audio.size} samples ({audio.size / self.sample_rate:.1f} s) "
                f"but the largest bucket holds {self.largest_bucket_samples} "
                f"({self.largest_bucket_samples / self.sample_rate:.1f} s). "
                "GigaAM is offline and full-context: segment on the client and "
                "pad each segment to its bucket (spec §11.1)."
            )

        if n_valid > self.max_samples:
            return (
                f"audio_len {n_valid} exceeds max_audio_s "
                f"({self.max_samples} samples). GigaAM is offline and "
                "full-context; segment before sending (spec §11.1)."
            )

        valid = audio[:n_valid]

        if not np.all(np.isfinite(valid)):
            return "audio contains NaN or Inf"

        peak = float(np.max(np.abs(valid)))
        if peak > MAX_PLAUSIBLE_AMPLITUDE:
            return (
                f"audio peak amplitude {peak:.1f} exceeds "
                f"{MAX_PLAUSIBLE_AMPLITUDE}: expected float32 in [-1, 1], this "
                "looks like int16 samples cast instead of converted. Divide by "
                "32768. Failing loudly because the model would otherwise return "
                "an empty transcript with no error (spec §11.2)."
            )

        return None
