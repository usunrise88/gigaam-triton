"""Triton python backend: the RNN-Transducer decode loop.

This is the known bottleneck (spec §7.4). Upstream's version calls
``sess.run()`` twice per step on plain numpy arrays, which means a fresh
allocation and a host round trip for the LSTM state on every symbol -- the
official benchmark puts e2e_rnnt at 0.403 s against 0.034 s for CTC, and
essentially all of that gap is here rather than in the encoder.

So this implementation, per §7.4.1-3:

* every buffer is allocated once in ``initialize`` and reused;
* the decoder state stays on the device, ping-ponged between two bound buffers
  instead of travelling through the host each step;
* the encoder output is uploaded once per request, transposed so that frame t is
  contiguous, and the joint's ``enc`` input is bound by pointer offset -- there
  is no per-frame copy at all;
* ``max_symbols_per_frame`` is a config parameter, not a module constant.

The provider is a config parameter too, because which side wins is not obvious:
the decoder is a 320-wide LSTM and the joint is two small matmuls, so kernel
launch overhead can easily exceed the arithmetic. Spec §7.4.5 asks for the split
between encoder, joint and Python overhead, and that needs both numbers.

Torch-free like the other backends (spec §4.2).
"""

import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import triton_python_backend_utils as pb_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.tokenizer import Tokenizer  # noqa: E402

# The graphs are fp16 for the GPU profile and fp32 for the CPU one, so the
# buffer dtype is read off the session rather than assumed -- a mismatch here
# surfaces as "Unexpected input data type" only once a request arrives.
_ORT_TO_NUMPY = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
}


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.join(args["model_repository"], args["model_version"])
        config = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in config.get("parameters", {}).items()}

        with open(os.path.join(model_dir, "meta.json")) as f:
            self.meta = json.load(f)

        self.tokenizer = Tokenizer.from_dir(model_dir)
        self.blank_id = int(self.meta["blank_id"])
        self.num_classes = int(self.meta["num_classes"])
        self.enc_hidden = int(self.meta["enc_hidden"])
        self.pred_hidden = int(self.meta["pred_hidden"])
        self.pred_layers = int(self.meta["pred_rnn_layers"])

        # Upstream hardcodes 3. For a char vocabulary that is already generous;
        # for this SentencePiece vocabulary it is about right, but it stays a
        # parameter so it can be tuned against the loop profile (§7.4.3).
        self.max_symbols = int(params.get("max_symbols_per_frame", 3))

        requested = params.get("provider", "cuda").lower()
        available = ort.get_available_providers()
        if requested == "cuda" and "CUDAExecutionProvider" in available:
            self.provider, self.device = "CUDAExecutionProvider", "cuda"
        else:
            self.provider, self.device = "CPUExecutionProvider", "cpu"
        self.device_id = 0
        self.needs_sync = self.device == "cuda"

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = int(params.get("intra_op_thread_count", 1))
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3

        def session(name):
            path = os.path.join(model_dir, name)
            if not os.path.exists(path):
                raise pb_utils.TritonModelException(f"{path} missing")
            return ort.InferenceSession(path, providers=[self.provider], sess_options=opts)

        self.dec_sess = session("decoder.onnx")
        self.joint_sess = session("joint.onnx")
        self.dec_io = self.dec_sess.io_binding()
        self.joint_io = self.joint_sess.io_binding()

        self.float_dtype = np.float16
        for inp in self.dec_sess.get_inputs():
            if inp.type in _ORT_TO_NUMPY:
                self.float_dtype = _ORT_TO_NUMPY[inp.type]
                break

        self._allocate()

        self.stats = {"requests": 0, "frames": 0, "steps": 0,
                      "loop_s": 0.0, "upload_s": 0.0}

    # -- buffers ---------------------------------------------------------

    def _dev(self, shape, dtype):
        return ort.OrtValue.ortvalue_from_numpy(
            np.zeros(shape, dtype=dtype), self.device, self.device_id
        )

    def _allocate(self):
        L, H, D = self.pred_layers, self.pred_hidden, self.enc_hidden
        FP = self.float_dtype

        self.label = self._dev((1, 1), np.int64)
        # Two state buffers per tensor: the decoder cannot read and write the
        # same memory, so the step alternates between them rather than copying.
        self.h = [self._dev((L, 1, H), FP), self._dev((L, 1, H), FP)]
        self.c = [self._dev((L, 1, H), FP), self._dev((L, 1, H), FP)]

        # dec comes out as [1, 1, H] and the joint wants [1, H, 1]. Both are the
        # same H contiguous halves, so the same allocation is bound twice under
        # two shapes and the transpose costs nothing.
        self.dec = self._dev((1, 1, H), FP)
        self.joint_out = self._dev((1, 1, 1, self.num_classes), FP)

        self.zeros_state = np.zeros((L, 1, H), dtype=FP)
        self.enc_dev = None          # per-request, sized by frame count
        self.enc_capacity = 0

        self.dec_io.bind_output(
            "dec", self.device, self.device_id, FP, (1, 1, H), self.dec.data_ptr()
        )
        self.joint_io.bind_input(
            "dec", self.device, self.device_id, FP, (1, H, 1), self.dec.data_ptr()
        )
        self.joint_io.bind_output(
            "joint", self.device, self.device_id, FP,
            (1, 1, 1, self.num_classes), self.joint_out.data_ptr(),
        )

    def _ensure_enc_capacity(self, frames):
        if self.enc_dev is not None and frames <= self.enc_capacity:
            return
        # Grow to the next power of two so a long request does not force a
        # reallocation on every subsequent one.
        capacity = 1
        while capacity < frames:
            capacity *= 2
        self.enc_dev = self._dev((capacity, self.enc_hidden), self.float_dtype)
        self.enc_capacity = capacity

    # -- decode ----------------------------------------------------------

    def _decode(self, encoded, n_frames):
        """encoded: [enc_hidden, T] in the graph's float dtype, one sample."""
        L, H, D = self.pred_layers, self.pred_hidden, self.enc_hidden
        FP = self.float_dtype
        itemsize = np.dtype(FP).itemsize

        started = time.perf_counter()
        self._ensure_enc_capacity(max(1, n_frames))
        # Transposed once so frame t is D contiguous values: the joint's enc
        # input is then just a pointer offset, never a copy.
        frames_first = np.ascontiguousarray(encoded[:, :n_frames].T, dtype=FP)
        padded = np.zeros((self.enc_capacity, D), dtype=FP)
        padded[:n_frames] = frames_first
        self.enc_dev.update_inplace(padded)
        upload_s = time.perf_counter() - started

        self.h[0].update_inplace(self.zeros_state)
        self.c[0].update_inplace(self.zeros_state)
        cur = 0
        label_host = np.array([[self.blank_id]], dtype=np.int64)
        self.label.update_inplace(label_host)

        hyp: list[int] = []
        scores: list[float] = []
        steps = 0
        base_ptr = self.enc_dev.data_ptr()

        loop_started = time.perf_counter()
        for t in range(n_frames):
            self.joint_io.bind_input(
                "enc", self.device, self.device_id, FP, (1, D, 1),
                base_ptr + t * D * itemsize,
            )
            for _ in range(self.max_symbols):
                nxt = 1 - cur
                self.dec_io.bind_input(
                    "x", self.device, self.device_id, np.int64, (1, 1), self.label.data_ptr()
                )
                self.dec_io.bind_input(
                    "hi", self.device, self.device_id, FP, (L, 1, H), self.h[cur].data_ptr()
                )
                self.dec_io.bind_input(
                    "ci", self.device, self.device_id, FP, (L, 1, H), self.c[cur].data_ptr()
                )
                self.dec_io.bind_output(
                    "ho", self.device, self.device_id, FP, (L, 1, H), self.h[nxt].data_ptr()
                )
                self.dec_io.bind_output(
                    "co", self.device, self.device_id, FP, (L, 1, H), self.c[nxt].data_ptr()
                )
                self.dec_sess.run_with_iobinding(self.dec_io)
                self.joint_sess.run_with_iobinding(self.joint_io)
                steps += 1

                # The next step's input is the previous step's output, and the
                # emit/stop decision reads the joint on the host, so the stream
                # has to be drained here rather than at the end of the request.
                if self.needs_sync:
                    self.joint_io.synchronize_outputs()

                logits = self.joint_out.numpy().reshape(-1)
                k = int(np.argmax(logits))
                if k == self.blank_id:
                    break

                hyp.append(k)
                scores.append(float(logits[k]))
                label_host[0, 0] = k
                self.label.update_inplace(label_host)
                cur = nxt          # commit the state only when a symbol is emitted

        loop_s = time.perf_counter() - loop_started

        self.stats["frames"] += n_frames
        self.stats["steps"] += steps
        self.stats["loop_s"] += loop_s
        self.stats["upload_s"] += upload_s
        return hyp, scores, steps, loop_s, upload_s

    # -- triton ----------------------------------------------------------

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                responses.append(self._handle(request))
            except Exception as exc:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"rnnt decode failed: {exc}")
                    )
                )
        return responses

    def _handle(self, request):
        encoded = pb_utils.get_input_tensor_by_name(request, "encoded").as_numpy()
        n_frames = int(
            pb_utils.get_input_tensor_by_name(request, "encoded_lengths")
            .as_numpy().reshape(-1)[0]
        )

        enc = np.asarray(encoded, dtype=self.float_dtype)
        if enc.ndim == 3:
            enc = enc[0]                       # [D, T]
        n_frames = max(0, min(n_frames, enc.shape[1]))

        if n_frames == 0:
            hyp, scores, steps, loop_s, upload_s = [], [], 0, 0.0, 0.0
        else:
            hyp, scores, steps, loop_s, upload_s = self._decode(enc, n_frames)

        self.stats["requests"] += 1
        text = self.tokenizer.decode(hyp)

        # Padded to the frame count so responses in a batch keep the same shape.
        width = max(1, int(enc.shape[1]))
        tokens = np.full(width, self.blank_id, dtype=np.int32)
        token_scores = np.zeros(width, dtype=np.float32)
        if hyp:
            n = min(len(hyp), width)
            tokens[:n] = np.asarray(hyp[:n], dtype=np.int32)
            token_scores[:n] = np.asarray(scores[:n], dtype=np.float32)

        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor("text", np.array([[text.encode("utf-8")]], dtype=object)),
                pb_utils.Tensor("tokens", tokens[None, :]),
                pb_utils.Tensor("token_scores", token_scores[None, :]),
                pb_utils.Tensor("tokens_len", np.array([[len(hyp)]], dtype=np.int32)),
                # The loop profile travels with the response so §7.4.5 can be
                # answered from a normal run rather than a special build.
                pb_utils.Tensor("decode_steps", np.array([[steps]], dtype=np.int32)),
                pb_utils.Tensor("loop_ms", np.array([[loop_s * 1000]], dtype=np.float32)),
                pb_utils.Tensor("upload_ms", np.array([[upload_s * 1000]], dtype=np.float32)),
            ]
        )

    def finalize(self):
        s = self.stats
        if s["requests"]:
            sys.stderr.write(
                f"[rnnt_decoder_joint] {self.provider} "
                f"requests={s['requests']} frames={s['frames']} steps={s['steps']} "
                f"steps_per_frame={s['steps'] / max(1, s['frames']):.2f} "
                f"loop={s['loop_s'] * 1000 / s['requests']:.1f} ms/req "
                f"upload={s['upload_s'] * 1000 / s['requests']:.2f} ms/req\n"
            )
            sys.stderr.flush()
