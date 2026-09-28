#!/usr/bin/env python3
"""PyTorch -> ONNX export for GigaAM v3 (spec §5).

Built on the stock ``gigaam`` model, NOT on ``triton_scripts/run_convert_onnx.py``.
That upstream script swaps the forward so CTC emits only ``token_ids = argmax``,
"to avoid variable output shapes" -- which throws away the log-probs our provider
needs to do its own decoding for confidence and word timings (spec §7.2). The
reasoning does not hold either: the vocabulary axis is fixed, only time is
dynamic, and time is dynamic regardless.

So we emit both. The graph returns:

    log_probs       FP32  [B, T, V]   for the provider to decode itself
    token_ids       INT32 [B, T]      per-frame argmax, done on the GPU
    token_logprobs  FP32  [B, T]      value at that argmax
    encoded_lengths INT32 [B]

The argmax lives in the graph on purpose. Moving a frames-by-vocab tensor into a
CPU Python backend just to take a max would cost more than the encoder pass we
are trying to keep under 25 ms; this way the Python step only collapses repeats.

  python scripts/export_onnx.py --variant v3_e2e_ctc --dtype fp16 --out /work/onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fbank_export  # noqa: E402
from backends.common import ctc as ctc_common  # noqa: E402
from backends.common.logmel import LogMel  # noqa: E402
from backends.common.tokenizer import Tokenizer  # noqa: E402

SAMPLE_RATE = 16000
GOLDEN_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav"
MAX_GOLDEN_CER = 0.005  # spec §5.4 / §9.2


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def ctc_forward_for_export(
    self: Any, features: Tensor, feature_lengths: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Encoder + CTC head, returning log-probs *and* the argmax.

    ``CTCHead.forward`` already ends in log_softmax, so log-probs cost nothing
    extra. Casting them to fp32 in-graph is what lets the ensemble expose
    ``logprobs`` as FP32 per spec §7.2 without a Python round trip.
    """
    encoded, encoded_len = self.encoder(features, feature_lengths)
    log_probs = self.head(encoded)                       # [B, T, V], log_softmax
    token_logprobs, token_ids = log_probs.max(dim=-1)
    return (
        log_probs.float(),
        token_ids.to(torch.int32),
        token_logprobs.float(),
        encoded_len.to(torch.int32),
    )


CTC_IO = {
    "input_names": ["features", "feature_lengths"],
    "output_names": ["log_probs", "token_ids", "token_logprobs", "encoded_lengths"],
    "dynamic_axes": {
        "features": {0: "batch_size", 2: "seq_len"},
        "feature_lengths": {0: "batch_size"},
        "log_probs": {0: "batch_size", 1: "seq_len"},
        "token_ids": {0: "batch_size", 1: "seq_len"},
        "token_logprobs": {0: "batch_size", 1: "seq_len"},
        "encoded_lengths": {0: "batch_size"},
    },
}


def export_ctc(model: Any, out_dir: Path, dtype: torch.dtype) -> dict:
    from gigaam.utils import onnx_converter

    saved = model.forward, model.forward_for_export
    model.forward_for_export = types.MethodType(ctc_forward_for_export, model)
    model.forward = model.forward_for_export
    try:
        with model.encoder.onnx_export_mode():
            onnx_converter(
                model_name="model",
                out_dir=str(out_dir),
                module=model,
                inputs=model.encoder.input_example(),
                export_dtype=dtype,
                **CTC_IO,
            )
    finally:
        model.forward, model.forward_for_export = saved

    return {
        "graph": "ctc",
        "files": {"encoder": "model.onnx"},
        "inputs": CTC_IO["input_names"],
        "outputs": CTC_IO["output_names"],
    }


def export_rnnt(model: Any, out_dir: Path, dtype: torch.dtype) -> dict:
    """Encoder / decoder / joint as three graphs.

    The transducer loop itself stays outside: it is the known bottleneck
    (spec §7.4) and gets its own treatment in the rnnt_decoder_joint backend.
    """
    model.to_onnx(str(out_dir), dtype=dtype)

    name = model.cfg.model_name
    renames = {
        f"{name}_encoder.onnx": "encoder.onnx",
        f"{name}_decoder.onnx": "decoder.onnx",
        f"{name}_joint.onnx": "joint.onnx",
    }
    for src, dst in renames.items():
        src_path = out_dir / src
        if src_path.exists():
            src_path.rename(out_dir / dst)

    # The stock to_onnx also drops a <model_name>.yaml next to the graphs; we
    # carry our own meta.json instead and leave that one alone.
    return {
        "graph": "rnnt",
        "files": {
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joint": "joint.onnx",
        },
        "inputs": ["audio_signal", "length"],
        "outputs": ["encoded", "encoded_len"],
    }


# --------------------------------------------------------------------------
# side artefacts
# --------------------------------------------------------------------------


def family_of(model_name: str) -> str:
    """v3 preprocessing differs from earlier families, and Triton shares one
    preprocessing model -- see spec §5.2 / §11.3."""
    for fam in ("v3", "v2", "v1"):
        if model_name.startswith(fam):
            return fam
    return "other"


def write_preprocessing(model: Any, out_root: Path, family: str) -> Path:
    """Filterbank goes to preprocessing/<family>/, and a second variant of the
    same family must agree with what is already there.

    Upstream writes preprocessing config to one shared directory, so whichever
    model was converted last silently wins (spec §11.3). Splitting by family and
    checksumming makes that collision an error instead of a mystery.
    """
    prep_dir = out_root / "preprocessing" / family
    prep_dir.mkdir(parents=True, exist_ok=True)
    fb_path = prep_dir / "filterbank.npz"

    arrays = fbank_export.extract(model.preprocessor)
    new_digest = _filterbank_digest(arrays)

    if fb_path.exists():  # noqa: SIM102 -- the mismatch branch needs the detail
        # Hash the array contents, not the file. np.savez writes a zip and zip
        # entries carry a timestamp, so identical filterbanks written a second
        # apart produce different bytes -- comparing files would reject a
        # perfectly compatible second variant.
        existing = dict(np.load(fb_path, allow_pickle=False))
        old_digest = _filterbank_digest(existing)
        if old_digest != new_digest:
            differing = sorted(
                k for k in set(existing) | set(arrays)
                if k not in existing or k not in arrays
                or not np.array_equal(np.asarray(existing[k]), np.asarray(arrays[k]))
            )
            raise SystemExit(
                f"preprocessing mismatch inside family '{family}':\n"
                f"  {fb_path} sha256 {old_digest[:16]}\n"
                f"  this model    sha256 {new_digest[:16]}\n"
                f"  differing: {', '.join(differing)}\n"
                "Two models of the same family disagree on feature extraction. "
                "They cannot share one Triton preprocessing model (spec §5.2)."
            )
        return fb_path, arrays

    np.savez(fb_path, **arrays)
    return fb_path, arrays


def _filterbank_digest(arrays: dict) -> str:
    """Content hash, independent of how the arrays were serialised."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.asarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def write_meta(
    model: Any, out_dir: Path, variant: str, dtype: str, graph: dict, fbank: dict
) -> dict:
    """meta.json is consumed by the SecondChain provider (spec §5.3) -- these
    values must come from the checkpoint, not be invented downstream."""
    cfg = model.cfg
    tokenizer_model = None

    sp_path = cfg.decoding.get("model_path")
    if sp_path and Path(sp_path).exists():
        tokenizer_model = f"{variant}_tokenizer.model"
        shutil.copy(sp_path, out_dir / tokenizer_model)
        vocab_size = len(Tokenizer(model_path=str(out_dir / tokenizer_model)))
    else:
        vocab = list(cfg.decoding.get("vocabulary"))
        (out_dir / "vocab.json").write_text(
            json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
        )
        vocab_size = len(vocab)

    is_e2e = "e2e" in variant
    meta = {
        "variant": variant,
        "family": family_of(variant),
        "graph": graph["graph"],
        "export_dtype": dtype,
        "sample_rate": int(fbank["sample_rate"]),
        "subsampling_factor": int(cfg.encoder.get("subsampling_factor", 4)),
        "frame_ms": round(
            1000 * int(fbank["hop_length"]) / int(fbank["sample_rate"])
            * int(cfg.encoder.get("subsampling_factor", 4))
        ),
        "max_audio_s": 25,
        "n_mels": int(fbank["n_mels"]),
        # Feature geometry travels with the model because it is NOT the library
        # default: v3 runs n_fft=320 with center=False where v2 uses 400 centred.
        # Every frame count downstream (TRT profile shapes, warmup, bucket
        # padding) is derived from these rather than assumed.
        "n_fft": int(fbank["n_fft"]),
        "hop_length": int(fbank["hop_length"]),
        "win_length": int(fbank["win_length"]),
        "center": bool(fbank["center"]),
        "vocab_size": vocab_size,
        # The CTC head emits one class more than the tokenizer holds: the blank
        # sits at index len(tokenizer). Declaring the tensor as vocab_size wide
        # makes Triton reject the model, so the two numbers are kept apart.
        "num_classes": vocab_size + 1,
        "blank_id": vocab_size,
        "tokenizer_model": tokenizer_model,
        "output_case": "cased" if is_e2e else "lowercase",
        "has_punctuation": is_e2e,
        "has_itn": is_e2e,
        "files": graph["files"],
    }

    if graph["graph"] == "rnnt":
        # The decode loop backend preallocates every buffer from these, so they
        # travel with the model instead of being rediscovered from the graph.
        head = cfg.head
        meta.update(
            {
                "enc_hidden": int(head.joint.enc_hidden),
                "pred_hidden": int(head.decoder.pred_hidden),
                "pred_rnn_layers": int(head.decoder.pred_rnn_layers),
                "joint_hidden": int(head.joint.joint_hidden),
            }
        )
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    return levenshtein(ref, hyp) / max(1, len(ref))


def golden_wavs(golden_dir: Path) -> list[Path]:
    golden_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(golden_dir.glob("*.wav"))
    if wavs:
        return wavs

    import urllib.request

    dest = golden_dir / "example.wav"
    print(f"golden set empty -- fetching {GOLDEN_URL}")
    try:
        urllib.request.urlretrieve(GOLDEN_URL, dest)
    except Exception as exc:
        raise SystemExit(
            f"no golden audio in {golden_dir} and download failed: {exc}. "
            "Export validation (spec §5.4) cannot run without reference audio."
        )
    return [dest]


def validate(
    model: Any, out_dir: Path, fb_path: Path, meta: dict, golden_dir: Path
) -> None:
    """End-to-end: torch reference against the exact path the server will run
    (numpy log-mel -> ONNX graph -> our CTC collapse).

    Comparing only the encoder would leave the two pieces most likely to drift --
    the torch-free front end and the collapse -- unchecked.
    """
    import onnxruntime as ort

    if meta["graph"] != "ctc":
        print("validation currently covers the CTC graph only -- skipping")
        return

    wavs = golden_wavs(golden_dir)
    logmel = LogMel.from_npz(str(fb_path))
    tokenizer = Tokenizer.from_dir(str(out_dir))

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(out_dir / "model.onnx"), providers=providers)
    feat_dtype = np.float16 if meta["export_dtype"] == "fp16" else np.float32

    print(f"\nvalidating export against torch on {len(wavs)} golden file(s)")
    worst = 0.0
    failures = []

    for wav in wavs:
        from gigaam.preprocess import load_audio

        audio = load_audio(str(wav)).numpy().astype(np.float32)

        with torch.inference_mode():
            reference = model.transcribe(str(wav)).text

        feats = logmel(audio, dtype=feat_dtype)[None, ...]
        lengths = np.array([audio.shape[0]], dtype=np.int64)
        feat_len = logmel.out_len(lengths).astype(np.int64)

        token_ids, token_logprobs, enc_len = sess.run(
            ["token_ids", "token_logprobs", "encoded_lengths"],
            {"features": feats, "feature_lengths": feat_len},
        )
        hyp, _, _ = ctc_common.decode(
            token_ids[0], token_logprobs[0], int(enc_len[0]), tokenizer.blank_id, tokenizer
        )

        score = cer(reference, hyp)
        worst = max(worst, score)
        mark = "ok " if score <= MAX_GOLDEN_CER else "FAIL"
        print(f"  {mark} {wav.name}: CER {score:.4f}")
        if score > MAX_GOLDEN_CER:
            failures.append((wav.name, reference, hyp, score))

    if failures:
        print("\nONNX export diverges from the torch reference:", file=sys.stderr)
        for name, ref, hyp, score in failures:
            print(f"  {name}: CER {score:.4f}", file=sys.stderr)
            print(f"    torch: {ref}", file=sys.stderr)
            print(f"    onnx : {hyp}", file=sys.stderr)
        raise SystemExit(
            f"golden CER above {MAX_GOLDEN_CER:.3f} (spec §5.4) -- not shipping this export"
        )

    print(f"validation PASSED -- worst CER {worst:.4f} (limit {MAX_GOLDEN_CER:.3f})")


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--out", required=True, help="output root")
    parser.add_argument("--golden-dir", default=None)
    parser.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_root = Path(args.out)
    out_dir = out_root / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # An fp16 and an fp32 export of the same variant are different artefacts that
    # want the same directory name. Overwriting silently would leave a repository
    # whose meta.json and model.onnx disagree about precision, which surfaces much
    # later as a dtype error deep in Triton. Keep them in separate roots.
    existing_meta = out_dir / "meta.json"
    if existing_meta.exists():
        previous = json.loads(existing_meta.read_text()).get("export_dtype")
        if previous and previous != args.dtype:
            raise SystemExit(
                f"{out_dir} already holds a {previous} export of {args.variant}.\n"
                f"Exporting {args.dtype} here would overwrite it. Use a separate "
                f"--out root per precision (build.sh uses artifacts/<dtype>)."
            )
    golden_dir = Path(args.golden_dir or repo_root / "testdata" / "golden")

    import gigaam

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is torch.float16 and device == "cpu":
        # The RNNT prediction network is an LSTM, and torch has no fp16 CPU
        # kernel for it -- tracing would die mid-export with an unhelpful error.
        raise SystemExit(
            "fp16 export needs a CUDA device (no fp16 LSTM kernel on CPU). "
            "Use --dtype fp32 for a CPU-only box."
        )

    print(f"loading {args.variant} on {device} ({args.dtype})")
    model = gigaam.load_model(
        args.variant, device=device, fp16_encoder=(dtype is torch.float16)
    )
    model.eval()

    family = family_of(model.cfg.model_name)
    fb_path, fbank = write_preprocessing(model, out_root, family)
    print(
        f"preprocessing -> {fb_path} "
        f"(n_fft={int(fbank['n_fft'])} hop={int(fbank['hop_length'])} "
        f"center={bool(fbank['center'])})"
    )

    is_ctc = "ctc" in args.variant
    graph = export_ctc(model, out_dir, dtype) if is_ctc else export_rnnt(model, out_dir, dtype)

    meta = write_meta(model, out_dir, args.variant, args.dtype, graph, fbank)
    print(f"meta -> {out_dir / 'meta.json'} (vocab {meta['vocab_size']}, blank {meta['blank_id']})")

    if args.skip_validate:
        print("validation skipped by request")
    else:
        validate(model, out_dir, fb_path, meta, golden_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
