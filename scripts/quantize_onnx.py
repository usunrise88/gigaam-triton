#!/usr/bin/env python3
"""Dynamic int8 quantisation for the CPU profile (spec §8).

Weights go to int8, activations stay float and are quantised on the fly. No
calibration set is needed, which is the point: it is the cheap option, and public
numbers say it is nearly free for v3 (RTFx 58.8 -> 52.2 on CTC) where the same
trick cost v2 a factor of three.

"Nearly free" is a claim about someone else's hardware, so this script always
reports accuracy either side of the conversion. Speed without an accuracy number
means nothing here -- a quantised model that is 15% faster and 3% worse is not an
improvement, it is a different product.

  python scripts/quantize_onnx.py --onnx /work/onnx/v3_e2e_ctc/model.onnx \\
      --out /work/onnx/v3_e2e_ctc/model.int8.onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.common import ctc as ctc_common  # noqa: E402
from backends.common.logmel import LogMel  # noqa: E402
from backends.common.tokenizer import Tokenizer  # noqa: E402

SAMPLE_RATE = 16000

# Conv is left alone deliberately. The subsampling front end is where the signal
# is still dense and int8 conv on CPU tends to cost accuracy for little speed;
# MatMul and Gemm carry the Conformer's parameters and are where the win is.
DEFAULT_OPS = ["MatMul", "Gemm"]


def cpu_int8_support() -> dict:
    """What the host CPU can actually do with int8.

    This decides both whether quantisation can pay off at all and whether the
    accumulator needs the reduced range. Without VNNI, int8 GEMM goes through a
    widening path that is no faster than fp32, and the u8s8 product can saturate
    unless weights are kept to 7 bits.
    """
    flags: set[str] = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass

    model = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    return {
        "model": model,
        "avx2": "avx2" in flags,
        "avx512": any(f.startswith("avx512") for f in flags),
        "vnni": "avx512_vnni" in flags or "avx_vnni" in flags,
        "amx_int8": "amx_int8" in flags,
    }


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


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def transcribe(session, logmel: LogMel, tokenizer: Tokenizer, audio: np.ndarray,
               feat_dtype) -> tuple[str, float]:
    feats = logmel(audio, dtype=feat_dtype)[None, ...]
    feat_len = logmel.out_len(np.array([audio.shape[0]])).astype(np.int64)

    started = time.perf_counter()
    token_ids, token_logprobs, enc_len = session.run(
        ["token_ids", "token_logprobs", "encoded_lengths"],
        {"features": feats, "feature_lengths": feat_len},
    )
    elapsed = time.perf_counter() - started

    text, _, _ = ctc_common.decode(
        token_ids[0], token_logprobs[0], int(enc_len[0]), tokenizer.blank_id, tokenizer
    )
    return text, elapsed


def evaluate(onnx_path: Path, wavs: list[Path], logmel, tokenizer, feat_dtype,
             threads: int, repeats: int) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"],
                                sess_options=opts)

    import soundfile as sf

    results = []
    for wav in wavs:
        audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            continue
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        transcribe(sess, logmel, tokenizer, audio, feat_dtype)  # warm
        times = []
        for _ in range(repeats):
            text, elapsed = transcribe(sess, logmel, tokenizer, audio, feat_dtype)
            times.append(elapsed)

        duration = audio.shape[0] / SAMPLE_RATE
        median = float(np.median(times))
        results.append(
            {
                "wav": wav.name,
                "text": text,
                "duration_s": round(duration, 2),
                "median_s": round(median, 4),
                "rtfx": round(duration / median, 1),
            }
        )
    return {"threads": threads, "files": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="fp32 ONNX to quantise")
    parser.add_argument("--out", default=None, help="default: <onnx>.int8.onnx")
    parser.add_argument("--golden-dir", default=None)
    parser.add_argument("--ops", default=",".join(DEFAULT_OPS))
    parser.add_argument("--per-channel", action="store_true", default=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"{onnx_path} not found")

    meta = json.loads((onnx_path.parent / "meta.json").read_text())
    if meta["export_dtype"] != "fp32":
        raise SystemExit(
            f"this export is {meta['export_dtype']}; dynamic quantisation expects an "
            "fp32 graph. Re-export with --dtype fp32 (that is what --runtime onnx-cpu "
            "does)."
        )

    out_path = Path(args.out) if args.out else onnx_path.with_suffix(".int8.onnx")

    cpu = cpu_int8_support()
    reduce_range = not cpu["vnni"]
    print(f"host CPU: {cpu['model']}")
    print(f"  avx2={cpu['avx2']} avx512={cpu['avx512']} vnni={cpu['vnni']} "
          f"amx_int8={cpu['amx_int8']}  ->  reduce_range={reduce_range}")
    if not cpu["vnni"]:
        print(
            "  WARNING: no VNNI. int8 GEMM has no hardware accumulate here, so\n"
            "  expect the quantised model to be no faster than fp32 -- the win is\n"
            "  memory, not latency. Spec §2 assumes an AVX-512 CPU target; numbers\n"
            "  measured on this host do not carry over to one."
        )

    fb = onnx_path.parent.parent / "preprocessing" / meta["family"] / "filterbank.npz"
    logmel = LogMel.from_npz(str(fb))
    tokenizer = Tokenizer.from_dir(str(onnx_path.parent))
    golden_dir = Path(args.golden_dir or Path(__file__).resolve().parent.parent
                      / "testdata" / "golden")
    wavs = sorted(golden_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no audio in {golden_dir} -- cannot report accuracy")

    print(f"baseline (fp32), {args.threads} intra-op threads")
    before = evaluate(onnx_path, wavs, logmel, tokenizer, np.float32,
                      args.threads, args.repeats)
    for f in before["files"]:
        print(f"  {f['wav']}: {f['median_s'] * 1000:.0f} ms, RTFx {f['rtfx']}")

    print(f"\nquantising {onnx_path.name} -> {out_path.name} "
          f"(ops: {args.ops}, per_channel={args.per_channel})")
    started = time.perf_counter()
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(out_path),
        op_types_to_quantize=[o.strip() for o in args.ops.split(",") if o.strip()],
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        # reduce_range keeps weights to 7 bits so the u8s8 accumulation cannot
        # saturate. It is only unnecessary where VNNI does the accumulation in
        # hardware, so it follows the CPU rather than being hardcoded either way.
        reduce_range=reduce_range,
        extra_options={"MatMulConstBOnly": True},
    )
    print(f"  done in {time.perf_counter() - started:.0f}s, "
          f"{onnx_path.stat().st_size / 1e6:.0f} MB -> {out_path.stat().st_size / 1e6:.0f} MB")

    print("\nquantised (int8)")
    after = evaluate(out_path, wavs, logmel, tokenizer, np.float32,
                     args.threads, args.repeats)
    for f in after["files"]:
        print(f"  {f['wav']}: {f['median_s'] * 1000:.0f} ms, RTFx {f['rtfx']}")

    print("\naccuracy and speed, fp32 -> int8")
    rows = []
    for b, a in zip(before["files"], after["files"]):
        reference = (golden_dir / b["wav"]).with_suffix(".txt")
        ref_text = reference.read_text(encoding="utf-8").strip() if reference.exists() else b["text"]
        row = {
            "wav": b["wav"],
            "cer_fp32": round(cer(ref_text, b["text"]), 5),
            "cer_int8": round(cer(ref_text, a["text"]), 5),
            "wer_fp32": round(wer(ref_text, b["text"]), 5),
            "wer_int8": round(wer(ref_text, a["text"]), 5),
            "rtfx_fp32": b["rtfx"],
            "rtfx_int8": a["rtfx"],
            "speedup": round(a["rtfx"] / b["rtfx"], 2) if b["rtfx"] else None,
            "text_fp32": b["text"],
            "text_int8": a["text"],
        }
        rows.append(row)
        print(f"  {row['wav']}: CER {row['cer_fp32']:.4f} -> {row['cer_int8']:.4f}   "
              f"WER {row['wer_fp32']:.4f} -> {row['wer_int8']:.4f}   "
              f"RTFx {row['rtfx_fp32']} -> {row['rtfx_int8']} (x{row['speedup']})")
        if row["text_fp32"] != row["text_int8"]:
            print(f"      fp32: {row['text_fp32']}")
            print(f"      int8: {row['text_int8']}")

    # The quantised weights are host-dependent in a way the graph does not
    # advertise: reduce_range bakes 7-bit weights in, which is right without VNNI
    # and leaves accuracy on the table with it. Record what this was built for so
    # a deploy onto different silicon can notice rather than guess.
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "reduce_range": reduce_range,
                "per_channel": args.per_channel,
                "ops_quantised": args.ops,
                "built_for_cpu": cpu,
                "note": (
                    "Re-run scripts/quantize_onnx.py on the target if its CPU "
                    "differs in VNNI support -- the graph will load either way, "
                    "but the weights were shaped for this one."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    report = {
        "onnx": str(onnx_path),
        "quantised": str(out_path),
        "ops_quantised": args.ops,
        "per_channel": args.per_channel,
        "reduce_range": reduce_range,
        "built_for_cpu": cpu,
        "intra_op_threads": args.threads,
        "rows": rows,
        "caveat": (
            "Measured on the golden set, which is a handful of files, not the call "
            "corpus. Spec §8 asks for WER either side of quantisation; that number "
            "needs the real corpus and belongs to the eval step. Treat these as a "
            "regression tripwire, not as the WER answer."
        ),
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.report}")

    print("\nNOTE: this is the golden set, not the call corpus. The WER comparison")
    print("spec §8 asks for needs real audio and lands with the eval step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
