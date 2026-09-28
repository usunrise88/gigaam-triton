#!/usr/bin/env python3
"""ONNX -> TensorRT engine with one optimization profile per duration bucket
(spec §6).

Uses ``trtexec --profile=N`` rather than the TensorRT Python API on purpose: the
Triton image ships libnvinfer but no ``tensorrt`` python module, and installing a
wheel to get one risks building the plan against a different TRT than the server
loads it with -- the exact mismatch §4.1 warns about.

Engines are not baked into the image. They are keyed to the GPU, the driver and
the TRT version (§6.3), so they are built on the target machine into a mounted
cache and rebuilt when any of that moves.

  python scripts/build_trt.py --onnx /work/onnx/v3_e2e_ctc/model.onnx \\
      --precision fp16 --buckets 2,5,10,20 --out /work/cache/engines/v3_e2e_ctc.plan
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import buckets as bucket_mod  # noqa: E402

TRTEXEC_CANDIDATES = ["/usr/src/tensorrt/bin/trtexec", "trtexec"]


def find_trtexec() -> str:
    for candidate in TRTEXEC_CANDIDATES:
        resolved = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if resolved:
            return resolved
    raise SystemExit("trtexec not found -- this must run inside the Triton image")


def trt_version() -> str:
    lib = ctypes.CDLL("libnvinfer.so")
    lib.getInferLibVersion.restype = ctypes.c_int
    raw = int(lib.getInferLibVersion())
    major, rest = divmod(raw, 10000)
    minor, patch = divmod(rest, 100)
    return f"{major}.{minor}.{patch}"


def gpu_identity() -> dict[str, str]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    name, compute_cap, driver = [p.strip() for p in out.split(",")]
    return {"gpu_name": name, "compute_cap": compute_cap, "driver_version": driver}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engine_key(onnx_sha: str, trt: str, gpu: dict[str, str], flags: list[str]) -> str:
    """Spec §6.3 verbatim: onnx + trt version + gpu + compute cap + driver + flags.

    A driver update alone invalidates a plan, so the driver belongs in the key
    even though nothing about the model changed (§11.4)."""
    payload = "\x00".join(
        [
            onnx_sha,
            trt,
            gpu["gpu_name"],
            gpu["compute_cap"],
            gpu["driver_version"],
            json.dumps(flags, sort_keys=True),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# The CTC graph takes features/feature_lengths, the RNNT encoder takes
# audio_signal/length. Same tensors, different names -- taken from the graph
# rather than assumed, because a wrong name here fails deep inside trtexec.
GRAPH_INPUTS = {
    "ctc": ("features", "feature_lengths"),
    "rnnt": ("audio_signal", "length"),
}


def shape_args(
    bucket_list: list[bucket_mod.Bucket],
    n_mels: int,
    min_bs: int,
    opt_bs: int,
    max_bs: int,
    feat_name: str,
    len_name: str,
) -> list[str]:
    """One --profile block per bucket, indices contiguous from 0.

    gen_model_repo.py pins Triton instances to these indices; scripts/buckets.py
    is the single source both read so the order cannot drift apart.
    """
    args: list[str] = []
    for b in bucket_list:
        args += [
            f"--profile={b.index}",
            f"--minShapes={feat_name}:{min_bs}x{n_mels}x{b.frames_min},{len_name}:{min_bs}",
            f"--optShapes={feat_name}:{opt_bs}x{n_mels}x{b.frames_opt},{len_name}:{opt_bs}",
            f"--maxShapes={feat_name}:{max_bs}x{n_mels}x{b.frames_max},{len_name}:{max_bs}",
        ]
    return args


def precision_flags(precision: str, strongly_typed: bool, sparsity: bool,
                    trt_major: int = 10, onnx_dtype: str | None = None) -> list[str]:
    flags: list[str] = []
    if trt_major >= 11:
        # TensorRT 11 dropped weakly typed networks: --fp16/--bf16 are gone and every
        # network is strongly typed, i.e. the engine runs in the ONNX tensor types.
        # The requested precision must therefore already be the export dtype
        # (build.sh exports fp16 for GPU runtimes), otherwise the engine would
        # silently come out in another precision than the one asked for.
        if precision not in ("fp16", "fp32"):
            raise SystemExit(f"--precision {precision} needs a {precision} ONNX export on TensorRT >= 11")
        if onnx_dtype and onnx_dtype != precision:
            raise SystemExit(
                f"--precision {precision} but the ONNX was exported as {onnx_dtype}; on TensorRT >= 11 "
                "the engine precision is the ONNX dtype. Re-export or change --precision."
            )
        if sparsity:
            flags.append("--sparsity=enable")
        return flags
    if precision == "fp16":
        flags.append("--fp16")
    elif precision == "bf16":
        # Worth measuring against fp16 on Blackwell: sometimes the same speed at
        # better numerical stability (spec §6.2). Not a default until measured.
        flags.append("--bf16")
    elif precision != "fp32":
        raise SystemExit(f"unknown precision {precision}")

    if strongly_typed:
        # Only meaningful when the ONNX is already fp16: removes implicit casts.
        flags.append("--stronglyTyped")
    if sparsity:
        flags.append("--sparsity=enable")
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--out", required=True, help="path of the .plan to write")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--buckets", default=None, help='"2,5,10,20" or "none"')
    parser.add_argument("--meta", default=None,
                        help="meta.json (default: next to the ONNX)")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--opt-batch-size", type=int, default=4)
    parser.add_argument("--workspace-mib", type=int, default=8192)
    parser.add_argument("--timing-cache", default=None)
    parser.add_argument("--strongly-typed", action="store_true")
    parser.add_argument("--sparsity", action="store_true")
    parser.add_argument("--builder-optimization-level", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="rebuild even on a cache hit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"{onnx_path} not found -- run scripts/export_onnx.py first")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    key_path = out_path.with_suffix(out_path.suffix + ".key.json")

    timing_cache = Path(args.timing_cache or out_path.parent / "trt_timing.cache")
    timing_cache.parent.mkdir(parents=True, exist_ok=True)

    # Profile shapes are derived from the model's own feature geometry, never
    # assumed: v3 runs n_fft=320 with center=False, so the centred formula would
    # put the max shape two frames short and a request at exactly the bucket
    # limit would fall outside its profile.
    meta_path = Path(args.meta) if args.meta else onnx_path.parent / "meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"{meta_path} not found -- the profile shapes depend on the feature "
            "geometry recorded there. Re-run scripts/export_onnx.py."
        )
    meta = json.loads(meta_path.read_text())
    bucket_list = bucket_mod.from_meta(meta, args.buckets)
    n_mels = int(meta["n_mels"])

    trt_major = int(trt_version().split(".")[0])
    flags = precision_flags(args.precision, args.strongly_typed, args.sparsity,
                            trt_major, meta.get("export_dtype"))
    flags += [
        f"--builderOptimizationLevel={args.builder_optimization_level}",
        f"--memPoolSize=workspace:{args.workspace_mib}",
    ]
    graph = meta.get("graph", "ctc")
    if graph not in GRAPH_INPUTS:
        raise SystemExit(f"unknown graph type '{graph}' in {meta_path}")
    feat_name, len_name = GRAPH_INPUTS[graph]

    shapes = shape_args(
        bucket_list, n_mels, 1, args.opt_batch_size, args.max_batch_size,
        feat_name, len_name,
    )

    trt = trt_version()
    gpu = gpu_identity()
    onnx_sha = sha256_file(onnx_path)
    key = engine_key(onnx_sha, trt, gpu, flags + shapes)

    if out_path.exists() and key_path.exists() and not args.force:
        stored = json.loads(key_path.read_text())
        if stored.get("key") == key:
            print(f"engine cache hit -- {out_path} is current (key {key[:12]})")
            return 0
        # Say why, not just that. A driver bump and a flag change look identical
        # from the outside and need different responses.
        reasons = [
            f"  {field}: {stored.get(field)!r} -> {value!r}"
            for field, value in [
                ("onnx_sha256", onnx_sha), ("tensorrt_version", trt),
                ("gpu_name", gpu["gpu_name"]), ("compute_cap", gpu["compute_cap"]),
                ("driver_version", gpu["driver_version"]),
            ]
            if stored.get(field) != value
        ]
        if stored.get("flags") != flags + shapes:
            reasons.append("  build flags or bucket shapes changed")
        print("engine cache miss, rebuilding:")
        print("\n".join(reasons) or "  (key changed)")

    print(f"building {out_path.name}: TRT {trt}, {gpu['gpu_name']} sm{gpu['compute_cap']}, "
          f"{args.precision}, {len(bucket_list)} profile(s)")
    print(bucket_mod.describe(bucket_list))

    cmd = [
        find_trtexec(),
        f"--onnx={onnx_path}",
        f"--saveEngine={out_path}",
        f"--timingCacheFile={timing_cache}",
        *flags,
        *shapes,
    ]
    if args.verbose:
        cmd.append("--verbose")

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started

    log_path = out_path.with_suffix(out_path.suffix + ".log")
    log_path.write_text(proc.stdout + proc.stderr)

    if proc.returncode != 0 or "&&&& PASSED" not in proc.stdout:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-30:])
        print(f"\ntrtexec failed (rc {proc.returncode}); full log at {log_path}\n{tail}",
              file=sys.stderr)
        return 1

    key_path.write_text(
        json.dumps(
            {
                "key": key,
                "onnx": str(onnx_path),
                "onnx_sha256": onnx_sha,
                "tensorrt_version": trt,
                **gpu,
                "precision": args.precision,
                "flags": flags + shapes,
                "buckets_s": [b.max_s for b in bucket_list],
                "build_seconds": round(elapsed, 1),
            },
            indent=2,
        )
        + "\n"
    )

    size_mib = out_path.stat().st_size / (1 << 20)
    print(f"built in {elapsed:.0f}s -> {out_path} ({size_mib:.0f} MiB), key {key[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
