#!/usr/bin/env python3
"""Generate the Triton model repository from templates (spec §7).

Deliberately not the upstream layout. ``triton_scripts/repos/*`` sets
``max_batch_size: 0`` throughout, which means Triton never batches anything and
the client has to pack a ragged tensor itself. Spec §7.3 wants Triton's own
dynamic batching, and that needs a real batch dimension in every model of the
ensemble -- so the whole repository is shaped differently, not patched.

One ensemble per duration bucket. Triton binds a TensorRT optimization profile to
an instance rather than selecting it from the request shape, so "four profiles in
one model" cannot work; four models can. The provider already pads to a bucket
(§6.1) and picks the matching name.

  python scripts/gen_model_repo.py --onnx-root /work/onnx --variant v3_e2e_ctc \\
      --runtime trt --engine /work/cache/engines/v3_e2e_ctc.plan --out /work/model_repo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import buckets as bucket_mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# cuda_graphs defaults to off even for TensorRT, and that is a measured decision
# rather than caution. Across the whole §9.1 matrix graphs made no difference
# outside run-to-run noise (3 s bucket, concurrency 1/8/50: 20.4/39.5/47.4 ms
# with, 19.5/39.7/48.3 ms without), while capture failed on some instances with
# "unable to record CUDA graph". The encoder spends 4-7 ms in GPU work, so the
# launch overhead graphs remove is not what this path is bound by. Spec §6.2
# asks for exactly this: switch it on by result, not by default. Override with
# --cuda-graphs on.
RUNTIMES = {
    #                 platform             kind        cuda_graphs  ort_threads
    "trt":       ("tensorrt_plan",    "KIND_GPU", False, False),
    "onnx-gpu":  ("onnxruntime_onnx", "KIND_GPU", False, False),
    "onnx-cpu":  ("onnxruntime_onnx", "KIND_CPU", False, True),
}

DTYPE_TO_TRITON = {"fp16": "TYPE_FP16", "fp32": "TYPE_FP32"}


def render(template_dir: Path, template: str, **ctx: Any) -> str:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,       # a missing variable is a bug, not a blank
        trim_blocks=False,
        keep_trailing_newline=True,
    )
    return env.get_template(template).render(**ctx)


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink the model weights instead of copying them per bucket.

    Four buckets sharing one 200 MB plan is 800 MB of identical bytes otherwise.
    Hardlinks rather than symlinks because the repository gets bind-mounted into
    a container where an absolute symlink target would not resolve.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_length_file(model_dir: Path, frames: int) -> str:
    """Warmup needs a real frame count; Triton reads it as raw bytes."""
    name = f"feature_lengths_{frames}"
    warmup_dir = model_dir / "warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    (warmup_dir / name).write_bytes(struct.pack("<q", frames))
    return name


def enforce_single_family(out_root: Path, family: str) -> None:
    """v2 and v3 feature extraction differ, and the ensemble shares one
    preprocessing model -- whichever variant was converted last would silently
    win (spec §11.3). Refuse the mix here as well as at export time."""
    marker = out_root / ".repo_family"
    if marker.exists():
        existing = marker.read_text().strip()
        if existing != family:
            raise SystemExit(
                f"model repository already holds family '{existing}', refusing to "
                f"add '{family}'.\n"
                "They cannot share one preprocessing model (spec §11.3). Generate "
                "a separate repository, or clear this one first."
            )
    else:
        out_root.mkdir(parents=True, exist_ok=True)
        marker.write_text(family + "\n")


def copy_python_backend(
    src_backend: Path, model_dir: Path, common_dir: Path, artifacts: list[Path]
) -> None:
    version_dir = model_dir / "1"
    if version_dir.exists():
        shutil.rmtree(version_dir)
    shutil.copytree(src_backend / "1", version_dir)

    # common/ lives inside the version directory: the backend puts its own
    # directory on sys.path, and Triton only guarantees the version dir is ours.
    dst_common = version_dir / "common"
    shutil.copytree(common_dir, dst_common, ignore=shutil.ignore_patterns("__pycache__"))

    for artifact in artifacts:
        if artifact.exists():
            shutil.copy2(artifact, version_dir / artifact.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-root", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--runtime", choices=sorted(RUNTIMES), default="trt")
    parser.add_argument("--buckets", default=None, help='e.g. "2,5,10,20" or "none"')
    parser.add_argument("--engine", default=None, help="path to the .plan (runtime=trt)")
    parser.add_argument(
        "--weights-name", default=None,
        help="ONNX file inside the export dir (default model.onnx; "
             "model.int8.onnx for the quantised CPU profile)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--cpu-instances", type=int, default=4)
    parser.add_argument("--preferred-batch-size", default="2,4,8")
    parser.add_argument("--max-queue-delay-us", type=int, default=1000)
    parser.add_argument("--warmup-count", type=int, default=3)
    parser.add_argument("--intra-op-threads", type=int, default=4)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    # Spec §6.2 lists CUDA graphs as something to switch on by measurement, not
    # by default. Capture also fails on some instances here ("unable to record
    # CUDA graph"), so being able to turn it off and compare is the point.
    parser.add_argument(
        "--cuda-graphs", choices=["auto", "on", "off"], default="auto",
        help="auto = on for TensorRT, off otherwise",
    )
    # RNNT decode loop (spec §7.4). Which provider wins is not obvious -- the
    # decoder is a 320-wide LSTM and the joint two small matmuls, so launch
    # overhead can outweigh the arithmetic. Both are measurable from here.
    # Default measured, not assumed: on this host the loop runs 127 ms on CPU
    # against 198 ms on CUDA for the same 11 s clip. The decoder is a 320-wide
    # LSTM and the joint two small matmuls, so per-step launch and
    # synchronisation cost more than the arithmetic they guard.
    parser.add_argument("--rnnt-provider", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--rnnt-instances", type=int, default=4)
    parser.add_argument("--rnnt-intra-op-threads", type=int, default=1)
    parser.add_argument("--max-symbols-per-frame", type=int, default=3)
    parser.add_argument("--templates", default=str(REPO_ROOT / "templates" / "config"))
    parser.add_argument("--backends", default=str(REPO_ROOT / "backends"))
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_root) / args.variant
    meta_path = onnx_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found -- run scripts/export_onnx.py first")
    meta = json.loads(meta_path.read_text())

    graph = meta["graph"]
    if graph not in ("ctc", "rnnt"):
        raise SystemExit(f"unknown graph type '{graph}' in {meta_path}")

    out_root = Path(args.out)
    enforce_single_family(out_root, meta["family"])

    platform, instance_kind, cuda_graphs, ort_threads = RUNTIMES[args.runtime]
    if args.cuda_graphs != "auto":
        cuda_graphs = args.cuda_graphs == "on"
    feature_dtype = DTYPE_TO_TRITON[meta["export_dtype"]]
    # Frame counts come from the model's own feature geometry in meta.json. v3
    # is not on the library defaults (n_fft=320, center=False), so assuming them
    # would put every warmup and profile shape two frames off.
    bucket_list = bucket_mod.from_meta(meta, args.buckets)
    preferred = [int(x) for x in args.preferred_batch_size.split(",")]

    templates = Path(args.templates)
    backends = Path(args.backends)
    common_dir = backends / "common"

    prep_name = "preprocessing"
    is_rnnt = graph == "rnnt"
    post_name = "rnnt_decoder_joint" if is_rnnt else "ctc_postprocessing"

    shared = dict(
        max_batch_size=args.max_batch_size,
        preferred_batch_size=preferred,
        max_queue_delay_us=args.max_queue_delay_us,
        n_mels=meta["n_mels"],
        vocab_size=meta["vocab_size"],
        # The CTC head is one class wider than the tokenizer (blank at the end).
        num_classes=meta.get("num_classes", meta["vocab_size"] + 1),
        feature_dtype=feature_dtype,
        enc_hidden=meta.get("enc_hidden"),
    )

    # ---- preprocessing -------------------------------------------------
    prep_dir = out_root / prep_name
    filterbank = Path(args.onnx_root) / "preprocessing" / meta["family"] / "filterbank.npz"
    if not filterbank.exists():
        raise SystemExit(f"{filterbank} missing -- rerun scripts/export_onnx.py")
    copy_python_backend(
        backends / "preprocessing", prep_dir, common_dir, [filterbank, meta_path]
    )
    (prep_dir / "config.pbtxt").write_text(
        render(templates, "preprocessing.pbtxt.j2",
               name=prep_name, cpu_instances=args.cpu_instances,
               largest_bucket_samples=bucket_list[-1].samples_max, **shared)
    )

    # ---- postprocessing / decode loop -----------------------------------
    post_dir = out_root / post_name
    artifacts = [meta_path, onnx_dir / "vocab.json"]
    if meta.get("tokenizer_model"):
        artifacts.append(onnx_dir / meta["tokenizer_model"])
    if is_rnnt:
        # The decode loop needs the decoder and joint graphs beside it; only the
        # encoder goes to TensorRT (spec §7.4).
        artifacts += [onnx_dir / "decoder.onnx", onnx_dir / "joint.onnx"]

    copy_python_backend(backends / post_name, post_dir, common_dir, artifacts)
    (post_dir / "config.pbtxt").write_text(
        render(
            templates,
            "rnnt_decoder_joint.pbtxt.j2" if is_rnnt else "ctc_postprocessing.pbtxt.j2",
            name=post_name,
            cpu_instances=args.cpu_instances,
            rnnt_provider=args.rnnt_provider,
            rnnt_instances=args.rnnt_instances,
            rnnt_kind="KIND_GPU" if args.rnnt_provider == "cuda" else "KIND_CPU",
            rnnt_intra_op_threads=args.rnnt_intra_op_threads,
            max_symbols_per_frame=args.max_symbols_per_frame,
            **shared,
        )
    )

    # ---- encoder(s) ----------------------------------------------------
    # TensorRT needs one model per bucket so each can pin its own optimization
    # profile. ONNX Runtime resolves dynamic shapes by itself, so a second copy
    # would only duplicate the session and its GPU memory -- one model, warmed on
    # every bucket shape, is the same thing for less.
    per_bucket_encoder = platform == "tensorrt_plan"

    if platform == "tensorrt_plan":
        if not args.engine:
            raise SystemExit("--engine is required for --runtime trt")
        weights_src, weights_name = Path(args.engine), "model.plan"
    else:
        default_name = "encoder.onnx" if is_rnnt else "model.onnx"
        source_name = args.weights_name or default_name
        weights_src, weights_name = onnx_dir / source_name, "model.onnx"
    if not weights_src.exists():
        raise SystemExit(
            f"{weights_src} not found"
            + (" -- run scripts/quantize_onnx.py first" if "int8" in str(weights_src) else "")
        )

    encoder_for_bucket: dict[int, str] = {}

    def emit_encoder(name: str, warmups: list[dict], profile_index: int | None) -> None:
        model_dir = out_root / name
        link_or_copy(weights_src, model_dir / "1" / weights_name)
        for w in warmups:
            w["length_file"] = write_length_file(model_dir, w["frames"])
        (model_dir / "config.pbtxt").write_text(
            render(
                templates,
                "rnnt_encoder.pbtxt.j2" if is_rnnt else "ctc_encoder.pbtxt.j2",
                name=name,
                platform=platform,
                instance_kind=instance_kind,
                instances=args.instances if instance_kind == "KIND_GPU" else args.cpu_instances,
                profile_index=profile_index,
                cuda_graphs=cuda_graphs,
                intra_op_threads=args.intra_op_threads if ort_threads else None,
                inter_op_threads=args.inter_op_threads if ort_threads else None,
                warmup_count=args.warmup_count,
                warmups=warmups,
                **shared,
            )
        )

    if per_bucket_encoder:
        for b in bucket_list:
            name = f"{args.variant}_encoder_{b.label}s"
            emit_encoder(
                name,
                [
                    {"name": f"warmup_{b.label}s_b1", "batch_size": 1, "frames": b.frames_max},
                    {"name": f"warmup_{b.label}s_bmax", "batch_size": args.max_batch_size,
                     "frames": b.frames_max},
                ],
                profile_index=b.index,
            )
            encoder_for_bucket[b.index] = name
    else:
        name = f"{args.variant}_encoder"
        warmups = []
        for b in bucket_list:
            warmups.append(
                {"name": f"warmup_{b.label}s_b1", "batch_size": 1, "frames": b.frames_max}
            )
        warmups.append(
            {"name": "warmup_bmax", "batch_size": args.max_batch_size,
             "frames": bucket_list[-1].frames_max}
        )
        emit_encoder(name, warmups, profile_index=None)
        encoder_for_bucket = {b.index: name for b in bucket_list}

    # ---- ensembles -----------------------------------------------------
    ensembles = []
    for b in bucket_list:
        name = f"gigaam_{args.variant}_{b.label}s"
        ens_dir = out_root / name
        (ens_dir / "1").mkdir(parents=True, exist_ok=True)   # Triton wants the dir
        (ens_dir / "config.pbtxt").write_text(
            render(
                templates,
                "rnnt_ensemble.pbtxt.j2" if is_rnnt else "ctc_ensemble.pbtxt.j2",
                name=name,
                bucket_s=b.label,
                preprocessing_model=prep_name,
                encoder_model=encoder_for_bucket[b.index],
                postprocessing_model=post_name,
                decoder_model=post_name,
                **shared,
            )
        )
        ensembles.append(
            {
                "name": name,
                "bucket_s": b.max_s,
                "pad_to_samples": b.samples_max,
                "frames": b.frames_max,
                "encoder": encoder_for_bucket[b.index],
            }
        )

    manifest = {
        "variant": args.variant,
        "runtime": args.runtime,
        "platform": platform,
        "family": meta["family"],
        "graph": graph,
        "export_dtype": meta["export_dtype"],
        "vocab_size": meta["vocab_size"],
        "num_classes": shared["num_classes"],
        "blank_id": meta["blank_id"],
        "max_batch_size": args.max_batch_size,
        # The provider reads this to map a padded duration to an ensemble name.
        "ensembles": ensembles,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"model repository -> {out_root}")
    print(bucket_mod.describe(bucket_list))
    for e in ensembles:
        print(f"  {e['name']:<36} pad to {e['pad_to_samples']:>7} samples -> {e['encoder']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
