#!/usr/bin/env python3
"""Latency / throughput matrix against a running server (spec §9.1).

Uses its own gRPC load generator rather than ``perf_analyzer``. Two reasons, both
practical: perf_analyzer ships in the ``-py3-sdk`` image, which is another ~20 GB
pull on a box that was already out of disk; and the matrix spans durations, which
here means different ensembles (one per bucket), while perf_analyzer drives one
model per run. If the binary is on PATH it is used for a cross-check instead.

Every result records the GPU's state at the time it was taken. On this host the
card is shared with a vLLM server holding ~27 GB, so these numbers are a lower
bound, not a verdict: a threshold met here will be met on the dedicated
production card, but one missed here has to be re-measured before it means
anything.

  python scripts/perf_profile.py --url localhost:18001 --repo /models \\
      --out /cache/perf/v3_e2e_ctc_trt_fp16.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000

DEFAULT_CONCURRENCY = [1, 4, 8, 16, 32, 50]
DEFAULT_DURATIONS_S = [0.5, 1.0, 3.0, 8.0, 15.0]

# Wall-clock budget for the warm samples behind each cold-vs-warm row.
WARM_BUDGET_S = 4.0


def gpu_snapshot() -> dict[str, Any]:
    """What else is on the card right now -- recorded with every measurement so a
    number can never be read without its context."""
    def q(query: str) -> list[str]:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
            )
            return [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
        except Exception:
            return []

    gpu = q("gpu=name,utilization.gpu,memory.used,memory.total")
    apps = q("compute-apps=pid,process_name,used_memory")

    parsed = {}
    if gpu:
        name, util, used, total = [p.strip() for p in gpu[0].split(",")]
        parsed = {
            "gpu_name": name,
            "utilization_pct": int(float(util)),
            "memory_used_mib": int(float(used)),
            "memory_total_mib": int(float(total)),
        }
    parsed["compute_apps"] = [
        dict(zip(("pid", "process", "memory_mib"), [p.strip() for p in row.split(",")]))
        for row in apps
    ]
    return parsed


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = min(len(ordered) - 1, max(0, int(round(p / 100 * (len(ordered) - 1)))))
        return ordered[idx]

    return {
        "p50_ms": round(pct(50) * 1000, 3),
        "p95_ms": round(pct(95) * 1000, 3),
        "p99_ms": round(pct(99) * 1000, 3),
        "min_ms": round(ordered[0] * 1000, 3),
        "max_ms": round(ordered[-1] * 1000, 3),
        "mean_ms": round(statistics.fmean(ordered) * 1000, 3),
    }


def pick_ensemble(manifest: dict, duration_s: float) -> dict | None:
    need = int(duration_s * SAMPLE_RATE)
    for entry in sorted(manifest["ensembles"], key=lambda e: e["pad_to_samples"]):
        if need <= entry["pad_to_samples"]:
            return entry
    return None


def make_payload(duration_s: float, pad_to: int):
    rng = np.random.default_rng(int(duration_s * 1000))
    n = int(duration_s * SAMPLE_RATE)
    audio = np.zeros(pad_to, dtype=np.float32)
    audio[:n] = (rng.standard_normal(n) * 0.05).astype(np.float32)
    return audio, n


def one_request(client, model: str, audio: np.ndarray, n_valid: int, outputs: list[str]):
    import tritonclient.grpc as gc

    a = gc.InferInput("audio", [1, audio.shape[0]], "FP32")
    a.set_data_from_numpy(audio[None, :])
    ln = gc.InferInput("audio_len", [1, 1], "INT32")
    ln.set_data_from_numpy(np.array([[n_valid]], dtype=np.int32))
    return client.infer(
        model, [a, ln], outputs=[gc.InferRequestedOutput(o) for o in outputs]
    )


def run_load(
    url: str, model: str, audio: np.ndarray, n_valid: int, concurrency: int,
    seconds: float, outputs: list[str],
) -> dict[str, Any]:
    """`concurrency` threads, each keeping exactly one request in flight."""
    import tritonclient.grpc as gc

    latencies: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    stop_at = time.perf_counter() + seconds

    def worker() -> None:
        client = gc.InferenceServerClient(url=url)      # one channel per thread
        local: list[float] = []
        local_err: list[str] = []
        while time.perf_counter() < stop_at:
            started = time.perf_counter()
            try:
                one_request(client, model, audio, n_valid, outputs)
                local.append(time.perf_counter() - started)
            except Exception as exc:
                local_err.append(str(exc).splitlines()[0][:120])
                if len(local_err) > 5:
                    break
        with lock:
            latencies.extend(local)
            errors.extend(local_err)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    mid_gpu = None
    time.sleep(min(seconds / 2, 2.0))
    mid_gpu = gpu_snapshot()                            # sampled under load
    for t in threads:
        t.join(timeout=seconds * 3)
    wall = time.perf_counter() - wall_start

    audio_seconds = len(latencies) * (n_valid / SAMPLE_RATE)
    return {
        "requests": len(latencies),
        "errors": len(errors),
        "error_samples": errors[:3],
        "wall_s": round(wall, 3),
        "throughput_rps": round(len(latencies) / wall, 2) if wall else 0.0,
        # RTFx: seconds of audio processed per second of wall clock.
        "rtfx": round(audio_seconds / wall, 1) if wall else 0.0,
        "gpu_under_load": mid_gpu,
        **percentiles(latencies),
    }


def measure_cold(client, manifest: dict, outputs: list[str]) -> dict:
    """First request to each bucket versus the ones right after.

    model_warmup is supposed to make these equal. If the first is several times
    slower, warmup did not do its job and the first production call of the day
    pays for it (spec §9.1).

    One throwaway request runs first, and its cost is reported separately. A
    fresh gRPC channel pays connection and stream setup on its first inference,
    and without this the whole of that lands on whichever bucket happens to be
    measured first: it showed up as a 4.9x "warmup did not take" on the 2 s
    bucket that vanished when the buckets were measured in the opposite order.
    """
    entries = sorted(manifest["ensembles"], key=lambda e: e["pad_to_samples"])

    handshake_entry = entries[-1]
    audio, n_valid = make_payload(
        handshake_entry["pad_to_samples"] / SAMPLE_RATE / 2,
        handshake_entry["pad_to_samples"],
    )
    started = time.perf_counter()
    one_request(client, handshake_entry["name"], audio, n_valid, outputs)
    channel_setup_ms = (time.perf_counter() - started) * 1000

    rows = []
    for entry in entries:
        duration = entry["pad_to_samples"] / SAMPLE_RATE / 2
        audio, n_valid = make_payload(duration, entry["pad_to_samples"])

        started = time.perf_counter()
        one_request(client, entry["name"], audio, n_valid, outputs)
        first = time.perf_counter() - started

        # Repeat count follows how slow the model actually is. A fixed 20 is
        # nothing on GPU and minutes per bucket on a CPU deployment, where a 20 s
        # clip takes seconds -- which is how this step came to outlast its own
        # timeout and report nothing at all.
        repeats = max(3, min(20, int(WARM_BUDGET_S / max(first, 1e-3))))
        warm = []
        for _ in range(repeats):
            started = time.perf_counter()
            one_request(client, entry["name"], audio, n_valid, outputs)
            warm.append(time.perf_counter() - started)

        median_warm = statistics.median(warm)
        rows.append(
            {
                "model": entry["name"],
                "first_request_ms": round(first * 1000, 3),
                "warm_median_ms": round(median_warm * 1000, 3),
                "ratio": round(first / median_warm, 2) if median_warm else None,
            }
        )
    return {
        "channel_setup_ms": round(channel_setup_ms, 3),
        "note": (
            "channel_setup_ms is the first inference on a fresh gRPC channel and "
            "includes connection setup; the per-bucket rows exclude it."
        ),
        "buckets": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:18001")
    parser.add_argument("--repo", default="/models")
    parser.add_argument("--out", default=None)
    parser.add_argument("--concurrency", default=",".join(map(str, DEFAULT_CONCURRENCY)))
    parser.add_argument("--durations", default=",".join(map(str, DEFAULT_DURATIONS_S)))
    parser.add_argument("--seconds-per-point", type=float, default=8.0)
    parser.add_argument(
        "--with-logprobs", action="store_true",
        help="also request the logprobs tensor, to price what it costs on the wire",
    )
    args = parser.parse_args()

    import tritonclient.grpc as gc

    manifest = json.loads((Path(args.repo) / "manifest.json").read_text())
    outputs = ["text"] + (["logprobs", "logprobs_len"] if args.with_logprobs else [])

    client = gc.InferenceServerClient(url=args.url)
    if not client.is_server_ready():
        print(f"server at {args.url} not ready", file=sys.stderr)
        return 1

    before = gpu_snapshot()
    shared = [a for a in before.get("compute_apps", []) if "triton" not in a["process"].lower()]
    print(f"GPU: {before.get('gpu_name')} "
          f"{before.get('memory_used_mib')}/{before.get('memory_total_mib')} MiB used")
    if shared:
        print("  sharing this GPU with: "
              + ", ".join(f"{a['process']} ({a['memory_mib']} MiB)" for a in shared))
        print("  -> treat every number below as a lower bound")

    print("\ncold vs warm (model_warmup should make these equal):")
    cold = measure_cold(client, manifest, outputs)
    print(f"  gRPC channel setup, charged to the first inference only: "
          f"{cold['channel_setup_ms']:.2f} ms")
    for row in cold["buckets"]:
        flag = "" if (row["ratio"] or 0) < 2 else "   <-- warmup did not take"
        print(f"  {row['model']:<32} first {row['first_request_ms']:>8.2f} ms   "
              f"warm {row['warm_median_ms']:>8.2f} ms   x{row['ratio']}{flag}")

    concurrency = [int(c) for c in args.concurrency.split(",")]
    durations = [float(d) for d in args.durations.split(",")]

    print(f"\nmatrix: {len(durations)} durations x {len(concurrency)} concurrency levels, "
          f"{args.seconds_per_point:g}s each\n")
    print(f"{'dur':>6} {'conc':>5} {'model':<30} {'p50':>9} {'p95':>9} {'p99':>9} "
          f"{'rps':>8} {'rtfx':>7} {'gpu%':>5}")

    results = []
    for duration in durations:
        entry = pick_ensemble(manifest, duration)
        if entry is None:
            print(f"{duration:>6} -- exceeds the largest bucket, skipped")
            continue
        audio, n_valid = make_payload(duration, entry["pad_to_samples"])
        for conc in concurrency:
            row = run_load(
                args.url, entry["name"], audio, n_valid, conc,
                args.seconds_per_point, outputs,
            )
            row.update({"duration_s": duration, "concurrency": conc,
                        "model": entry["name"], "bucket_s": entry["bucket_s"]})
            results.append(row)
            util = (row.get("gpu_under_load") or {}).get("utilization_pct", "?")
            print(f"{duration:>6} {conc:>5} {entry['name']:<30} "
                  f"{row.get('p50_ms', 0):>9.2f} {row.get('p95_ms', 0):>9.2f} "
                  f"{row.get('p99_ms', 0):>9.2f} {row['throughput_rps']:>8.1f} "
                  f"{row['rtfx']:>7.1f} {util:>5}"
                  + (f"   errors={row['errors']}" if row["errors"] else ""))

    report = {
        "variant": manifest["variant"],
        "runtime": manifest["runtime"],
        "export_dtype": manifest["export_dtype"],
        "max_batch_size": manifest["max_batch_size"],
        "requested_outputs": outputs,
        "gpu_before": before,
        "gpu_after": gpu_snapshot(),
        "gpu_shared_with": shared,
        "caveat": (
            "Measured on the staging RTX PRO 5000 while a vLLM server held part of "
            "the card. Thresholds in spec §9.2 were written for the dedicated "
            "production RTX PRO 6000; read these as a lower bound."
        ),
        "cold_vs_warm": cold,
        "matrix": results,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
