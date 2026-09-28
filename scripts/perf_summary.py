#!/usr/bin/env python3
"""Assemble the CTC-vs-RNNT comparison from the perf reports in cache/perf.

Reads whatever ``perf_profile.py`` has written rather than restating numbers by
hand, so the table cannot drift from the measurements it claims to summarise.
Missing configurations are printed as gaps instead of being quietly dropped.

  python scripts/perf_summary.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# label -> (report file, note)
CONFIGS = [
    ("CTC  GPU (TRT fp16)", "v3_e2e_ctc_trt_fp16.json", "encoder TensorRT, greedy decode"),
    ("RNNT GPU (TRT fp16)", "v3_e2e_rnnt_trt_fp16_cpuloop.json",
     "encoder TensorRT, decode loop on CPU"),
    ("CTC  CPU (ONNX fp32)", "v3_e2e_ctc_cpu_fp32.json", "6 instances x 4 threads"),
    ("RNNT CPU (ONNX fp32)", "v3_e2e_rnnt_cpu_fp32.json", "6 instances x 4 threads"),
]

POINTS = [(0.5, 1), (0.5, 8), (3.0, 1), (3.0, 8), (15.0, 1), (15.0, 8)]


def index(report: dict) -> dict:
    return {(m["duration_s"], m["concurrency"]): m for m in report.get("matrix", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-dir", default=None)
    parser.add_argument("--metric", default="p50_ms",
                        choices=["p50_ms", "p95_ms", "p99_ms", "throughput_rps", "rtfx"])
    args = parser.parse_args()

    perf_dir = Path(args.perf_dir or Path(__file__).resolve().parent.parent / "cache" / "perf")

    loaded = []
    for label, filename, note in CONFIGS:
        path = perf_dir / filename
        if not path.exists():
            print(f"MISSING: {label} -- no {filename}")
            continue
        loaded.append((label, note, index(json.loads(path.read_text()))))

    if not loaded:
        print("no reports found")
        return 1

    for metric in ("p50_ms", "p99_ms", "rtfx"):
        title = {"p50_ms": "p50 latency (ms)", "p99_ms": "p99 latency (ms)",
                 "rtfx": "RTFx (seconds of audio per second)"}[metric]
        print(f"\n{title}")
        header = f"{'config':<24}" + "".join(f"{f'{d:g}s/c{c}':>12}" for d, c in POINTS)
        print(header)
        print("-" * len(header))
        for label, _note, data in loaded:
            cells = []
            for point in POINTS:
                m = data.get(point)
                cells.append(f"{m[metric]:>12.1f}" if m else f"{'--':>12}")
            print(f"{label:<24}" + "".join(cells))

    print("\nratios, RNNT / CTC (lower is better for RNNT)")
    by_label = {label: data for label, _n, data in loaded}
    for gpu_cpu, ctc_key, rnnt_key in [
        ("GPU", "CTC  GPU (TRT fp16)", "RNNT GPU (TRT fp16)"),
        ("CPU", "CTC  CPU (ONNX fp32)", "RNNT CPU (ONNX fp32)"),
    ]:
        if ctc_key not in by_label or rnnt_key not in by_label:
            print(f"  {gpu_cpu}: incomplete, skipped")
            continue
        cells = []
        for point in POINTS:
            a, b = by_label[ctc_key].get(point), by_label[rnnt_key].get(point)
            cells.append(f"{b['p99_ms'] / a['p99_ms']:>11.2f}x" if a and b else f"{'--':>12}")
        print(f"  {gpu_cpu} p99{'':<15}" + "".join(cells))

    print("\nEvery number was taken on the staging RTX PRO 5000 while a vLLM server")
    print("held part of the GPU, on a CPU without AVX-512. Read as a lower bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
