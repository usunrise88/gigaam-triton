#!/usr/bin/env python3
"""Detect the GPU and pick a Triton base image that can actually build a
TensorRT engine for it. Writes cache/env.json (spec §4.1).

Why this exists: the upstream GigaAM instructions pin
``nvcr.io/nvidia/tensorrt:24.10-py3``, which predates SM120. On a Blackwell RTX
PRO the engine build either fails or silently degrades. Hardcoding a version is
forbidden, so we probe: for each candidate image, freshest first, actually build
a small Conformer-shaped engine for the detected compute capability and take the
first image that succeeds.

The chosen image is used for BOTH the builder and the runtime. If they diverge,
the TRT version baked into the .plan won't match the server and Triton refuses to
load it.

On total failure this exits non-zero and prints what was tried. It never falls
back to CPU quietly -- a silent CPU fallback in an ASR server is a latency
regression nobody notices until production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Freshest first. Anything not present locally is skipped unless --pull is given,
# because a Triton image is ~30 GB and this script should not silently fill a disk.
DEFAULT_CANDIDATES = [
    "nvcr.io/nvidia/tritonserver:25.06-py3",
]

PROBE_FLAGS = [
    "--fp16",
    "--builderOptimizationLevel=5",
    "--minShapes=features:1x64x32",
    "--optShapes=features:4x64x300",
    "--maxShapes=features:8x64x2000",
    "--memPoolSize=workspace:4096",
]

# Substrings in the trtexec log that mean "it built, but not for this GPU".
# A build can report success while having fallen back to a generic/JIT path, and
# that is exactly the degradation §4 warns about.
FALLBACK_MARKERS = [
    "falling back to",
    "no kernel image is available",
    "PTX JIT",
    "compiled for a different",
]

# Emitted by the in-container script so we can separate our JSON from trtexec noise.
JSON_MARKER = "###ENVJSON###"

CONTAINER_SCRIPT = r"""
set -uo pipefail

TRTEXEC=""
for c in "$(command -v trtexec 2>/dev/null)" /usr/src/tensorrt/bin/trtexec; do
  if [ -n "$c" ] && [ -x "$c" ]; then TRTEXEC="$c"; break; fi
done
if [ -z "$TRTEXEC" ]; then echo "PROBE_FAIL: trtexec not found in image"; exit 3; fi

if [ ! -f /w/probe.onnx ]; then
  pip install --quiet --no-cache-dir onnx 'numpy<3' >/tmp/pip.log 2>&1
  if [ $? -ne 0 ]; then echo "PROBE_FAIL: cannot install onnx to generate probe"; tail -5 /tmp/pip.log; exit 4; fi
  python3 /s/gen_probe_onnx.py /w/probe.onnx >/dev/null 2>&1
  if [ $? -ne 0 ]; then echo "PROBE_FAIL: probe onnx generation failed"; exit 5; fi
fi

"$TRTEXEC" --onnx=/w/probe.onnx --saveEngine=/tmp/probe.plan __FLAGS__ >/tmp/trtexec.log 2>&1
BUILD_RC=$?

echo "###TRTEXECLOG###"
cat /tmp/trtexec.log

echo "__JSON_MARKER__"
python3 - "$TRTEXEC" "$BUILD_RC" <<'PY'
import ctypes, json, os, subprocess, sys

trtexec, build_rc = sys.argv[1], int(sys.argv[2])

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ""

def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""

trt_int = None
try:
    lib = ctypes.CDLL("libnvinfer.so")
    lib.getInferLibVersion.restype = ctypes.c_int
    trt_int = int(lib.getInferLibVersion())
except Exception:
    pass

# 101100 -> "10.11.0"
trt_str = None
if trt_int:
    major, rest = divmod(trt_int, 10000)
    minor, patch = divmod(rest, 100)
    trt_str = f"{major}.{minor}.{patch}"

nvcc = sh("nvcc --version | tail -2 | head -1")
cuda_toolkit = ""
if "release" in nvcc:
    cuda_toolkit = nvcc.split("release", 1)[1].split(",")[0].strip()

backends = []
try:
    backends = sorted(os.listdir("/opt/tritonserver/backends"))
except OSError:
    pass

def importable(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False

print(json.dumps({
    "build_rc": build_rc,
    "trtexec_path": trtexec,
    "tensorrt_lib_version_int": trt_int,
    "tensorrt_version": trt_str,
    "cuda_toolkit": cuda_toolkit,
    "triton_version": read("/opt/tritonserver/TRITON_VERSION"),
    "python": sys.version.split()[0],
    "backends": backends,
    "preinstalled_python_pkgs": {
        "tensorrt": importable("tensorrt"),
        "onnxruntime": importable("onnxruntime"),
        "torch": importable("torch"),
    },
}))
PY
"""


def run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def query_gpu() -> dict[str, Any]:
    """Read GPU identity from nvidia-smi. Everything in the engine cache key
    (spec §6.3) comes from here, so a wrong value silently invalidates engines."""
    if not shutil.which("nvidia-smi"):
        sys.exit("nvidia-smi not found -- cannot detect GPU. Refusing to guess.")

    fields = "name,compute_cap,driver_version,memory.total"
    proc = run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if proc.returncode != 0:
        sys.exit(f"nvidia-smi failed:\n{proc.stderr}")

    first = proc.stdout.strip().splitlines()[0]
    name, compute_cap, driver, vram = [p.strip() for p in first.split(",")]

    cuda_driver = ""
    smi = run(["nvidia-smi"]).stdout
    if "CUDA Version:" in smi:
        cuda_driver = smi.split("CUDA Version:", 1)[1].split()[0].strip()

    return {
        "gpu_name": name,
        "compute_cap": compute_cap,
        "driver_version": driver,
        "cuda_driver_version": cuda_driver,
        "vram_total_mib": int(float(vram)),
    }


def image_present(image: str) -> bool:
    return run(["docker", "image", "inspect", image], timeout=120).returncode == 0


def probe_image(image: str, cache_dir: Path, scripts_dir: Path) -> dict[str, Any]:
    """Build the probe engine inside `image`. Returns a result dict; `ok` says
    whether this image is usable for the detected GPU."""
    script = CONTAINER_SCRIPT.replace("__FLAGS__", " ".join(PROBE_FLAGS)).replace(
        "__JSON_MARKER__", JSON_MARKER
    )

    proc = run(
        [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{cache_dir}:/w",
            "-v", f"{scripts_dir}:/s:ro",
            image, "bash", "-c", script,
        ]
    )
    out = proc.stdout

    result: dict[str, Any] = {"image": image, "ok": False}

    if JSON_MARKER not in out:
        result["reason"] = "probe did not reach the version report"
        result["detail"] = (out + proc.stderr).strip()[-1500:]
        return result

    log_part, json_part = out.split(JSON_MARKER, 1)
    trtexec_log = log_part.split("###TRTEXECLOG###", 1)[-1]

    try:
        info = json.loads(json_part.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        result["reason"] = f"could not parse version report: {exc}"
        return result

    result.update(info)

    if info.get("build_rc") != 0 or "&&&& PASSED" not in trtexec_log:
        result["reason"] = "trtexec failed to build the probe engine"
        result["detail"] = trtexec_log.strip()[-1500:]
        return result

    hits = [m for m in FALLBACK_MARKERS if m.lower() in trtexec_log.lower()]
    if hits:
        result["reason"] = f"engine built but log shows fallback: {hits}"
        result["detail"] = trtexec_log.strip()[-1500:]
        return result

    result["ok"] = True
    result["throughput_qps"] = _grep_float(trtexec_log, "Throughput:")
    return result


def _grep_float(text: str, needle: str) -> float | None:
    for line in text.splitlines():
        if needle in line:
            for token in line.split(needle, 1)[1].split():
                try:
                    return float(token)
                except ValueError:
                    continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default=",".join(DEFAULT_CANDIDATES),
        help="comma-separated Triton images, freshest first",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="pull candidates that are not present locally (~30 GB each)",
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "cache" / "env.json"))
    args = parser.parse_args()

    cache_dir = Path(args.out).resolve().parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(__file__).resolve().parent

    gpu = query_gpu()
    print(
        f"GPU: {gpu['gpu_name']} (compute_cap {gpu['compute_cap']}, "
        f"driver {gpu['driver_version']}, {gpu['vram_total_mib']} MiB)"
    )

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    attempts: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None

    for image in candidates:
        if not image_present(image):
            if not args.pull:
                print(f"skip {image}: not present locally (pass --pull to fetch)")
                attempts.append(
                    {"image": image, "ok": False, "reason": "not present locally, --pull not given"}
                )
                continue
            print(f"pulling {image} ...")
            if run(["docker", "pull", image], timeout=7200).returncode != 0:
                attempts.append({"image": image, "ok": False, "reason": "docker pull failed"})
                continue

        print(f"probing {image} ...")
        result = probe_image(image, cache_dir, scripts_dir)
        attempts.append(result)
        if result["ok"]:
            chosen = result
            print(f"  PASSED (TensorRT {result.get('tensorrt_version')})")
            break
        print(f"  rejected: {result.get('reason')}")

    if chosen is None:
        print("\nNo candidate image can build a TensorRT engine for this GPU.", file=sys.stderr)
        print("Tried:", file=sys.stderr)
        for a in attempts:
            print(f"  - {a['image']}: {a.get('reason', 'unknown')}", file=sys.stderr)
            if a.get("detail"):
                print(f"      {a['detail'].splitlines()[-1]}", file=sys.stderr)
        print(
            "\nNot falling back to CPU. Add a newer image via --candidates.",
            file=sys.stderr,
        )
        return 1

    probe_onnx = cache_dir / "probe.onnx"
    probe_sha = (
        hashlib.sha256(probe_onnx.read_bytes()).hexdigest() if probe_onnx.exists() else None
    )

    env = {
        "probed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": gpu,
        "selected_image": {
            "image": chosen["image"],
            "triton_version": chosen.get("triton_version"),
            "tensorrt_version": chosen.get("tensorrt_version"),
            "tensorrt_lib_version_int": chosen.get("tensorrt_lib_version_int"),
            "cuda_toolkit": chosen.get("cuda_toolkit"),
            "python": chosen.get("python"),
            "trtexec_path": chosen.get("trtexec_path"),
            "backends": chosen.get("backends"),
            "preinstalled_python_pkgs": chosen.get("preinstalled_python_pkgs"),
        },
        "sm_build_probe": {
            "status": "PASSED",
            "compute_cap": gpu["compute_cap"],
            "onnx": "scripts/gen_probe_onnx.py",
            "onnx_sha256": probe_sha,
            "flags": PROBE_FLAGS,
            "throughput_qps": chosen.get("throughput_qps"),
        },
        "attempts": [
            {k: v for k, v in a.items() if k != "detail"} for a in attempts
        ],
    }

    Path(args.out).write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
