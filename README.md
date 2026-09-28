# gigaam-triton

**English** · [Русский](README.ru.md)

Serves the Russian speech recognition model [GigaAM v3](https://github.com/salute-developers/GigaAM)
on NVIDIA Triton Inference Server, using TensorRT on Blackwell GPUs (SM120) and ONNX Runtime on CPU.

- All four v3 variants: `v3_e2e_ctc`, `v3_e2e_rnnt`, `v3_ctc`, `v3_rnnt`.
- Three runtimes: TensorRT, ONNX Runtime on GPU, and ONNX Runtime on CPU with dynamic int8.
- Every variant and runtime has the same input: raw 16 kHz float32 samples. The output includes
  text, tokens and per-token scores; CTC variants also return full log-probabilities, so clients
  can run their own decoding.
- The runtime image has no torch. Feature extraction is a numpy re-implementation that must
  match torchaudio before the build can pass.
- Bad input fails loudly. On int16-scaled audio the model itself silently returns an empty
  string, so the server validates dtype and range and returns an explicit error instead.

| Document | Contents |
|---|---|
| [`INTEGRATION.md`](INTEGRATION.md) | Client contract: tensors, ports, errors, latency figures (in Russian) |
| [`PREP.md`](PREP.md) | Development log: environment, upstream findings, measurements, decisions (in Russian) |
| [`gigaam-triton-spec.md`](gigaam-triton-spec.md) | Original design specification (in Russian) |

---

## Contents

1. [Host requirements](#1-host-requirements)
2. [Quick start](#2-quick-start)
3. [Build options](#3-build-options)
4. [Verification and benchmarks](#4-verification-and-benchmarks)
5. [CPU profile](#5-cpu-profile)
6. [RNNT](#6-rnnt)
7. [Evaluation](#7-evaluation)
8. [Repository layout](#8-repository-layout)
9. [Things to know up front](#9-things-to-know-up-front)

---

## 1. Host requirements

Everything heavy runs in containers. The host needs only:

| Requirement | Check |
|---|---|
| NVIDIA GPU + driver | `nvidia-smi` |
| Docker ≥ 24 | `docker --version` |
| NVIDIA Container Toolkit | `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` |
| Python 3.8+ (standard library only) | `python3 --version` |
| ~60 GB of free disk space | `df -h /` |
| Internet access | NVIDIA NGC images, PyPI, GigaAM weights CDN |

Building needs **no pip packages on the host**: `build.sh` and `scripts/detect_env.py` use only
the standard library. The evaluation module is the one exception (see [§7](#7-evaluation)).

Disk usage: the Triton base image is ~31 GB, the builder image ~44 GB and the runtime image
~35 GB (they share layers, so the real total is smaller). ONNX exports take ~2 GB, and each
variant's TensorRT engine takes ~0.8 GB. If space runs short, `docker builder prune` usually
frees tens of gigabytes.

---

## 2. Quick start

```bash
git clone https://github.com/usunrise88/gigaam-triton.git && cd gigaam-triton
./build.sh --variant v3_e2e_ctc --runtime trt --precision fp16 \
           --buckets 2,5,10,20 --verify
```

The first run does everything:

1. picks a base image that works on this GPU;
2. builds the builder image;
3. checks the feature frontend against torchaudio;
4. downloads the weights (~1.7 GB, cached in `~/.cache/gigaam`);
5. exports ONNX and builds the TensorRT engine;
6. builds the runtime image.

With `--verify` it also starts the server and runs the smoke test.

**Expect about an hour**, ~40 minutes of which is the TensorRT engine build. Re-runs are
idempotent: nothing with a valid cache key is rebuilt.

Then start the server:

```bash
docker run -d --gpus all --name gigaam-triton \
  -p 18000:18000 -p 18001:18001 -p 18002:18002 \
  -v "$PWD/cache:/cache" \
  -e VARIANT=v3_e2e_ctc -e RUNTIME=trt -e PRECISION=fp16 \
  gigaam-triton:0.1

curl -sf http://localhost:18000/v2/health/ready && echo ready
```

Ports 18000/18001/18002 are HTTP, gRPC and metrics. Client usage is described in
[`INTEGRATION.md`](INTEGRATION.md).

If the container is started with different `BUCKETS` or `PRECISION`, the entrypoint notices the
engine mismatch and rebuilds the engine on startup. Budget time for that.

---

## 3. Build options

| Flag | Values | Default |
|---|---|---|
| `--variant` | `v3_e2e_ctc`, `v3_e2e_rnnt`, `v3_ctc`, `v3_rnnt`; comma-separated list allowed | `v3_e2e_ctc` |
| `--runtime` | `trt`, `onnx-gpu`, `onnx-cpu` | `trt` |
| `--precision` | `fp16`, `bf16`, `fp32`, `int8` (`onnx-cpu` only) | `fp16` on GPU, `int8` on CPU |
| `--buckets` | comma-separated duration bounds in seconds, or `none` | `2,5,10,20` |
| `--max-batch-size` | | `8` |
| `--instances` | encoder instances per GPU | `2` |
| `--cpu-instances` / `--intra-op-threads` | CPU profile | `6` / `4` |
| `--image` / `--model-repo` / `--cache` | image tag and paths | `gigaam-triton:0.1`, `./model_repo`, `./cache` |
| `--skip-trt` | build the engine on first container start instead | off |
| `--verify` | start the server and run the smoke test | off |
| `--force` | rebuild everything, ignoring the cache | off |
| `--pull` | allow pulling candidate base images (~30 GB each) | off |

### Build steps

1. **Environment.** `detect_env.py` reads `nvidia-smi`, goes through candidate Triton images and,
   for each one, **builds a real probe engine** for the detected compute capability. The first
   image that passes is written to `cache/env.json`. No image version is hard-coded: upstream's
   instructions pin a TensorRT release that predates SM120 support, and on Blackwell that either
   crashes or silently falls back to slow JIT. If every candidate fails, the script stops with the
   list of what it tried. It does **not** silently fall back to CPU.
2. **Builder image:** torch, `gigaam` and the ONNX tooling. It is never shipped.
3. **Log-mel gate.** The runtime has no torch, so the frontend is recomputed in numpy using a
   filterbank exported from torchaudio. The gate compares both against a float64 reference, and
   the build fails if they diverge. A silent mismatch here shows up as a WER regression that
   people then go looking for in the model.
4. **ONNX export**, validated against torch on the golden set (CER threshold 0.5 %).
5. **TensorRT engine:** a single `.plan` with one optimization profile per duration bucket.
6. **Runtime image:** slim, with no torch and no `gigaam` package.

Artifacts go to `artifacts/<precision>/`. Engines and reports go to `cache/`.

---

## 4. Verification and benchmarks

```bash
# Backend unit checks: CTC collapse, tokenizer, log-mel
docker run --rm -v "$PWD/artifacts/fp16:/work/onnx" gigaam-triton-builder:0.1 \
  python3 scripts/test_backends.py --onnx-root /work/onnx --variant v3_e2e_ctc

# End-to-end smoke test against a running server
docker exec gigaam-triton python3 /opt/gigaam/scripts/smoke_test.py \
  --url localhost:18001 --repo /models --golden-dir /testdata/golden

# Latency / throughput matrix
docker exec gigaam-triton python3 /opt/gigaam/scripts/perf_profile.py \
  --url localhost:18001 --repo /models --out /cache/perf/report.json

# Summary table across all collected reports
python3 scripts/perf_summary.py
```

The smoke test covers three things a green build does not guarantee:

- it warms up the ensemble (Triton cannot do that by itself);
- it compares the transcript with the torch reference;
- it checks that **bad input fails loudly**: int16-scaled samples, NaN and a broken length.

Tune these runtime variables to your load instead of keeping the defaults:

| Variable | Default | Notes |
|---|---|---|
| `INSTANCES` | `2` | A second encoder instance paid off only at ~32 concurrent requests in our measurements. |
| `MAX_QUEUE_DELAY_US` | `1000` | With a single client stream this is pure added latency, paid at every ensemble step. |
| `WARMUP_COUNT` | `3` | Warm-up on CPU can take minutes. |

---

## 5. CPU profile

```bash
./build.sh --variant v3_e2e_ctc --runtime onnx-cpu --precision int8
```

Quantization is dynamic, so no calibration set is needed. The script detects CPU capabilities
and prints quality before and after quantization, because a speed figure means nothing without
the quality figure next to it.

Two caveats:

- **Without VNNI, int8 gives no speedup.** The only gain is graph size (886 → 321 MB).
  `reduce_range` is chosen from the CPU flags; on a non-VNNI CPU, turning it off saturates the
  accumulator and hurts quality.
- **Warm-up cost is `instances × buckets × count`.** Six instances can take minutes to warm up;
  lower `WARMUP_COUNT` for CPU deployments.

---

## 6. RNNT

```bash
./build.sh --variant v3_e2e_rnnt --runtime trt --precision fp16
docker run -d --gpus all ... -e VARIANT=v3_e2e_rnnt -e RNNT_PROVIDER=cpu ...
```

The transducer loop is the known bottleneck; here it accounts for 64–80 % of request time. The
implementation keeps all buffers preallocated, and LSTM state never goes through the host. The
joint input is taken by pointer offset, with no per-frame copies.

`RNNT_PROVIDER=cpu` is the default because it measured faster: the loop runs 1.5× faster on
CPU than on CUDA. At these matrix sizes, kernel launch and synchronization cost more than the
arithmetic itself.

On a GPU, the CTC variant is faster at every point we measured. Use RNNT only if its accuracy
matters more than the latency.

---

## 7. Evaluation

This is the only part that needs Python packages on the host:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install numpy scipy soundfile 'tritonclient[grpc]' num2words
pip install git+https://github.com/voicekit-team/T-one.git   # only to compare against T-one
```

Run:

```bash
docker cp gigaam-triton:/models/manifest.json /tmp/manifest.json
python eval/run_eval.py --audio-dir eval/audio \
  --gigaam-manifest /tmp/manifest.json --skip-tone
```

Put audio into `eval/audio/` as pairs: `name.wav` plus a reference transcript `name.txt`. Audio
files are not part of the repository.

When comparing against another ASR system:

- **Run the competitor through its own official client**, not your re-implementation of its
  decoder. Otherwise you measure your reconstruction rather than the system.
- **References must not be produced by any of the systems being compared.** A reference copied
  from one system's output gives that system zero error on it by construction.

`eval/normalize.py` brings both sides to the same spoken form: lowercase, `ё` → `е`, no
punctuation, numbers spelled out as words. It has built-in self-tests: `python eval/normalize.py`.

---

## 8. Repository layout

```
build.sh                  single entry point
docker/
  Dockerfile.builder      torch + gigaam + TensorRT; heavy, never shipped
  Dockerfile.runtime      slim: no torch, no gigaam
  entrypoint.sh           engine check -> model repository generation -> tritonserver
scripts/
  detect_env.py           GPU probe, base image selection -> cache/env.json
  export_onnx.py          PyTorch -> ONNX, meta.json, filterbank, golden check
  check_logmel.py         gate: numpy log-mel must match torchaudio
  quantize_onnx.py        dynamic int8 + quality before/after
  buckets.py              shared bucket table, so shapes never drift apart
  build_trt.py            ONNX -> TensorRT, one profile per bucket, cache key
  gen_model_repo.py       model repository generation
  smoke_test.py           end-to-end test + negative checks
  perf_profile.py         latency matrix
  perf_summary.py         summary table from reports
  test_backends.py        backend unit checks
backends/                 Triton Python backends (no torch)
templates/config/         config.pbtxt templates
eval/                     normalization, engine wrappers, comparison runner
testdata/golden/          reference transcripts for the smoke test
artifacts/<precision>/    export output (git-ignored)
cache/                    env.json, engines, timing cache, perf reports (engines are git-ignored)
```

---

## 9. Things to know up front

**Don't start from upstream's `triton_scripts/`.** Its converter replaces the CTC log-probs with
`argmax`, which throws away exactly the output a client needs for its own decoding. This project
returns both: log-probs for the client, and an argmax computed on the GPU so the Python backend
stays cheap.

**v3 preprocessing is not on the library defaults.** v3 uses `n_fft=320` with `center=False`;
v2 uses 400 with centering. All frame counts are derived from `meta.json`, never assumed. A
check forbids mixing v2 and v3 in one model repository.

**The CTC head is one class wider than the tokenizer:** 256 tokens, 257 classes, with blank at
index 256. `vocab_size` and `num_classes` are separate fields for a reason.

**`ffmpeg` is an undeclared dependency of `gigaam`.** The Triton image doesn't include it, so both
Dockerfiles install it.

**Engines are tied to the machine.** The GPU, driver, TensorRT version and build flags are all
part of the cache key, so a driver update invalidates the `.plan`. This is detected
automatically, but deployments need to budget time for the rebuild.

**Defaults set by measurement** (change them only with a new measurement):

- numpy thread pools are pinned to one thread. On small arrays the pool costs more than the
  work, and pinning made preprocessing 3× faster.
- CUDA graphs are off. They gave no measurable gain, and capture fails on some instances.
- The RNNT loop runs on CPU (see [§6](#6-rnnt)).
