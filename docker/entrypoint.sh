#!/usr/bin/env bash
# Serving entrypoint: make sure the engine matches this machine, render the model
# repository, then hand over to tritonserver.
#
# The engine is built here rather than baked into the image because a TensorRT
# plan is tied to the GPU, the driver and the TRT version (spec §6.3). Baking one
# in makes the image non-reproducible and silently broken after a driver update.
# Building four buckets takes minutes, which is why it happens at deploy time and
# not on the first call.
set -euo pipefail

ROOT="${GIGAAM_ROOT:-/opt/gigaam}"
CACHE="${CACHE_DIR:-/cache}"
REPO="${MODEL_REPO:-/models}"
ONNX_ROOT="${ROOT}/onnx"

VARIANT="${VARIANT:-v3_e2e_ctc}"
RUNTIME="${RUNTIME:-trt}"
PRECISION="${PRECISION:-fp16}"
BUCKETS="${BUCKETS:-2,5,10,20}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
INSTANCES="${INSTANCES:-2}"
CPU_INSTANCES="${CPU_INSTANCES:-6}"
# RNNT decode loop knobs (spec §7.4). Which provider wins is a measurement, not
# an assumption -- the decoder is a 320-wide LSTM and the joint two small
# matmuls, so launch overhead can outweigh the arithmetic.
RNNT_PROVIDER="${RNNT_PROVIDER:-cpu}"
RNNT_INSTANCES="${RNNT_INSTANCES:-8}"
MAX_SYMBOLS_PER_FRAME="${MAX_SYMBOLS_PER_FRAME:-3}"
INTRA_OP_THREADS="${INTRA_OP_THREADS:-4}"
# Warmup cost is instances x buckets x count, and on CPU one 20 s-bucket
# inference is seconds rather than milliseconds -- six instances saturate all
# cores for minutes before the server reports ready. Tunable for that reason.
WARMUP_COUNT="${WARMUP_COUNT:-3}"
# Dynamic batching waits this long for a batch to form. At concurrency 1 the
# batch never grows, so the wait is pure latency -- and it is paid once per
# ensemble step. Worth setting to 0 on a latency-bound single-stream deployment.
MAX_QUEUE_DELAY_US="${MAX_QUEUE_DELAY_US:-1000}"

log() { printf '[entrypoint] %s\n' "$*"; }

# A bare `tritonserver ...` or any other command runs as given -- keeps the image
# usable for one-off debugging without fighting the entrypoint.
if [ "$#" -gt 0 ] && [ "${1}" != "serve" ]; then
    exec "$@"
fi

if [ ! -d "${ONNX_ROOT}/${VARIANT}" ]; then
    log "FATAL: no exported artefacts for variant '${VARIANT}' in ${ONNX_ROOT}"
    log "available: $(ls -1 "${ONNX_ROOT}" 2>/dev/null | tr '\n' ' ')"
    exit 1
fi

mkdir -p "${CACHE}/engines" "${REPO}"

# The quantised graph ships in the image (it is portable, unlike a TRT plan).
WEIGHTS_ARG=()
if [ "${RUNTIME}" = "onnx-cpu" ] && [ "${PRECISION}" = "int8" ]; then
    if [ ! -f "${ONNX_ROOT}/${VARIANT}/model.int8.onnx" ]; then
        log "FATAL: PRECISION=int8 but ${VARIANT}/model.int8.onnx is not in the image."
        log "Rebuild with: ./build.sh --runtime onnx-cpu --precision int8"
        exit 1
    fi
    WEIGHTS_ARG=(--weights-name model.int8.onnx)

    # int8 weights are shaped for the CPU they were quantised on: reduce_range
    # bakes 7-bit weights in, which is correct without VNNI and needlessly lossy
    # with it. The graph loads either way, so say something rather than silently
    # serving a mismatch.
    python3 - "${ONNX_ROOT}/${VARIANT}/model.int8.json" <<'PY' || true
import json, sys
try:
    built = json.load(open(sys.argv[1]))
except OSError:
    sys.exit(0)
flags = set()
try:
    for line in open("/proc/cpuinfo"):
        if line.startswith("flags"):
            flags = set(line.split(":", 1)[1].split()); break
except OSError:
    pass
here = "avx512_vnni" in flags or "avx_vnni" in flags
there = bool(built.get("built_for_cpu", {}).get("vnni"))
if here != there:
    print(f"[entrypoint] WARNING: int8 weights were quantised on a CPU with "
          f"vnni={there}, this host has vnni={here}.")
    print("[entrypoint] They will run, but re-quantise on the target for the "
          "accuracy and speed that CPU can actually give.")
PY
fi

ENGINE_ARG=()
if [ "${RUNTIME}" = "trt" ]; then
    ENGINE="${CACHE}/engines/${VARIANT}_${PRECISION}.plan"
    # The CTC graph is one file called model.onnx; RNNT splits into
    # encoder/decoder/joint and only the encoder goes to TensorRT. meta.json says
    # which is which, so the name is read rather than assumed.
    ENCODER_ONNX="$(python3 -c "import json,sys;print(json.load(open('${ONNX_ROOT}/${VARIANT}/meta.json'))['files']['encoder'])")"
    log "checking TensorRT engine for ${VARIANT} (${PRECISION}, buckets ${BUCKETS}, ${ENCODER_ONNX})"
    # build_trt.py is a no-op on a cache hit and prints which key component moved
    # on a miss -- a driver bump and a flag change need different responses.
    python3 "${ROOT}/scripts/build_trt.py" \
        --onnx "${ONNX_ROOT}/${VARIANT}/${ENCODER_ONNX}" \
        --out "${ENGINE}" \
        --precision "${PRECISION}" \
        --buckets "${BUCKETS}" \
        --max-batch-size "${MAX_BATCH_SIZE}" \
        --timing-cache "${CACHE}/trt_timing.cache"
    ENGINE_ARG=(--engine "${ENGINE}")
fi

log "rendering model repository -> ${REPO}"
rm -rf "${REPO:?}"/*
python3 "${ROOT}/scripts/gen_model_repo.py" \
    --onnx-root "${ONNX_ROOT}" \
    --variant "${VARIANT}" \
    --runtime "${RUNTIME}" \
    --buckets "${BUCKETS}" \
    --max-batch-size "${MAX_BATCH_SIZE}" \
    --instances "${INSTANCES}" \
    --cpu-instances "${CPU_INSTANCES}" \
    --intra-op-threads "${INTRA_OP_THREADS}" \
    --warmup-count "${WARMUP_COUNT}" \
    --max-queue-delay-us "${MAX_QUEUE_DELAY_US}" \
    --templates "${ROOT}/templates/config" \
    --backends "${ROOT}/backends" \
    --cuda-graphs "${CUDA_GRAPHS:-auto}" \
    --rnnt-provider "${RNNT_PROVIDER}" \
    --rnnt-instances "${RNNT_INSTANCES}" \
    --max-symbols-per-frame "${MAX_SYMBOLS_PER_FRAME}" \
    --out "${REPO}" \
    "${ENGINE_ARG[@]}" "${WEIGHTS_ARG[@]}"

log "starting tritonserver"
# --exit-on-error: a model that fails to load must stop the container rather than
# leave a server up that answers some buckets and 404s the rest.
exec tritonserver \
    --model-repository="${REPO}" \
    --http-port=18000 \
    --grpc-port=18001 \
    --metrics-port=18002 \
    --exit-on-error=true \
    --log-verbose="${TRITON_LOG_VERBOSE:-0}"
