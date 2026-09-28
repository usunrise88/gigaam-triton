#!/usr/bin/env bash
# Single entry point (spec §8).
#
#   ./build.sh --variant v3_e2e_ctc --runtime trt --precision fp16 \
#              --buckets 2,5,10,20 --verify
#
# Idempotent: rerunning with the same arguments rebuilds nothing that is already
# in the cache with a valid key. The base image is never hardcoded -- it comes
# from cache/env.json, which scripts/detect_env.py fills in by probing the GPU.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

VARIANTS="v3_e2e_ctc"
RUNTIME="trt"
PRECISION=""
BUCKETS="2,5,10,20"
MAX_BATCH_SIZE=8
INSTANCES=2
# CPU profile knobs (spec §7.3): N x intra_op should land near the physical core
# count, and both extremes are worth measuring -- many single-threaded instances
# against one wide one.
CPU_INSTANCES=6
INTRA_OP_THREADS=4
IMAGE="gigaam-triton:0.1"
BUILDER_IMAGE="gigaam-triton-builder:0.1"
MODEL_REPO="${HERE}/model_repo"
CACHE="${HERE}/cache"
ARTIFACTS=""   # set once EXPORT_DTYPE is known
SKIP_TRT=0
VERIFY=0
FORCE=0
PULL=0

usage() { sed -n '2,12p' "$0"; exit "${1:-0}"; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --variant)         VARIANTS="$2"; shift 2 ;;
        --runtime)         RUNTIME="$2"; shift 2 ;;
        --precision)       PRECISION="$2"; shift 2 ;;
        --buckets)         BUCKETS="$2"; shift 2 ;;
        --max-batch-size)  MAX_BATCH_SIZE="$2"; shift 2 ;;
        --instances)       INSTANCES="$2"; shift 2 ;;
        --cpu-instances)   CPU_INSTANCES="$2"; shift 2 ;;
        --intra-op-threads) INTRA_OP_THREADS="$2"; shift 2 ;;
        --image)           IMAGE="$2"; shift 2 ;;
        --model-repo)      MODEL_REPO="$2"; shift 2 ;;
        --cache)           CACHE="$2"; shift 2 ;;
        --skip-trt)        SKIP_TRT=1; shift ;;
        --verify)          VERIFY=1; shift ;;
        --force)           FORCE=1; shift ;;
        --pull)            PULL=1; shift ;;
        -h|--help)         usage 0 ;;
        *) echo "unknown flag: $1" >&2; usage 1 ;;
    esac
done

log() { printf '\n=== %s\n' "$*"; }

# fp16 on GPU, int8 on CPU: the CPU default is dynamic ONNX Runtime quantisation,
# which public numbers say is nearly free for v3 (unlike v2, where it cost 3x).
# Verify on this hardware and always measure WER either side of it -- speed
# without WER means nothing here (spec §8).
if [ -z "${PRECISION}" ]; then
    case "${RUNTIME}" in
        onnx-cpu) PRECISION="int8" ;;
        *)        PRECISION="fp16" ;;
    esac
fi

EXPORT_DTYPE="fp16"
[ "${RUNTIME}" = "onnx-cpu" ] && EXPORT_DTYPE="fp32"

# One artefact root per precision. fp16 and fp32 exports of the same variant want
# the same directory name, and overwriting silently leaves a repository whose
# meta.json and model.onnx disagree about dtype.
ARTIFACTS_REL="artifacts/${EXPORT_DTYPE}"
ARTIFACTS="${HERE}/${ARTIFACTS_REL}"

if [ "${PRECISION}" = "int8" ] && [ "${RUNTIME}" != "onnx-cpu" ]; then
    echo "--precision int8 only applies to --runtime onnx-cpu." >&2
    exit 2
fi

mkdir -p "${CACHE}/engines" "${ARTIFACTS}" "${MODEL_REPO}"

# ---------------------------------------------------------------- 1. environment
log "1/6  environment"
if [ ! -f "${CACHE}/env.json" ] || [ "${FORCE}" = 1 ]; then
    PULL_ARG=()
    [ "${PULL}" = 1 ] && PULL_ARG=(--pull)
    python3 scripts/detect_env.py --out "${CACHE}/env.json" "${PULL_ARG[@]}"
else
    echo "using existing ${CACHE}/env.json (--force to re-probe)"
fi

BASE_IMAGE="$(python3 -c "import json,sys;print(json.load(open('${CACHE}/env.json'))['selected_image']['image'])")"
TRT_VERSION="$(python3 -c "import json,sys;print(json.load(open('${CACHE}/env.json'))['selected_image']['tensorrt_version'])")"
echo "base image: ${BASE_IMAGE} (TensorRT ${TRT_VERSION})"

# ---------------------------------------------------------------- 2. builder
log "2/6  builder image"
docker build \
    -f docker/Dockerfile.builder \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${BUILDER_IMAGE}" .

# Model weights are cached on the host so the container never re-downloads 1.7 GB.
GIGAAM_CACHE="${HOME}/.cache/gigaam"
mkdir -p "${GIGAAM_CACHE}"

builder_run() {
    docker run --rm --gpus all \
        -v "${ARTIFACTS}:/work/onnx" \
        -v "${CACHE}:/work/cache" \
        -v "${HERE}/testdata:/work/testdata" \
        -v "${GIGAAM_CACHE}:/root/.cache/gigaam" \
        "${BUILDER_IMAGE}" "$@"
}

# ---------------------------------------------------------------- 3. log-mel gate
# Runs before anything is produced. The runtime recomputes log-mel with numpy
# because it has no torch; if that replay drifts from torchaudio nothing crashes,
# the model just gets slightly wrong features and the regression gets blamed on
# the model months later.
log "3/6  torch-free log-mel gate"
FIRST_VARIANT="${VARIANTS%%,*}"
builder_run python3 scripts/check_logmel.py \
    --variant "${FIRST_VARIANT}" \
    --golden-dir /work/testdata/golden \
    --filterbank-out /work/cache/filterbank_check.npz

# ---------------------------------------------------------------- 4. export
log "4/6  ONNX export (${EXPORT_DTYPE})"
IFS=',' read -ra VARIANT_LIST <<< "${VARIANTS}"
for variant in "${VARIANT_LIST[@]}"; do
    meta="${ARTIFACTS}/${variant}/meta.json"
    if [ -f "${meta}" ] && [ "${FORCE}" = 0 ] \
       && [ "$(python3 -c "import json;print(json.load(open('${meta}'))['export_dtype'])")" = "${EXPORT_DTYPE}" ]; then
        echo "  ${variant}: up to date, skipping"
        continue
    fi
    builder_run python3 scripts/export_onnx.py \
        --variant "${variant}" \
        --dtype "${EXPORT_DTYPE}" \
        --out /work/onnx \
        --golden-dir /work/testdata/golden
done

# ---------------------------------------------------------------- 4b. quantise
# The int8 graph is portable, unlike a TensorRT plan, so it is produced once here
# and baked into the image rather than rebuilt on every target.
WEIGHTS_ARG=()
if [ "${RUNTIME}" = "onnx-cpu" ] && [ "${PRECISION}" = "int8" ]; then
    log "4b/6 int8 dynamic quantisation"
    for variant in "${VARIANT_LIST[@]}"; do
        if [ -f "${ARTIFACTS}/${variant}/model.int8.onnx" ] && [ "${FORCE}" = 0 ]; then
            echo "  ${variant}: already quantised, skipping"
            continue
        fi
        builder_run python3 scripts/quantize_onnx.py \
            --onnx "/work/onnx/${variant}/model.onnx" \
            --golden-dir /work/testdata/golden \
            --report "/work/cache/perf/quantize_${variant}.json"
    done
    WEIGHTS_ARG=(--weights-name model.int8.onnx)
fi

# ---------------------------------------------------------------- 5. engines
if [ "${RUNTIME}" = "trt" ] && [ "${SKIP_TRT}" = 0 ]; then
    log "5/6  TensorRT engines (${PRECISION}, buckets ${BUCKETS})"
    for variant in "${VARIANT_LIST[@]}"; do
        builder_run python3 scripts/build_trt.py \
            --onnx "/work/onnx/${variant}/model.onnx" \
            --out "/work/cache/engines/${variant}_${PRECISION}.plan" \
            --precision "${PRECISION}" \
            --buckets "${BUCKETS}" \
            --max-batch-size "${MAX_BATCH_SIZE}" \
            --timing-cache /work/cache/trt_timing.cache
    done
else
    log "5/6  TensorRT engines skipped"
    echo "The runtime entrypoint will build them on first start (spec §6.3)."
fi

# ---------------------------------------------------------------- 6. runtime
log "6/6  runtime image"
docker build \
    -f docker/Dockerfile.runtime \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg ARTIFACTS="${ARTIFACTS_REL}" \
    -t "${IMAGE}" .

echo
echo "built ${IMAGE}"
echo "run:  docker run --rm --gpus all --name gigaam-triton \\"
echo "        -p 18000:18000 -p 18001:18001 -p 18002:18002 \\"
echo "        -v ${CACHE}:/cache \\"
echo "        -e VARIANT=${FIRST_VARIANT} -e RUNTIME=${RUNTIME} \\"
echo "        -e PRECISION=${PRECISION} -e BUCKETS=${BUCKETS} \\"
echo "        ${IMAGE}"

# ---------------------------------------------------------------- verify
if [ "${VERIFY}" = 1 ]; then
    log "verify: starting server and running smoke test"
    NAME="gigaam-triton-verify-$$"
    # 17000-17002 belong to the production T-one deployment on this host.
    docker run -d --rm --gpus all --name "${NAME}" \
        -p 18000:18000 -p 18001:18001 -p 18002:18002 \
        -v "${CACHE}:/cache" \
        -v "${HERE}/testdata:/testdata" \
        -e VARIANT="${FIRST_VARIANT}" \
        -e RUNTIME="${RUNTIME}" \
        -e PRECISION="${PRECISION}" \
        -e BUCKETS="${BUCKETS}" \
        -e CPU_INSTANCES="${CPU_INSTANCES}" \
        -e INTRA_OP_THREADS="${INTRA_OP_THREADS}" \
        -e MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
        -e INSTANCES="${INSTANCES}" \
        "${IMAGE}" >/dev/null

    cleanup() { docker rm -f "${NAME}" >/dev/null 2>&1 || true; }
    trap cleanup EXIT

    echo -n "waiting for readiness (engine build can take minutes) "
    ready=0
    for _ in $(seq 1 180); do
        if curl -sf -o /dev/null http://localhost:18000/v2/health/ready; then ready=1; break; fi
        if ! docker ps -q --filter "name=${NAME}" | grep -q .; then
            echo " container exited"; docker logs "${NAME}" 2>&1 | tail -40; exit 1
        fi
        echo -n "."; sleep 5
    done
    echo
    if [ "${ready}" = 0 ]; then
        echo "server never became ready" >&2
        docker logs "${NAME}" 2>&1 | tail -60
        exit 1
    fi

    docker exec "${NAME}" python3 /opt/gigaam/scripts/smoke_test.py \
        --url localhost:18001 --repo /models --golden-dir /testdata/golden
fi
