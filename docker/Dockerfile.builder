# Conversion image: PyTorch -> ONNX -> TensorRT. Heavy, never shipped anywhere.
#
# BASE_IMAGE is passed in by build.sh from cache/env.json. It is deliberately not
# defaulted to a literal here: the whole point of scripts/detect_env.py is that the
# image is chosen by probing the GPU, and a default in a Dockerfile is exactly the
# kind of stale pin that breaks on Blackwell (spec §4).
#
# This must be the SAME base as Dockerfile.runtime. A .plan built against one TRT
# version will not load in a server running another.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# gigaam.preprocess.load_audio shells out to ffmpeg -- it is an undeclared
# dependency of the package, and the Triton images do not ship it. Without this
# every wav read dies at the first subprocess call.
# libsndfile1 backs soundfile, which gigaam.utils uses to read durations.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# CUDA build of torch: the fp16 export path traces a real forward pass, and
# several ops (LSTM in the RNNT decoder above all) have no fp16 CPU kernel, so a
# CPU-only wheel cannot export the RNNT head at all. cu128 carries sm_120 kernels.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} torch torchaudio

RUN pip install --no-cache-dir \
    "numpy>=2,<3" \
    "onnx==1.19.*" \
    "onnxruntime-gpu==1.23.*" \
    "hydra-core==1.3.*" \
    "omegaconf==2.3.*" \
    polygraphy \
    soundfile \
    sentencepiece \
    tqdm \
    jinja2 \
    scipy

# Pinned commit, not a floating branch: the export contract (tensor names, output
# layout) is read straight out of this source, so an upstream change must be an
# explicit bump here rather than a surprise on the next rebuild.
ARG GIGAAM_REF=559d88d6b72541412743929f633a6ae7c9950b85
RUN pip install --no-cache-dir --no-deps \
    "git+https://github.com/salute-developers/GigaAM@${GIGAAM_REF}"

WORKDIR /workspace
COPY scripts/ /workspace/scripts/
COPY templates/ /workspace/templates/
COPY backends/ /workspace/backends/

ENV PYTHONUNBUFFERED=1
