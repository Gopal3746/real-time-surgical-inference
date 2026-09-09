#!/usr/bin/env bash

set -euo pipefail

IMAGE="${TRITON_SDK_IMAGE:-nvcr.io/nvidia/tritonserver:26.08-py3-sdk}"
TRITON_URL="${TRITON_URL:-localhost:8000}"
MODEL_NAME="${TRITON_MODEL_NAME:-surgical_phase}"
CONCURRENCY_RANGE="${CONCURRENCY_RANGE:-1:32:1}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Triton Performance Analyzer container requires Linux."
    echo "Current platform: $(uname -s)"
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed or unavailable on PATH."
    exit 1
fi

echo "Triton SDK image: ${IMAGE}"
echo "Server: ${TRITON_URL}"
echo "Model: ${MODEL_NAME}"
echo "Concurrency: ${CONCURRENCY_RANGE}"

docker run \
    --rm \
    --network host \
    "${IMAGE}" \
    perf_analyzer \
    -m "${MODEL_NAME}" \
    -u "${TRITON_URL}" \
    -i http \
    -b 1 \
    --shape input:3,224,224 \
    --percentile=99 \
    --concurrency-range "${CONCURRENCY_RANGE}"
