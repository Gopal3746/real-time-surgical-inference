#!/usr/bin/env bash

set -euo pipefail

IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:26.08-py3}"

PROJECT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

MODEL_REPOSITORY="${TRITON_MODEL_REPOSITORY:-${PROJECT_DIR}/deployment/model_repository}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Triton GPU serving requires a Linux host with an NVIDIA GPU."
    echo "Current platform: $(uname -s)"
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed or is not available on PATH."
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No NVIDIA GPU runtime detected."
    exit 1
fi

if [[ ! -d "${MODEL_REPOSITORY}" ]]; then
    echo "Triton model repository does not exist:"
    echo "${MODEL_REPOSITORY}"
    exit 1
fi

echo "Triton image: ${IMAGE}"
echo "Model repository: ${MODEL_REPOSITORY}"

docker run \
    --gpus all \
    --rm \
    -it \
    --shm-size=1g \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v "${MODEL_REPOSITORY}:/models:ro" \
    "${IMAGE}" \
    tritonserver \
    --model-repository=/models