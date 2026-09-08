#!/usr/bin/env bash

set -euo pipefail

IMAGE="${TENSORRT_IMAGE:-nvcr.io/nvidia/tensorrt:26.08-py3}"

PROJECT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

echo "TensorRT image: ${IMAGE}"
echo "Project: ${PROJECT_DIR}"

docker run \
    --gpus all \
    --rm \
    -it \
    -v "${PROJECT_DIR}:/workspace/project" \
    -w /workspace/project \
    "${IMAGE}" \
    bash