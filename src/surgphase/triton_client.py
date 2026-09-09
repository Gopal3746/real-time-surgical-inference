from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch

from surgphase.model import NUM_PHASES
from surgphase.triton_repository import (
    DEFAULT_MODEL_NAME,
    INPUT_NAME,
    OUTPUT_NAME,
)

DEFAULT_TRITON_URL = "localhost:8000"


@dataclass(frozen=True)
class TritonInferenceResult:
    logits: torch.Tensor
    latency_ms: float


class TritonPhaseClient:
    def __init__(
        self,
        url: str = DEFAULT_TRITON_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        model_version: str = "1",
        *,
        http_module: Any | None = None,
        client: Any | None = None,
    ) -> None:
        if not url:
            raise ValueError(
                "Triton URL cannot be empty"
            )

        if not model_name:
            raise ValueError(
                "Model name cannot be empty"
            )

        if http_module is None:
            import tritonclient.http as httpclient

            http_module = httpclient

        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self._http = http_module

        self._client = (
            client
            if client is not None
            else self._http.InferenceServerClient(
                url=url,
            )
        )

    def check_ready(self) -> None:
        if not self._client.is_server_live():
            raise RuntimeError(
                "Triton server is not live"
            )

        if not self._client.is_server_ready():
            raise RuntimeError(
                "Triton server is not ready"
            )

        if not self._client.is_model_ready(
            self.model_name,
            self.model_version,
        ):
            raise RuntimeError(
                f"Triton model is not ready: "
                f"{self.model_name}"
            )

    def infer(
        self,
        inputs: torch.Tensor,
    ) -> TritonInferenceResult:
        self._validate_inputs(inputs)

        input_array = (
            inputs
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )

        infer_input = self._http.InferInput(
            INPUT_NAME,
            list(input_array.shape),
            "FP32",
        )

        infer_input.set_data_from_numpy(
            input_array,
            binary_data=True,
        )

        requested_output = (
            self._http.InferRequestedOutput(
                OUTPUT_NAME,
                binary_data=True,
            )
        )

        start = perf_counter()

        result = self._client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[
                infer_input,
            ],
            outputs=[
                requested_output,
            ],
        )

        latency_ms = (
            perf_counter() - start
        ) * 1000.0

        output_array = result.as_numpy(
            OUTPUT_NAME
        )

        if output_array is None:
            raise RuntimeError(
                f"Triton response did not contain "
                f"{OUTPUT_NAME}"
            )

        output_array = np.asarray(
            output_array,
            dtype=np.float32,
        )

        expected_shape = (
            inputs.shape[0],
            NUM_PHASES,
        )

        if output_array.shape != expected_shape:
            raise ValueError(
                "Unexpected Triton output shape: "
                f"expected {expected_shape}, "
                f"got {output_array.shape}"
            )

        return TritonInferenceResult(
            logits=torch.from_numpy(
                output_array.copy()
            ),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _validate_inputs(
        inputs: torch.Tensor,
    ) -> None:
        if not isinstance(
            inputs,
            torch.Tensor,
        ):
            raise TypeError(
                "Inputs must be a PyTorch tensor"
            )

        if inputs.ndim != 4:
            raise ValueError(
                "Inputs must have shape "
                "[batch, 3, 224, 224]"
            )

        if tuple(inputs.shape[1:]) != (
            3,
            224,
            224,
        ):
            raise ValueError(
                "Expected input shape "
                "[batch, 3, 224, 224], "
                f"got {tuple(inputs.shape)}"
            )

        if inputs.shape[0] <= 0:
            raise ValueError(
                "Batch cannot be empty"
            )