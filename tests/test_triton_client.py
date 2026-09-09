import numpy as np
import pytest
import torch

from surgphase.triton_client import (
    TritonPhaseClient,
)


class FakeInferInput:
    def __init__(
        self,
        name: str,
        shape: list[int],
        datatype: str,
    ) -> None:
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.data: np.ndarray | None = None

    def set_data_from_numpy(
        self,
        data: np.ndarray,
        binary_data: bool = True,
    ) -> None:
        self.data = data


class FakeRequestedOutput:
    def __init__(
        self,
        name: str,
        binary_data: bool = True,
    ) -> None:
        self.name = name


class FakeHTTP:
    InferInput = FakeInferInput
    InferRequestedOutput = FakeRequestedOutput


class FakeResult:
    def __init__(
        self,
        logits: np.ndarray,
    ) -> None:
        self.logits = logits

    def as_numpy(
        self,
        name: str,
    ) -> np.ndarray:
        return self.logits


class FakeClient:
    def __init__(self) -> None:
        self.last_inputs = None

    def is_server_live(self) -> bool:
        return True

    def is_server_ready(self) -> bool:
        return True

    def is_model_ready(
        self,
        model_name: str,
        model_version: str,
    ) -> bool:
        return True

    def infer(
        self,
        **kwargs,
    ) -> FakeResult:
        self.last_inputs = kwargs["inputs"]

        batch_size = (
            kwargs["inputs"][0]
            .data
            .shape[0]
        )

        logits = np.zeros(
            (
                batch_size,
                7,
            ),
            dtype=np.float32,
        )

        logits[:, 2] = 10.0

        return FakeResult(
            logits
        )


def create_client() -> TritonPhaseClient:
    return TritonPhaseClient(
        http_module=FakeHTTP,
        client=FakeClient(),
    )


def test_triton_client_inference() -> None:
    client = create_client()

    inputs = torch.randn(
        2,
        3,
        224,
        224,
    )

    result = client.infer(
        inputs
    )

    assert result.logits.shape == (
        2,
        7,
    )

    assert (
        result.logits.argmax(
            dim=1
        ).tolist()
        == [2, 2]
    )

    assert result.latency_ms >= 0.0


def test_triton_client_health_check() -> None:
    client = create_client()

    client.check_ready()


def test_triton_client_rejects_bad_shape() -> None:
    client = create_client()

    with pytest.raises(
        ValueError,
        match="224",
    ):
        client.infer(
            torch.randn(
                1,
                3,
                128,
                128,
            )
        )


def test_triton_client_rejects_empty_url() -> None:
    with pytest.raises(
        ValueError,
        match="URL",
    ):
        TritonPhaseClient(
            url="",
            http_module=FakeHTTP,
            client=FakeClient(),
        )