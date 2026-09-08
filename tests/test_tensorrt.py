from pathlib import Path

import pytest

from surgphase.tensorrt_tools import (
    TensorRTBenchmark,
    benchmark_engine_command,
    build_engine_command,
    format_input_shape,
    parse_trtexec_benchmark,
)


def test_format_input_shape() -> None:
    assert (
        format_input_shape(8)
        == "8x3x224x224"
    )


def test_build_engine_command() -> None:
    command = build_engine_command(
        onnx_path=Path(
            "models/model.onnx"
        ),
        engine_path=Path(
            "models/model.plan"
        ),
        min_batch=1,
        opt_batch=8,
        max_batch=32,
    )

    assert (
        "--onnx=models/model.onnx"
        in command
    )

    assert (
        "--saveEngine=models/model.plan"
        in command
    )

    assert (
        "--minShapes=input:1x3x224x224"
        in command
    )

    assert (
        "--optShapes=input:8x3x224x224"
        in command
    )

    assert (
        "--maxShapes=input:32x3x224x224"
        in command
    )

    assert "--skipInference" in command

    assert "--fp16" not in command


def test_build_engine_rejects_invalid_profile() -> None:
    with pytest.raises(
        ValueError,
        match="min_batch",
    ):
        build_engine_command(
            onnx_path=Path(
                "model.onnx"
            ),
            engine_path=Path(
                "model.plan"
            ),
            min_batch=8,
            opt_batch=4,
            max_batch=32,
        )


def test_benchmark_engine_command() -> None:
    command = benchmark_engine_command(
        engine_path=Path(
            "models/model.plan"
        ),
        batch_size=4,
        warmup_ms=500,
        duration_seconds=10,
        iterations=100,
    )

    assert (
        "--loadEngine=models/model.plan"
        in command
    )

    assert (
        "--shapes=input:4x3x224x224"
        in command
    )

    assert "--warmUp=500" in command
    assert "--duration=10" in command
    assert "--iterations=100" in command


def test_parse_trtexec_benchmark() -> None:
    output = """
[I] === Performance summary ===
[I] Throughput: 812.45 qps
[I] Latency: min = 1.102 ms, max = 2.104 ms, mean = 1.231 ms, median = 1.205 ms, percentile(90%) = 1.320 ms, percentile(95%) = 1.410 ms, percentile(99%) = 1.650 ms
"""

    result = parse_trtexec_benchmark(
        output=output,
        batch_size=1,
    )

    assert isinstance(
        result,
        TensorRTBenchmark,
    )

    assert result.batch_size == 1

    assert (
        result.throughput_qps
        == pytest.approx(812.45)
    )

    assert (
        result.latency_mean_ms
        == pytest.approx(1.231)
    )

    assert (
        result.latency_median_ms
        == pytest.approx(1.205)
    )

    assert (
        result.latency_p95_ms
        == pytest.approx(1.410)
    )

    assert (
        result.latency_p99_ms
        == pytest.approx(1.650)
    )


def test_parse_trtexec_rejects_missing_metrics() -> None:
    with pytest.raises(
        ValueError,
        match="throughput",
    ):
        parse_trtexec_benchmark(
            output="TensorRT output without metrics",
            batch_size=1,
        )