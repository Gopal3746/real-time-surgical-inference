import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

INPUT_NAME = "input"

DEFAULT_INPUT_SHAPE = (
    3,
    224,
    224,
)

DEFAULT_MIN_BATCH_SIZE = 1
DEFAULT_OPT_BATCH_SIZE = 8
DEFAULT_MAX_BATCH_SIZE = 32

DEFAULT_WORKSPACE_MIB = 4096


@dataclass(frozen=True)
class TensorRTBenchmark:
    batch_size: int
    throughput_qps: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


def format_input_shape(
    batch_size: int,
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
) -> str:
    if batch_size <= 0:
        raise ValueError(
            "Batch size must be positive"
        )

    if any(
        dimension <= 0
        for dimension in input_shape
    ):
        raise ValueError(
            f"Invalid input shape: {input_shape}"
        )

    dimensions = (
        batch_size,
        *input_shape,
    )

    return "x".join(
        str(dimension)
        for dimension in dimensions
    )


def validate_batch_profile(
    min_batch: int,
    opt_batch: int,
    max_batch: int,
) -> None:
    if min_batch <= 0:
        raise ValueError(
            "Minimum batch size must be positive"
        )

    if not (
        min_batch
        <= opt_batch
        <= max_batch
    ):
        raise ValueError(
            "Batch profile must satisfy "
            "min_batch <= opt_batch <= max_batch"
        )


def build_engine_command(
    onnx_path: Path,
    engine_path: Path,
    *,
    min_batch: int = DEFAULT_MIN_BATCH_SIZE,
    opt_batch: int = DEFAULT_OPT_BATCH_SIZE,
    max_batch: int = DEFAULT_MAX_BATCH_SIZE,
    workspace_mib: int = DEFAULT_WORKSPACE_MIB,
    executable: str = "trtexec",
) -> list[str]:
    validate_batch_profile(
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
    )

    if workspace_mib <= 0:
        raise ValueError(
            "Workspace size must be positive"
        )

    min_shape = format_input_shape(
        min_batch
    )

    opt_shape = format_input_shape(
        opt_batch
    )

    max_shape = format_input_shape(
        max_batch
    )

    return [
        executable,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        (
            f"--minShapes="
            f"{INPUT_NAME}:{min_shape}"
        ),
        (
            f"--optShapes="
            f"{INPUT_NAME}:{opt_shape}"
        ),
        (
            f"--maxShapes="
            f"{INPUT_NAME}:{max_shape}"
        ),
        (
            f"--memPoolSize="
            f"workspace:{workspace_mib}"
        ),
        "--skipInference",
    ]


def benchmark_engine_command(
    engine_path: Path,
    *,
    batch_size: int = 1,
    warmup_ms: int = 500,
    duration_seconds: int = 10,
    iterations: int = 100,
    executable: str = "trtexec",
) -> list[str]:
    if warmup_ms < 0:
        raise ValueError(
            "Warm-up duration cannot be negative"
        )

    if duration_seconds < 0:
        raise ValueError(
            "Benchmark duration cannot be negative"
        )

    if iterations <= 0:
        raise ValueError(
            "Iterations must be positive"
        )

    shape = format_input_shape(
        batch_size
    )

    return [
        executable,
        f"--loadEngine={engine_path}",
        f"--shapes={INPUT_NAME}:{shape}",
        f"--warmUp={warmup_ms}",
        f"--duration={duration_seconds}",
        f"--iterations={iterations}",
    ]


def run_trtexec(
    command: Sequence[str],
) -> str:
    executable = command[0]

    if shutil.which(executable) is None:
        raise RuntimeError(
            f"{executable} was not found on PATH. "
            "Run this command inside a TensorRT "
            "container or NVIDIA TensorRT installation."
        )

    process = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )

    output = (
        process.stdout
        + "\n"
        + process.stderr
    )

    if process.returncode != 0:
        raise RuntimeError(
            "TensorRT command failed:\n"
            f"{output}"
        )

    return output


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    min_batch: int = DEFAULT_MIN_BATCH_SIZE,
    opt_batch: int = DEFAULT_OPT_BATCH_SIZE,
    max_batch: int = DEFAULT_MAX_BATCH_SIZE,
    workspace_mib: int = DEFAULT_WORKSPACE_MIB,
) -> Path:
    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model does not exist: {onnx_path}"
        )

    engine_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = build_engine_command(
        onnx_path=onnx_path,
        engine_path=engine_path,
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
        workspace_mib=workspace_mib,
    )

    run_trtexec(command)

    if not engine_path.is_file():
        raise RuntimeError(
            "TensorRT completed without producing "
            f"an engine: {engine_path}"
        )

    return engine_path


def _parse_float(
    text: str,
    pattern: str,
    metric_name: str,
) -> float:
    match = re.search(
        pattern,
        text,
    )

    if match is None:
        raise ValueError(
            f"Could not parse TensorRT "
            f"{metric_name}"
        )

    return float(
        match.group(1)
    )


def parse_trtexec_benchmark(
    output: str,
    batch_size: int,
) -> TensorRTBenchmark:
    throughput = _parse_float(
        output,
        r"Throughput:\s*([0-9.]+)\s*qps",
        "throughput",
    )

    latency_line = next(
        (
            line
            for line in output.splitlines()
            if (
                "Latency:" in line
                and "mean =" in line
            )
        ),
        None,
    )

    if latency_line is None:
        raise ValueError(
            "Could not find TensorRT latency summary"
        )

    mean = _parse_float(
        latency_line,
        r"mean\s*=\s*([0-9.]+)\s*ms",
        "mean latency",
    )

    median = _parse_float(
        latency_line,
        r"median\s*=\s*([0-9.]+)\s*ms",
        "median latency",
    )

    p95 = _parse_float(
        latency_line,
        (
            r"percentile\(95%\)\s*="
            r"\s*([0-9.]+)\s*ms"
        ),
        "p95 latency",
    )

    p99 = _parse_float(
        latency_line,
        (
            r"percentile\(99%\)\s*="
            r"\s*([0-9.]+)\s*ms"
        ),
        "p99 latency",
    )

    return TensorRTBenchmark(
        batch_size=batch_size,
        throughput_qps=throughput,
        latency_mean_ms=mean,
        latency_median_ms=median,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
    )


def benchmark_tensorrt_engine(
    engine_path: Path,
    *,
    batch_size: int = 1,
    warmup_ms: int = 500,
    duration_seconds: int = 10,
    iterations: int = 100,
) -> TensorRTBenchmark:
    if not engine_path.is_file():
        raise FileNotFoundError(
            f"TensorRT engine does not exist: "
            f"{engine_path}"
        )

    command = benchmark_engine_command(
        engine_path=engine_path,
        batch_size=batch_size,
        warmup_ms=warmup_ms,
        duration_seconds=duration_seconds,
        iterations=iterations,
    )

    output = run_trtexec(
        command
    )

    return parse_trtexec_benchmark(
        output=output,
        batch_size=batch_size,
    )