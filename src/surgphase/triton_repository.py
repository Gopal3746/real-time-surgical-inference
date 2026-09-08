import shutil
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_NAME = "surgical_phase"
DEFAULT_MODEL_VERSION = 1
DEFAULT_MAX_BATCH_SIZE = 32
DEFAULT_INSTANCE_COUNT = 1

INPUT_NAME = "input"
OUTPUT_NAME = "logits"

INPUT_DIMS = (
    3,
    224,
    224,
)

OUTPUT_DIMS = (7,)


@dataclass(frozen=True)
class TritonRepositoryPaths:
    repository: Path
    model_directory: Path
    version_directory: Path
    config_path: Path
    engine_path: Path


def validate_model_name(
    model_name: str,
) -> None:
    if not model_name:
        raise ValueError(
            "Model name cannot be empty"
        )

    if "/" in model_name or "\\" in model_name:
        raise ValueError(
            "Model name cannot contain path separators"
        )


def validate_model_version(
    version: int,
) -> None:
    if version <= 0:
        raise ValueError(
            "Model version must be positive"
        )


def render_triton_config(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    instance_count: int = DEFAULT_INSTANCE_COUNT,
    max_queue_delay_microseconds: int = 0,
) -> str:
    validate_model_name(
        model_name
    )

    if max_batch_size <= 0:
        raise ValueError(
            "Maximum batch size must be positive"
        )

    if instance_count <= 0:
        raise ValueError(
            "Instance count must be positive"
        )

    if max_queue_delay_microseconds < 0:
        raise ValueError(
            "Queue delay cannot be negative"
        )

    if max_queue_delay_microseconds == 0:
        dynamic_batching = (
            "dynamic_batching { }"
        )
    else:
        dynamic_batching = (
            "dynamic_batching {\n"
            "  max_queue_delay_microseconds: "
            f"{max_queue_delay_microseconds}\n"
            "}"
        )

    return (
        f'name: "{model_name}"\n'
        'platform: "tensorrt_plan"\n'
        f"max_batch_size: {max_batch_size}\n"
        "\n"
        "input [\n"
        "  {\n"
        f'    name: "{INPUT_NAME}"\n'
        "    data_type: TYPE_FP32\n"
        "    format: FORMAT_NCHW\n"
        "    dims: [ 3, 224, 224 ]\n"
        "  }\n"
        "]\n"
        "\n"
        "output [\n"
        "  {\n"
        f'    name: "{OUTPUT_NAME}"\n'
        "    data_type: TYPE_FP32\n"
        "    dims: [ 7 ]\n"
        "  }\n"
        "]\n"
        "\n"
        "instance_group [\n"
        "  {\n"
        f"    count: {instance_count}\n"
        "    kind: KIND_GPU\n"
        "  }\n"
        "]\n"
        "\n"
        f"{dynamic_batching}\n"
    )


def get_repository_paths(
    repository: Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    version: int = DEFAULT_MODEL_VERSION,
) -> TritonRepositoryPaths:
    validate_model_name(
        model_name
    )

    validate_model_version(
        version
    )

    model_directory = (
        repository / model_name
    )

    version_directory = (
        model_directory / str(version)
    )

    return TritonRepositoryPaths(
        repository=repository,
        model_directory=model_directory,
        version_directory=version_directory,
        config_path=(
            model_directory / "config.pbtxt"
        ),
        engine_path=(
            version_directory / "model.plan"
        ),
    )


def prepare_triton_repository(
    engine_source: Path,
    repository: Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    version: int = DEFAULT_MODEL_VERSION,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    instance_count: int = DEFAULT_INSTANCE_COUNT,
    max_queue_delay_microseconds: int = 0,
) -> TritonRepositoryPaths:
    if not engine_source.is_file():
        raise FileNotFoundError(
            f"TensorRT engine does not exist: "
            f"{engine_source}"
        )

    paths = get_repository_paths(
        repository=repository,
        model_name=model_name,
        version=version,
    )

    paths.version_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = render_triton_config(
        model_name=model_name,
        max_batch_size=max_batch_size,
        instance_count=instance_count,
        max_queue_delay_microseconds=(
            max_queue_delay_microseconds
        ),
    )

    paths.config_path.write_text(
        config,
        encoding="utf-8",
    )

    shutil.copy2(
        engine_source,
        paths.engine_path,
    )

    return paths