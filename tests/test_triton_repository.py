from pathlib import Path

import pytest

from surgphase.triton_repository import (
    get_repository_paths,
    prepare_triton_repository,
    render_triton_config,
)


def test_render_triton_config() -> None:
    config = render_triton_config()

    assert (
        'name: "surgical_phase"'
        in config
    )

    assert (
        'platform: "tensorrt_plan"'
        in config
    )

    assert (
        "max_batch_size: 32"
        in config
    )

    assert (
        'name: "input"'
        in config
    )

    assert (
        "data_type: TYPE_FP32"
        in config
    )

    assert (
        "format: FORMAT_NCHW"
        in config
    )

    assert (
        "dims: [ 3, 224, 224 ]"
        in config
    )

    assert (
        'name: "logits"'
        in config
    )

    assert (
        "dims: [ 7 ]"
        in config
    )

    assert (
        "kind: KIND_GPU"
        in config
    )

    assert (
        "dynamic_batching { }"
        in config
    )


def test_render_config_with_queue_delay() -> None:
    config = render_triton_config(
        max_queue_delay_microseconds=100
    )

    assert (
        "max_queue_delay_microseconds: 100"
        in config
    )


def test_render_config_rejects_invalid_batch() -> None:
    with pytest.raises(
        ValueError,
        match="batch size",
    ):
        render_triton_config(
            max_batch_size=0
        )


def test_repository_paths() -> None:
    paths = get_repository_paths(
        repository=Path(
            "deployment/model_repository"
        ),
        model_name="surgical_phase",
        version=3,
    )

    assert paths.config_path == Path(
        "deployment/model_repository/"
        "surgical_phase/config.pbtxt"
    )

    assert paths.engine_path == Path(
        "deployment/model_repository/"
        "surgical_phase/3/model.plan"
    )


def test_prepare_triton_repository(
    tmp_path: Path,
) -> None:
    engine_source = (
        tmp_path / "source.plan"
    )

    engine_source.write_bytes(
        b"fake-tensorrt-engine"
    )

    repository = (
        tmp_path / "model_repository"
    )

    paths = prepare_triton_repository(
        engine_source=engine_source,
        repository=repository,
    )

    assert paths.config_path.is_file()
    assert paths.engine_path.is_file()

    assert (
        paths.engine_path.read_bytes()
        == b"fake-tensorrt-engine"
    )

    config = paths.config_path.read_text(
        encoding="utf-8"
    )

    assert (
        'platform: "tensorrt_plan"'
        in config
    )

    assert (
        "max_batch_size: 32"
        in config
    )


def test_prepare_rejects_missing_engine(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        prepare_triton_repository(
            engine_source=(
                tmp_path / "missing.plan"
            ),
            repository=(
                tmp_path / "repository"
            ),
        )


def test_model_version_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="version",
    ):
        get_repository_paths(
            repository=Path("repository"),
            version=0,
        )