"""Integration tests for the fine-tuning component.

Requires the small Aurora checkpoint. See [README](../../../README.md#testing) for
setup instructions.

To do:
- New variables (including atmospheric)
- More assertions and config variations
"""

import logging
import runpy
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from setup.common.constants import FINETUNE_MODULE
from setup.components.common.models import AuroraConfig, DataMode, FinetuneConfig
from tests.conftest import (
    BASE_DATE,
    CHECKPOINT_PATH,
    N_TIMESTAMPS,
    get_dataset_time_range,
    requires_checkpoint,
)


def _run_finetuning(  # noqa: PLR0913
    cfg: FinetuneConfig,
    model_path: Path,
    data_path: str | Path,
    loss_path: Path,
    prediction_path: Path,
    checkpoint_path: Path,
) -> None:
    """Execute fine-tuning main via runpy."""
    if data_path:
        start_date, end_date = get_dataset_time_range(data_path)
    else:
        start_date = BASE_DATE
        # end_datetime must provide enough timestamps for the fine-tuning type
        end_date = BASE_DATE + timedelta(hours=24)
    sys.argv = [
        "main",
        "--model", str(model_path),
        "--data", str(data_path),
        "--start_datetime", start_date.isoformat(),
        "--end_datetime", end_date.isoformat(),
        "--config", cfg.model_dump_json(),
        "--loss", str(loss_path),
        "--prediction", str(prediction_path),
        "--checkpoint", str(checkpoint_path),
    ]
    runpy.run_module(FINETUNE_MODULE, run_name="__main__", alter_sys=True)


def _output_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    """Build standard output paths for fine-tuning tests."""
    return run_dir / "losses.npy", run_dir / "prediction.nc", run_dir / "finetuned.ckpt"


def _assert_outputs(
    loss_path: Path,
    prediction_path: Path,
    checkpoint_path: Path,
    epochs: int,
) -> None:
    """Verify fine-tuning produced valid outputs."""
    losses = np.load(loss_path)
    assert losses.shape == (epochs,)
    assert all(np.isfinite(losses))

    ds = xr.open_dataset(prediction_path)
    assert "time" in ds.dims

    assert checkpoint_path.exists()


@requires_checkpoint
@pytest.mark.usefixtures("use_small_model")
def test_finetune_short_lead_test_mode(
    run_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test short-lead fine-tuning with low res in-memory data."""
    loss_path, prediction_path, checkpoint_path = _output_paths(run_dir)
    cfg = FinetuneConfig(mode=DataMode.TEST, type="short", epochs=1)
    with caplog.at_level(logging.INFO, logger="__main__"):
        _run_finetuning(
        cfg,
        CHECKPOINT_PATH,
        "",
        loss_path,
        prediction_path,
        checkpoint_path,
    )
    _assert_outputs(loss_path, prediction_path, checkpoint_path, cfg.epochs)
    assert "strict=True" in caplog.text
    assert "lora=False" in caplog.text
    assert "lora_only=False" in caplog.text


@requires_checkpoint
@pytest.mark.usefixtures("use_small_model")
def test_finetune_short_lead_test_mode_lora(
    run_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test short-lead fine-tuning with low res in-memory data and LoRA."""
    loss_path, prediction_path, checkpoint_path = _output_paths(run_dir)
    cfg = FinetuneConfig(
        mode=DataMode.TEST,
        aurora_config=AuroraConfig(use_lora=True),
        type="short",
        epochs=1,
    )
    with caplog.at_level(logging.INFO, logger="__main__"):
        _run_finetuning(
            cfg,
            CHECKPOINT_PATH,
            "",
            loss_path,
            prediction_path,
            checkpoint_path,
        )
    _assert_outputs(loss_path, prediction_path, checkpoint_path, cfg.epochs)
    assert "strict=False" in caplog.text
    assert "lora=True" in caplog.text
    assert "lora_only=True" in caplog.text


@requires_checkpoint
@pytest.mark.usefixtures("use_small_model")
def test_finetune_short_lead_data_mode(run_dir: Path, era5_dataset: Path) -> None:
    """Test short-lead fine-tuning with low res on-disk data."""
    loss_path, prediction_path, checkpoint_path = _output_paths(run_dir)
    cfg = FinetuneConfig(mode=DataMode.ERA5, type="short", epochs=1)
    _run_finetuning(
        cfg,
        CHECKPOINT_PATH,
        era5_dataset,
        loss_path,
        prediction_path,
        checkpoint_path,
    )
    _assert_outputs(loss_path, prediction_path, checkpoint_path, cfg.epochs)


@requires_checkpoint
@pytest.mark.usefixtures("use_small_model")
def test_finetune_short_raises_insufficient_timestamps(
    run_dir: Path,
    era5_dataset: Path,
) -> None:
    """Test fine-tuning raises when epochs exceeds usable timestamps."""
    loss_path, prediction_path, checkpoint_path = _output_paths(run_dir)
    cfg = FinetuneConfig(mode=DataMode.ERA5, type="short", epochs=N_TIMESTAMPS + 1)
    with pytest.raises(ValueError, match="Insufficient timestamps for epochs"):
        _run_finetuning(
            cfg,
            CHECKPOINT_PATH,
            era5_dataset,
            loss_path,
            prediction_path,
            checkpoint_path,
        )


@requires_checkpoint
@pytest.mark.usefixtures("use_small_model")
def test_finetune_rollout_test_mode(run_dir: Path) -> None:
    """Test autoregressive rollout fine-tuning with low res in-memory data."""
    loss_path, prediction_path, checkpoint_path = _output_paths(run_dir)
    cfg = FinetuneConfig(mode=DataMode.TEST, type="rollout", epochs=1, rollout_steps=2)
    _run_finetuning(
        cfg,
        CHECKPOINT_PATH,
        "",
        loss_path,
        prediction_path,
        checkpoint_path,
    )
    _assert_outputs(loss_path, prediction_path, checkpoint_path, cfg.epochs)


@requires_checkpoint
@pytest.mark.usefixtures("use_small_model")
def test_finetune_rollout_data_mode(run_dir: Path, era5_dataset: Path) -> None:
    """Test autoregressive rollout fine-tuning with low res on-disk data."""
    loss_path, prediction_path, checkpoint_path = _output_paths(run_dir)
    cfg = FinetuneConfig(mode=DataMode.ERA5, type="rollout", epochs=1, rollout_steps=2)
    _run_finetuning(
        cfg,
        CHECKPOINT_PATH,
        era5_dataset,
        loss_path,
        prediction_path,
        checkpoint_path,
    )
    _assert_outputs(loss_path, prediction_path, checkpoint_path, cfg.epochs)
