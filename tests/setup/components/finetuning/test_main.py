"""Integration tests for the fine-tuning component.

Requires the small Aurora checkpoint. See [README](../../../README.md#running-tests) for
setup instructions.

Exercises both short-lead and autoregressive rollout fine-tuning with test-mode data.

To do:
- Negative test cases (overridden config values, insufficient data, etc.)
- New variables (including atmospheric)
- Additional assertions
- More config variations (lora, etc.)
"""

import runpy
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from setup.components.common.models import DataMode, FinetuneConfig
from tests.conftest import (
    BASE_DATE,
    CHECKPOINT_PATH,
    get_dataset_time_range,
    requires_checkpoint,
)


def _run_finetuning(  # noqa: PLR0913
    cfg: FinetuneConfig,
    model_path: str,
    data_path: str,
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
        "--model", model_path,
        "--data", data_path,
        "--start_datetime", start_date.isoformat(),
        "--end_datetime", end_date.isoformat(),
        "--config", cfg.model_dump_json(),
        "--loss", str(loss_path),
        "--prediction", str(prediction_path),
        "--checkpoint", str(checkpoint_path),
    ]
    runpy.run_module(
        "setup.components.finetuning.main",
        run_name="__main__",
        alter_sys=True,
    )


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
def test_finetune_short_lead_test_mode(run_dir: Path) -> None:
    """Run short-lead fine-tuning with low res in-memory data."""
    loss_path, prediction_path, checkpoint_path = _output_paths(run_dir)
    cfg = FinetuneConfig(mode=DataMode.TEST, type="short", epochs=1)
    _run_finetuning(
        cfg,
        str(CHECKPOINT_PATH),
        "",
        loss_path,
        prediction_path,
        checkpoint_path,
    )
    _assert_outputs(loss_path, prediction_path, checkpoint_path, cfg.epochs)


@requires_checkpoint
@pytest.mark.usefixtures("use_small_model")
def test_finetune_short_lead_data_mode(run_dir: Path, era5_dataset: Path) -> None:
    """Run short-lead fine-tuning with low res on-disk data."""
    loss_path, prediction_path, checkpoint_path = _output_paths(run_dir)
    cfg = FinetuneConfig(mode=DataMode.ERA5, type="short", epochs=1)
    _run_finetuning(
        cfg,
        str(CHECKPOINT_PATH),
        str(era5_dataset),
        loss_path,
        prediction_path,
        checkpoint_path,
    )
    _assert_outputs(loss_path, prediction_path, checkpoint_path, cfg.epochs)


@requires_checkpoint
@pytest.mark.usefixtures("use_small_model")
def test_finetune_rollout_test_mode(run_dir: Path) -> None:
    """Run autoregressive rollout fine-tuning with low res in-memory data."""
    loss_path, prediction_path, checkpoint_path = _output_paths(run_dir)
    cfg = FinetuneConfig(mode=DataMode.TEST, type="rollout", epochs=1, rollout_steps=2)
    _run_finetuning(
        cfg,
        str(CHECKPOINT_PATH),
        "",
        loss_path,
        prediction_path,
        checkpoint_path,
    )
    _assert_outputs(loss_path, prediction_path, checkpoint_path, cfg.epochs)


@requires_checkpoint
@pytest.mark.usefixtures("use_small_model")
def test_finetune_rollout_data_mode(run_dir: Path, era5_dataset: Path) -> None:
    """Run autoregressive rollout fine-tuning with low res on-disk data."""
    loss_path, prediction_path, checkpoint_path = _output_paths(run_dir)
    cfg = FinetuneConfig(mode=DataMode.ERA5, type="rollout", epochs=1, rollout_steps=2)
    _run_finetuning(
        cfg,
        str(CHECKPOINT_PATH),
        str(era5_dataset),
        loss_path,
        prediction_path,
        checkpoint_path,
    )
    _assert_outputs(loss_path, prediction_path, checkpoint_path, cfg.epochs)
