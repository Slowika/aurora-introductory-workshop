"""Integration tests for the inference component.

Requires the small Aurora checkpoint. See [README](../../../README.md#running-tests) for
setup instructions.

Exercises both test and ERA5 data modes but not inference of a model with additional,
non-standard fine-tuned variables.

To do:
- Negative test cases (overridden config values, insufficient data, etc.)
- New variables (including atmospheric), requires fine-tuned model checkpoint
- Additional assertions
"""

import runpy
import sys
from pathlib import Path

import pytest
import xarray as xr

from setup.components.common.models import DataMode, InferenceConfig
from tests.conftest import (
    BASE_DATE,
    CHECKPOINT_PATH,
    get_dataset_time_range,
    requires_checkpoint,
)


def _run_inference(
    cfg: InferenceConfig,
    model_path: str,
    data_path: str,
    output_path: Path,
) -> None:
    """Execute inference main via runpy."""
    if data_path:
        start_date, _ = get_dataset_time_range(data_path)
    else:
        start_date = BASE_DATE
    sys.argv = [
        "main",
        "--model", model_path,
        "--data", data_path,
        "--start_datetime", start_date.isoformat(),
        "--config", cfg.model_dump_json(),
        "--predictions", str(output_path),
    ]
    runpy.run_module(
        "setup.components.inference.main",
        run_name="__main__",
        alter_sys=True,
    )


@requires_checkpoint
@pytest.mark.usefixtures("use_small_model")
def test_inference_end_to_end_test_mode(run_dir: Path) -> None:
    """Run inference with low res in-memory data."""
    output_path = run_dir / "predictions.nc"
    cfg = InferenceConfig(mode=DataMode.TEST, steps=2)
    _run_inference(cfg, str(CHECKPOINT_PATH), "", output_path)
    assert output_path.exists()
    ds = xr.open_dataset(output_path)
    assert ds.sizes["time"] == cfg.steps


@requires_checkpoint
@pytest.mark.usefixtures("use_small_model")
def test_inference_end_to_end_data_mode(run_dir: Path, era5_dataset: Path) -> None:
    """Run inference with low res on-disk data."""
    output_path = run_dir / "predictions.nc"
    cfg = InferenceConfig(mode=DataMode.ERA5, steps=2)
    _run_inference(cfg, str(CHECKPOINT_PATH), str(era5_dataset), output_path)
    assert output_path.exists()
    ds = xr.open_dataset(output_path)
    assert ds.sizes["time"] == cfg.steps
