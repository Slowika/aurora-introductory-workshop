"""Shared test fixtures automatically used in tests in this directory and below."""

from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib as mpl
import mlflow
import pandas as pd
import pytest
import torch
import xarray as xr
from aurora import AuroraSmallPretrained, Batch, Metadata

from setup.components.common.constants import (
    ATMOS_LEVELS,
    ATMOS_VAR_MAP,
    STATIC_VAR_MAP,
    SURF_VAR_MAP,
)

mpl.use("Agg")
CHECKPOINT_PATH = Path("tests/models/aurora-0.25-small-pretrained.ckpt")
BASE_DATE = datetime(2025, 1, 1, tzinfo=UTC)
N_TIMESTAMPS = 10  # sufficient no. for inference and fine-tuning

requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason=(
        f"Checkpoint not found at {CHECKPOINT_PATH}. Download with:\n"
        f"huggingface-cli download microsoft/aurora {CHECKPOINT_PATH.name} --local-dir "
        f"{CHECKPOINT_PATH.parent}\nor manually from\n"
        "https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-small-pretrained.ckpt"
    ),
)


def _redirect_cuda(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Replace any 'cuda' device reference with 'cpu'."""
    args = tuple(
        "cpu" if isinstance(a, (str, torch.device)) and "cuda" in str(a) else a
        for a in args
    )
    if "device" in kwargs and "cuda" in str(kwargs["device"]):
        kwargs = {**kwargs, "device": "cpu"}
    return args, kwargs


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect all CUDA device placement to CPU."""
    _real_tensor_to = torch.Tensor.to
    _real_module_to = torch.nn.Module.to
    _real_randn = torch.randn

    def _tensor_to(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        args, kwargs = _redirect_cuda(args, kwargs)
        return _real_tensor_to(self, *args, **kwargs)

    def _module_to(self: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.nn.Module:  # noqa: ANN401
        args, kwargs = _redirect_cuda(args, kwargs)
        return _real_module_to(self, *args, **kwargs)

    def _randn(*args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        _, kwargs = _redirect_cuda((), kwargs)
        return _real_randn(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", _tensor_to)
    monkeypatch.setattr(torch.nn.Module, "to", _module_to)
    monkeypatch.setattr(torch, "randn", _randn)


@pytest.fixture
def make_batch() -> Callable[[datetime], Batch]:
    """Minimal CPU batch for a given datetime factory."""

    def _make(time: datetime = BASE_DATE) -> Batch:
        levels = (500, 700, 850)
        lats = torch.linspace(90, -90, 4)
        lons = torch.linspace(0, 360, 8 + 1)[:-1]
        return Batch(
            surf_vars={
                v: torch.randn(1, 1, len(lats), len(lons))
                for v in SURF_VAR_MAP.values()
            },
            static_vars={
                v: torch.randn(len(lats), len(lons)) for v in STATIC_VAR_MAP.values()
            },
            atmos_vars={
                v: torch.randn(1, 1, len(levels), len(lats), len(lons))
                for v in ATMOS_VAR_MAP.values()
            },
            metadata=Metadata(
                lat=lats,
                lon=lons,
                time=(time,),
                atmos_levels=levels,
            ),
        )

    return _make


@pytest.fixture
def sample_batch(make_batch: Callable[[datetime], Batch]) -> Batch:
    """Single timestamp Batch at BASE_DATE."""
    return make_batch(BASE_DATE)


@pytest.fixture
def target_batch(make_batch: Callable[[datetime], Batch]) -> Batch:
    """Single timestamp Batch at BASE_DATE + 6h, for use as ground-truth target."""
    return make_batch(BASE_DATE + timedelta(hours=6))


@pytest.fixture
def run_dir(tmp_path: Path) -> Generator[Path]:
    """Temporary working directory for component outputs and MLflow tracking."""
    mlflow_uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    yield tmp_path
    mlflow.end_run()


@pytest.fixture
def use_small_model() -> Generator[None]:
    """Patch AuroraPretrained with AuroraSmallPretrained for test model loading."""
    with patch(
        "setup.components.common.utils.AuroraPretrained",
        AuroraSmallPretrained,
    ):
        yield


@pytest.fixture(scope="session")
def era5_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a minimal GCP ERA5-schema Zarr."""
    zarr_path = tmp_path_factory.mktemp("data") / "era5.zarr"
    # 16 lats x 32 lons, the mininum size accepted by Aurora
    # matches setup.components.common.utils.make_lowres_batch
    lats = torch.linspace(90, -90, 16).numpy()
    lons = torch.linspace(0, 360, 32 + 1)[:-1].numpy()
    n_lat = len(lats)
    n_lon = len(lons)
    times = pd.date_range(
        BASE_DATE.replace(tzinfo=None) - timedelta(hours=6),
        periods=N_TIMESTAMPS,
        freq="6h",
    )
    surf_vars = {
        k: (
            ("time", "latitude", "longitude"),
            torch.randn(N_TIMESTAMPS, n_lat, n_lon).numpy(),
        ) for k in SURF_VAR_MAP
    }
    static_vars = {
        k: (
            ("time", "latitude", "longitude"),
            torch.randn(N_TIMESTAMPS, n_lat, n_lon).numpy(),
        ) for k in STATIC_VAR_MAP
    }
    atmos_vars = {
        k: (
            ("time", "level", "latitude", "longitude"),
            torch.randn(N_TIMESTAMPS, len(ATMOS_LEVELS), n_lat, n_lon).numpy(),
        ) for k in ATMOS_VAR_MAP
    }
    ds = xr.Dataset(
        data_vars=surf_vars | static_vars | atmos_vars,
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
            "level": list(ATMOS_LEVELS),
        },
    )
    ds.to_zarr(zarr_path, zarr_format=2)
    return zarr_path


def get_dataset_time_range(data_path: str | Path) -> tuple[datetime, datetime]:
    """Extract the second and last datetimes from a dataset."""
    ds = xr.open_dataset(data_path, engine="zarr")
    start_date = pd.to_datetime(ds.time.to_numpy()[1]).to_pydatetime()
    end_date = pd.to_datetime(ds.time.to_numpy()[-1]).to_pydatetime()
    return start_date, end_date
