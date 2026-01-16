"""Common utility logic."""

import logging
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import torch
import xarray as xr
from aurora import AuroraPretrained, Batch, Metadata

try:
    from common.constants import ATMOS_VAR_MAP, SFC_VAR_MAP, STATIC_VAR_MAP
except ImportError:
    from setup.components.common.constants import (
        ATMOS_VAR_MAP,
        SFC_VAR_MAP,
        STATIC_VAR_MAP,
    )


def create_logger() -> logging.Logger:
    """Create a configured logger.

    Returns
    -------
    logging.Logger
        Configured logger.

    """
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_model(model_path: str, **finetune_cfg: dict[str, Any]) -> AuroraPretrained:
    """Load the Aurora pre-trained model from a local checkpoint.

    Parameters
    ----------
    model_path : str
        Path to the local model checkpoint.
    finetune_cfg : dict[str, Any]
        Additional keyword arguments to pass to the AuroraPretrained constructor.

    Returns
    -------
    aurora.AuroraPretrained
        Loaded Aurora model.

    """
    strict = bool(finetune_cfg.pop("strict", True))
    model = AuroraPretrained(
        use_lora=bool(finetune_cfg.pop("use_lora", False)),
        **finetune_cfg,
    )
    model.load_checkpoint_local(model_path, strict)
    train = bool(finetune_cfg)
    if train and hasattr(model, "configure_activation_checkpointing"):
        model.configure_activation_checkpointing()
    return model.to("cuda").train(mode=train)


def make_lowres_batch(
    start_datetime: datetime,
    surf_vars: dict[str, Any] = SFC_VAR_MAP,
    static_vars: dict[str, Any] = STATIC_VAR_MAP,
    atmos_vars: dict[str, Any] = ATMOS_VAR_MAP,
    times: int = 2,
    **_: dict,
) -> Batch:
    """Create a small 17x32 batch on the given device.

    Parameters
    ----------
    start_datetime : datetime.datetime
        Start datetime for the batch.
    surf_vars : dict[str, Any], default = SFC_VAR_MAP
        Surface variable mapping.
    static_vars : dict[str, Any], default = STATIC_VAR_MAP
        Static variable mapping.
    atmos_vars : dict[str, Any], default = ATMOS_VAR_MAP
        Atmospheric variable mapping.
    times : int, default = 2
        Number of time steps in the batch.

    Returns
    -------
    aurora.Batch
        Generated low-resolution dummy Batch.

    """
    return Batch(
        surf_vars={
            k: torch.randn(1, times, 16, 32, device="cuda") for k in surf_vars.values()
        },
        static_vars={
            k: torch.randn(16, 32, device="cuda") for k in static_vars.values()
        },
        atmos_vars={
            k: torch.randn(1, times, 4, 16, 32, device="cuda")
            for k in atmos_vars.values()
        },
        metadata=Metadata(
            lat=torch.linspace(90, -90, 16),
            lon=torch.linspace(0, 360, 32 + 1)[:-1],
            time=(start_datetime,),
            atmos_levels=(100, 250, 500, 850),
        ),
    )


def load_batch_from_asset(  # noqa: PLR0913
    data_path: str,
    start_datetime: datetime,
    surf_vars: dict[str, Any] = SFC_VAR_MAP,
    static_vars: dict[str, Any] = STATIC_VAR_MAP,
    atmos_vars: dict[str, Any] = ATMOS_VAR_MAP,
    times: int = 2,
) -> Batch:
    """Load a Batch from a local (mounted or downloaded) Azure ML data asset.

    Parameters
    ----------
    data_path : str
        Path to the data asset.
    start_datetime : datetime.datetime
        Start datetime for the batch.
    surf_vars : dict[str, Any], default = SFC_VAR_MAP
        Surface variable mapping.
    static_vars : dict[str, Any], default = STATIC_VAR_MAP
        Static variable mapping.
    atmos_vars : dict[str, Any], default = ATMOS_VAR_MAP
        Atmospheric variable mapping.
    times : int, default = 2
        Number of time steps in the batch.

    Returns
    -------
    aurora.Batch
        Loaded Batch object.

    """
    ds = xr.open_dataset(data_path, engine="zarr", chunks={})
    time_indexer = [start_datetime]
    for _ in range(1, times):
        time_indexer.insert(0, time_indexer[0] - timedelta(hours=6))
    ds_sel = ds.sel(time=time_indexer)
    return Batch(
        # produces Tensors of shape [1, times, 720, lons]
        surf_vars={
            v: torch.from_numpy(ds_sel[k].values[:, :720, :]).unsqueeze(0).to("cuda")
            for k, v in surf_vars.items()
        },
        # produces Tensors of shape [720, lons]
        static_vars={
            v: torch.from_numpy(ds_sel[k].isel(time=-1).values[:720, :]).to("cuda")
            for k, v in static_vars.items()
        },
        # produces Tensors of shape [1, times, levels, 720, lons]
        atmos_vars={
            v: torch.from_numpy(ds_sel[k].values[:, :, :720, :]).unsqueeze(0).to("cuda")
            for k, v in atmos_vars.items()
        },
        metadata=Metadata(
            lat=torch.from_numpy(ds_sel["latitude"].values[:720]).to("cuda"),
            lon=torch.from_numpy(ds_sel["longitude"].values).to("cuda"),
            time=(start_datetime,),
            atmos_levels=ds_sel["level"].values.tolist(),
        ),
    )


def batch_to_xarray(batch: Batch) -> xr.Dataset:
    """Convert a Batch to an xarray Dataset.

    Parameters
    ----------
    batch : aurora.Batch
        Batch to convert.

    Returns
    -------
    xarray.Dataset
        Converted xarray Dataset.

    """
    surf_vars = {
        k: (("time", "lat", "lon"), v.squeeze(0).detach().cpu().numpy())
        for k, v in batch.surf_vars.items()
    }
    static_vars = {
        k: (("lat", "lon"), v.detach().cpu().numpy())
        for k, v in batch.static_vars.items()
    }
    # append _atmos to avoid name clashes / overwrites, solely for static and atmos "z"
    atmos_vars = {
        f"{k}_atmos": (
            ("time", "level", "lat", "lon"),
            v.squeeze(0).detach().cpu().numpy(),
        )
        for k, v in batch.atmos_vars.items()
    }
    return xr.Dataset(
        data_vars=surf_vars | static_vars | atmos_vars,
        coords={
            "lat": (("lat",), batch.metadata.lat.detach().cpu().numpy()),
            "lon": (("lon",), batch.metadata.lon.detach().cpu().numpy()),
            "time": (("time",), [batch.metadata.time[0]]),
            "level": (("level",), np.array(batch.metadata.atmos_levels)),
        },
    )

# mapping of data modes to batch creation functions
BATCH_FNS: dict[str, Callable[..., Batch]] = {
    "test": make_lowres_batch,
    "eval": load_batch_from_asset,
}
