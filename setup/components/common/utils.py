"""Common utility logic."""

import logging
import sys
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import torch
import xarray as xr
from aurora import AuroraPretrained, Batch, Metadata

from .constants import ATMOS_VAR_MAP, SFC_VAR_MAP, STATIC_VAR_MAP


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


def load_model(
    model_path: str,
    *,
    train: bool,
    lora: bool,
    **aurora_kwargs: dict[str, Any],
) -> AuroraPretrained:
    """Load the Aurora pre-trained model from a local checkpoint.

    Parameters
    ----------
    model_path : str
        Path to the local model checkpoint.
    train : bool
        Whether to set the model to training (True) or eval (False) mode.
    lora : bool
        Whether to enable Low-Rank Adaptation (LoRA).
    aurora_kwargs : dict[str, Any]
        Additional keyword arguments to pass to the AuroraPretrained constructor.

    Returns
    -------
    aurora.AuroraPretrained
        Loaded Aurora model.

    """
    model = AuroraPretrained(use_lora=lora, **aurora_kwargs)
    model.load_checkpoint_local(model_path)
    return model.to("cuda").train(mode=train)


def make_lowres_batch(start_datetime: datetime) -> Batch:
    """Create a small 17x32 batch on the given device.

    Parameters
    ----------
    start_datetime : datetime.datetime
        Start datetime for the batch.

    Returns
    -------
    aurora.Batch
        Generated low-resolution dummy Batch.

    """
    return Batch(
        surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 17),
            lon=torch.linspace(0, 360, 32 + 1)[:-1],
            time=(start_datetime,),
            atmos_levels=(100, 250, 500, 850),
        ),
    )


def load_batch_from_asset(data_path: str, start_datetime: datetime) -> Batch:
    """Load a Batch from a local (mounted or downloaded) Azure ML data asset.

    Parameters
    ----------
    data_path : str
        Path to the data asset.
    start_datetime : datetime.datetime
        Start datetime for the batch.

    Returns
    -------
    aurora.Batch
        Loaded Batch object.

    """
    ds = xr.open_dataset(data_path, engine="zarr", chunks={})
    prev_datetime = start_datetime - timedelta(hours=6)
    # select given datetime and that 6h prior
    ds_sel = ds.sel(time=[prev_datetime, start_datetime])
    return Batch(
        # produces Tensors of shape [1, 2, lats, lons]
        surf_vars={
            v: torch.from_numpy(ds_sel[k].values).unsqueeze(0)
            for k, v in SFC_VAR_MAP.items()
        },
        # produces Tensors of shape [lats, lons]
        static_vars={
            v: torch.from_numpy(ds_sel[k].isel(time=-1).values)
            for k, v in STATIC_VAR_MAP.items()
        },
        # produces Tensors of shape [1, 2, levels, lats, lons]
        atmos_vars={
            v: torch.from_numpy(ds_sel[k].values).unsqueeze(0)
            for k, v in ATMOS_VAR_MAP.items()
        },
        metadata=Metadata(
            lat=torch.from_numpy(ds_sel["latitude"].values),
            lon=torch.from_numpy(ds_sel["longitude"].values),
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
        k: (("time", "lat", "lon"), v.squeeze(0).cpu().numpy())
        for k, v in batch.surf_vars.items()
    }
    static_vars = {
        k: (("lat", "lon"), v.cpu().numpy()) for k, v in batch.static_vars.items()
    }
    atmos_vars = {
        f"{k}_atmos": (("time", "level", "lat", "lon"), v.squeeze(0).cpu().numpy())
        for k, v in batch.atmos_vars.items()
    }
    return xr.Dataset(
        data_vars=surf_vars | static_vars | atmos_vars,
        coords={
            "lat": (("lat",), batch.metadata.lat.cpu().numpy()),
            "lon": (("lon",), batch.metadata.lon.cpu().numpy()),
            "time": (("time",), [batch.metadata.time[0]]),
            "level": (("level",), np.array(batch.metadata.atmos_levels)),
        },
    )
