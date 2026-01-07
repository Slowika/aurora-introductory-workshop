"""Aurora inference Azure ML Component logic."""

import argparse
from datetime import datetime

import numpy as np
import torch
import xarray as xr
from aurora import Batch, rollout
from dask import config as dask_config

try:
    from common.utils import (
        create_logger,
        load_batch_from_asset,
        load_model,
        make_lowres_batch,
    )
except ImportError:
    from setup.components.common.utils import (
        create_logger,
        load_batch_from_asset,
        load_model,
        make_lowres_batch,
    )

LOG = create_logger()
dask_config.set(scheduler="threads", num_workers=10)


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
    return xr.Dataset(
        data_vars={
            "2t": (("time", "lat", "lon"), batch.surf_vars["2t"].squeeze(0).numpy()),
            "10u": (("time", "lat", "lon"), batch.surf_vars["10u"].squeeze(0).numpy()),
            "10v": (("time", "lat", "lon"), batch.surf_vars["10v"].squeeze(0).numpy()),
            "msl": (("time", "lat", "lon"), batch.surf_vars["msl"].squeeze(0).numpy()),
            "lsm": (("lat", "lon"), batch.static_vars["lsm"].numpy()),
            "z": (("lat", "lon"), batch.static_vars["z"].numpy()),
            "slt": (("lat", "lon"), batch.static_vars["slt"].numpy()),
            "z_atmos": (
                ("time", "level", "lat", "lon"),
                batch.atmos_vars["z"].squeeze(0).numpy(),
            ),
            "u": (
                ("time", "level", "lat", "lon"),
                batch.atmos_vars["u"].squeeze(0).numpy(),
            ),
            "v": (
                ("time", "level", "lat", "lon"),
                batch.atmos_vars["v"].squeeze(0).numpy(),
            ),
            "t": (
                ("time", "level", "lat", "lon"),
                batch.atmos_vars["t"].squeeze(0).numpy(),
            ),
            "q": (
                ("time", "level", "lat", "lon"),
                batch.atmos_vars["q"].squeeze(0).numpy(),
            ),
        },
        coords={
            "lat": (("lat",), batch.metadata.lat.numpy()),
            "lon": (("lon",), batch.metadata.lon.numpy()),
            "time": (("time",), [batch.metadata.time[0]]),
            "level": (("level",), np.array(batch.metadata.atmos_levels)),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aurora Inference Component")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        help="Path to the initial state data, if not a dry run.",
    )
    parser.add_argument(
        "--start_datetime",
        type=datetime.fromisoformat,
        help="Start ISO 8601 format datetime e.g. 2026-01-01T00:00:00.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of autoregressive steps to perform during inference, default 1.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="Path to which output predictions will be written.",
    )
    args = parser.parse_args()

    LOG.info("Loading model: path=%s", args.model)
    model = load_model(args.model)
    LOG.info("Loaded model.")

    if args.data is None:
        LOG.info("Generating low resolution dummy batch.")
        batch = make_lowres_batch()
        LOG.info("Generated low resolution dummy batch.")
    else:
        LOG.info("Loading data: path=%s, start=%s", args.data, args.start_datetime)
        batch = load_batch_from_asset(args.data, args.start_datetime)
        LOG.info("Loaded data.")

    LOG.info("Starting inference: start=%s, steps=%d", args.start_datetime, args.steps)
    with torch.inference_mode():
        datasets = []
        for pred in rollout(model, batch, steps=args.steps):
            LOG.info(
                "Predicted step: no=%s, timestamp=%s",
                pred.metadata.rollout_step,
                pred.metadata.time,
            )
            datasets.append(batch_to_xarray(pred))
    LOG.info("Completed inference.")

    LOG.info("Concatenating and writing predictions: path=%s", args.predictions)
    preds_ds = xr.concat(datasets, dim="time")
    preds_ds.to_zarr(args.predictions)
    LOG.info("Wrote predictions: path=%s", args.predictions)
