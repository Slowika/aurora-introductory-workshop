"""Aurora inference script.

This script takes a set of arguments through the command line to perform autoregressive
forecasting using a pretrained Aurora model, local initial state date OR synthetic test
data, a starting datetime, and the number of steps to perform. All generated forecasts
are written to a NetCDF file at the specified output path.

Running locally:
    python -m setup.components.inference.main \
        --model <path to local model checkpoint e.g. ./aurora-0.25-pretrained.ckpt> \
        --data <path to local initial state data e.g. ./era5_subset.zarr, optional> \
        --start_datetime <ISO 8601 format datetime e.g. 2026-01-01T00:00:00> \
        --steps <number of inference steps to perform e.g. 10> \
        --predictions <path to output NetCDF file of forecasts e.g. ./fcst.nc>

Running in Azure Machine Learning:
    See setup/components/inference/component.py for definition and deployment, and
    notebooks/0_aurora_workshop.ipynb for example usage.
"""

import argparse
from datetime import datetime

import torch
import xarray as xr
from aurora import rollout
from dask import config as dask_config

# NOTE: enable imports in local and remote environments
try:
    from common.utils import (
        batch_to_xarray,
        create_logger,
        load_batch_from_asset,
        load_model,
        make_lowres_batch,
    )
except ImportError:
    from setup.components.common.utils import (
        batch_to_xarray,
        create_logger,
        load_batch_from_asset,
        load_model,
        make_lowres_batch,
    )

LOG = create_logger()
# limit threads for writing Zarr to mounted storage
dask_config.set(scheduler="threads", num_workers=10)

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
        help="Path to the initial state data, if not a test run.",
    )
    parser.add_argument(
        "--start_datetime",
        type=datetime.fromisoformat,
        help="Start ISO 8601 format datetime e.g. 2026-01-01T00:00:00.",
    )
    parser.add_argument(
        "--steps",
        type=lambda x: max(int(x), 1),
        help="Number of inference steps to perform, default and minimum 1.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "test"],
        help=(
            "Whether to run in eval mode with real data or test mode with synthetic "
            "data."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="Path to which output predictions will be written.",
    )
    args = parser.parse_args()

    if args.mode == "test":
        LOG.info("Test mode enabled, using synthetic test data as initial state.")
        batch = make_lowres_batch(args.start_datetime)
    else:
        LOG.info(
            "Eval mode enabled, using real data as initial state: path=%s",
            args.data,
        )
        batch = load_batch_from_asset(args.data, args.start_datetime)

    LOG.info("Loading model: path=%s", args.model)
    model = load_model(args.model, train=False, lora=False)

    LOG.info("Starting inference: start=%s, steps=%d", args.start_datetime, args.steps)
    with torch.inference_mode():
        datasets = []
        for pred in rollout(model, batch, steps=args.steps):
            LOG.info(
                "Inference step complete: no=%s, timestamp=%s",
                pred.metadata.rollout_step,
                pred.metadata.time[0].isoformat(timespec="hours"),
            )
            datasets.append(batch_to_xarray(pred))
    LOG.info("Completed %d inference steps.", args.steps)

    LOG.info("Concatenating and writing predictions: path=%s", args.predictions)
    preds_ds = xr.concat(datasets, dim="time")
    preds_ds.to_netcdf(args.predictions)

    LOG.info("Done!")
