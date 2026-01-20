"""Aurora inference script.

This script takes a set of arguments through the command line to perform autoregressive
forecasting using a pretrained Aurora model, local initial state date OR synthetic test
data, a starting datetime, and the number of steps to perform. All generated forecasts
are written to a NetCDF file at the specified output path.

Running locally:
    python -m setup.components.inference.main \
        --model <path to local model checkpoint e.g. ./aurora-0.25-pretrained.ckpt> \
        --data <path to local initial state data e.g. ./era5_subset.zarr> \
        --start_datetime <ISO 8601 format datetime e.g. 2026-01-01T00:00:00> \
        --config <JSON-formatted string of inference configuration> \
        --predictions <path to output NetCDF file of forecasts e.g. ./fcst.nc>

Running in Azure Machine Learning:
    See setup/components/inference/component.py for definition and deployment, and
    notebooks/0_aurora_workshop.ipynb for example usage.
"""

import argparse
import json
from datetime import datetime

import torch
import xarray as xr
from aurora import rollout

# NOTE: enable imports in local and remote environments
try:
    from common.utils import (
        batch_to_xarray,
        create_logger,
        load_model,
        register_new_variables,
        validate_common_config,
    )
except ImportError:
    from setup.components.common.utils import (
        batch_to_xarray,
        create_logger,
        load_model,
        register_new_variables,
        validate_common_config,
    )

LOG = create_logger()

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
        help="Path to the training data, ignored if configured mode is test.",
    )
    parser.add_argument(
        "--start_datetime",
        type=datetime.fromisoformat,
        help="Start ISO 8601 format datetime e.g. 2026-01-01T00:00:00.",
    )
    parser.add_argument(
        "--config",
        type=json.loads,
        help="JSON string of inference configuration.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="Path to which output predictions will be written.",
    )
    args = parser.parse_args()

    batch_fn, inf_steps = validate_common_config(args.config)
    var_map, var_cfg = register_new_variables(args.config.get("extra_variables", {}))
    init_batch = batch_fn(
        data_path=args.data,
        start_datetime=args.start_datetime,
        **var_map,
    )
    LOG.info("%s mode enabled.", args.config["mode"])

    LOG.info("Loading model: path=%s", args.model)
    model = load_model(args.model, train=False, **var_cfg)
    LOG.info("Loaded model using config: %s", var_cfg)

    LOG.info("Starting inference: start=%s, steps=%d", args.start_datetime, inf_steps)
    with torch.inference_mode():
        datasets = []
        for pred in rollout(model, init_batch, steps=inf_steps):
            LOG.info(
                "Inference step complete: no=%s, timestamp=%s",
                pred.metadata.rollout_step,
                pred.metadata.time[0].isoformat(timespec="hours"),
            )
            datasets.append(batch_to_xarray(pred))
    LOG.info("Completed %d inference steps.", inf_steps)

    LOG.info("Concatenating and writing predictions: path=%s", args.predictions)
    preds_ds = xr.concat(datasets, dim="time")
    preds_ds.to_netcdf(args.predictions)

    LOG.info("Done!")
