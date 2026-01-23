"""Aurora inference script.

This script takes a set of arguments through the command line to perform autoregressive
forecasting using a pretrained Aurora model. All generated forecasts are written to a
NetCDF file at the specified output path. The final prediction is evaluated against
ground-truth data and relevant metrics and figures are logged with MLflow.

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

Key configuration parameters:
- steps: Number of inference steps to perform.
- mode: Whether to use synthetic data ("test") or real data ("era5").
- [optional] extra_variables: Dictionary defining additional variables to include, e.g.:
    {
        <variable_era5_longname>: {
            "kind": <surf_vars or atmos_vars>,
            "key": <variable_era5_shortname>
        }
    }
"""

import argparse
import json
from datetime import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import mlflow
import torch
import xarray as xr
from aurora import rollout

# NOTE: enable imports in local and remote environments
try:
    from common.loss import rmse_loss
    from common.utils import (
        batch_to_xarray,
        create_logger,
        load_model,
        register_new_variables,
        validate_common_config,
    )
except ImportError:
    from setup.components.common.loss import rmse_loss
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
        help="Path to the pre-trained model checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the training data, ignored if configured mode is test.",
    )
    parser.add_argument(
        "--start_datetime",
        type=datetime.fromisoformat,
        help=(
            "Start ISO 8601 format datetime e.g. 2025-01-01T00:00:00. "
            "This datetime and that -6 hours must be present in the data."
        ),
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

    if (inf_steps := args.config.get("steps", 0)) < 1:
        msg = "Absent or invalid 'steps' field, must be at least 1."
        raise ValueError(msg)
    batch_fn = validate_common_config(args.config)
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
    model = model.to("cpu")

    LOG.info("Concatenating and writing predictions: path=%s", args.predictions)
    preds_ds = xr.concat(datasets, dim="time")
    preds_ds.to_netcdf(args.predictions)

    LOG.info("Starting evaluation.")
    eval_datetime = pred.metadata.time[0]
    target = batch_fn(
        data_path=args.data,
        start_datetime=eval_datetime,
        times=1,
        **var_map,
    )
    for longname, shortname in var_map["surf_vars"].items():
        LOG.info("Starting evaluation for variable: %s", longname)
        target_t = target.surf_vars[shortname]
        prediction_t = pred.surf_vars[shortname]

        # compute and log RMSE for the whole planet
        rmse = rmse_loss(prediction_t, target_t).item()
        LOG.info("RMSE: %.4f", rmse)
        mlflow.log_metric(f"{longname} RMSE", rmse)

        # display difference between prediction and ground truth
        diff_t = (prediction_t - target_t).squeeze().cpu().numpy()
        fig = plt.figure(figsize=(40, 50))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        extent = (-180., 180., -90., 90.)
        ax.set_extent(extents=extent)
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        im = ax.imshow(
            diff_t,
            origin="upper",
            extent=extent,
            transform=ccrs.PlateCarree(),
            vmin=diff_t.min(),
            vmax=diff_t.max(),
        )
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        ax.set_title(
            f"Predicted vs. ground-truth - {longname} - {eval_datetime} - "
            f"{inf_steps} steps",
        )
        mlflow.log_figure(fig, f"{longname}_{eval_datetime}_prediction_error_map.png")
        plt.close(fig)
        LOG.info("Finished evaluation for variable: %s", longname)

    LOG.info("Done!")
