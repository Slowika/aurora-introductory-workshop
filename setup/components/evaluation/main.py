"""Aurora evaluation script.

This script takes a pair of prediction and ground truth meteorological data and produces
outputs that 

Running locally:
    python -m setup.components.evaluation.main \
        --target <path to local predicted state data e.g. ./era5_subset.zarr, optional> \
        --predictions <path to local ground truth state data e.g. ./era5_subset.zarr, optional> \
        --start_datetime <ISO 8601 format datetime e.g. 2026-01-01T00:00:00> \
        --steps <number of inference steps performed e.g. 10>

Running in Azure Machine Learning:
    See setup/components/inference/component.py for definition and deployment, and
    notebooks/0_aurora_workshop.ipynb for example usage.
"""

import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import mlflow
import xarray as xr


# NOTE: enable imports in local and remote environments
try:
    from common.utils import (
        create_logger,
        load_batch_from_asset,
    )
    from common.loss import rmse_loss
    from common.constants import SFC_VAR_MAP, SURF_VAR_RENAME
except ImportError:
    from setup.components.common.utils import (
        create_logger,
        load_batch_from_asset,
    )
    from setup.components.common.loss import rmse_loss
    from setup.components.common.constants import SFC_VAR_MAP, SURF_VAR_RENAME

LOG = create_logger()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aurora Inference Component")
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the ground truth data.",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        help="Path to the predicted data.",
    )
    parser.add_argument(
        "--start_datetime",
        type=datetime.fromisoformat,
        help="Start ISO 8601 format datetime e.g. 2026-01-01T00:00:00.",
    )
    parser.add_argument(
        "--steps",
        type=lambda x: max(int(x), 1),
        help="Number of inference steps that were performed, default and minimum 1.",
    )
    args = parser.parse_args()

    with mlflow.start_run():
        target = load_batch_from_asset(
            args.data,
            args.start_datetime + args.steps * timedelta(hours=6)
        )
        prediction = load_batch_from_asset(
            args.prediction,
            args.start_datetime + args.steps * timedelta(hours=6)
        )

        # iterate over all surface variables
        for (desc, key_str) in SFC_VAR_MAP.items():
            target_t = target.surf_vars[key_str]
            prediction_t = prediction.surf_vars[SURF_VAR_RENAME[key_str]]

            # compute and log RMSE for the whole planet
            rmse = rmse_loss(prediction_t, target_t).item()
            LOG.info("RMSE for prediction: %.4f", rmse)
            mlflow.log_metric(f"{desc} (RMSE)", rmse)

            # display difference between prediction and target surface temperature
            diff_t = prediction_t - target_t
            fig = plt.figure(figsize=(40, 50))
            ax = plt.axes(projection=ccrs.PlateCarree())
            extent = [-180, 180, -90, 90]
            ax.set_extent(extent=extent)
            ax.coastlines()
            ax.gridlines(draw_labels=True)
            ax.imshow(diff_t, origin='lower', extent=extent, transform=ccrs.PlateCarree())
            ax.set_title(f"Predicted vs ground-truth {desc} "
                         f"({args.start_datetime}; {args.steps} steps)")
            mlflow.log_figure(fig, "temp_prediction_error_map.png")

    LOG.info("Done!")
