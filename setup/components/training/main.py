"""Aurora fine-tuning script.

This script takes a set of arguments through the command line to perform simple fine-
tuning (updating all weights) of a pretrained Aurora model. The loss history of all
fine-tuning epochs and the final forecast made with the fine-tuned model are written to
specified output paths.

Running locally:
    python -m setup.components.training.main \
        --model <path to local model checkpoint e.g. ./aurora-0.25-pretrained.ckpt> \
        --data <path to local initial state data e.g. ./era5_subset.zarr, optional> \
        --start_datetime <ISO 8601 format datetime e.g. 2026-01-01T00:00:00> \
        --config <JSON-formatted string of fine-tuning configuration> \
        --loss <path to output loss history NumPy file e.g. ./losses.npy> \
        --prediction <path to output NetCDF of the final prediction e.g. ./fcst.nc>

Running in Azure Machine Learning:
    See setup/components/training/component.py for definition and deployment, and
    notebooks/0_aurora_workshop.ipynb for example usage.

Key configuration parameters:
- type: Whether to perform short lead ("short") or autoregressive ("rollout")
    fine-tuning.
- mode: Whether to use synthetic data ("test") or real data ("era5").
- epochs: Number of fine-tuning epochs.
- learning_rate: Learning rate for the optimiser.
- [optional] rollout_steps: Number of autoregressive rollout steps per fine-tuning
    step, only used if type is "rollout".
- [optional] aurora_config: Dictionary of Aurora model configuration parameters to
    override the default model configuration, e.g.:
    {"aurora_config": {"use_lora": true, "lora_steps": 40}}.
    See aurora.Aurora documentation for all valid keyword arguments.
- [optional] extra_variables: Dictionary defining additional variables to include, e.g.:
    {
        <variable_era5_longname>: {
            "kind": <surf_vars or atmos_vars>,
            "key": <variable_era5_shortname>,
            "location": <float>,
            "scale": <float>
        }
    }
"""

import argparse
import dataclasses
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import torch

# NOTE: enable imports in local and remote environments
try:
    from common.utils import (
        batch_to_xarray,
        create_logger,
        load_model,
        register_new_variables,
        validate_common_config,
    )
    from common.loss import weighted_mae_loss
except ImportError:
    from setup.components.common.utils import (
        batch_to_xarray,
        create_logger,
        load_model,
        register_new_variables,
        validate_common_config,
    )
    from setup.components.common.loss import weighted_mae_loss

LOG = create_logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
            "Start ISO 8601 format datetime e.g. 2026-01-01T00:00:00. "
            "This datetime and that -6 hours must be present in the data."
        ),
    )
    parser.add_argument(
        "--end_datetime",
        type=datetime.fromisoformat,
        help=(
            "End ISO 8601 format datetime e.g. 2026-01-01T00:00:00. "
            "This datetime is only possibly used as a target."
        ),
    )
    parser.add_argument(
        "--config",
        type=json.loads,
        help="JSON string of fine-tuning configuration.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        help="Path to which the loss history NumPy array will be written.",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        help="Path to which the final prediction NetCDF file will be written.",
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        help="Path to which the fine-tuned model state .ckpt file will be written.",
    )
    args = parser.parse_args()

    try:
        ft_type = args.config["type"]
        finetune_fn = FINETUNE_FNS[ft_type]
    except KeyError as e:
        msg = (
            "Missing 'type' field or invalid value, must be one of "
            f"{list(FINETUNE_FNS.keys())}."
        )
        raise KeyError(msg) from e
    LOG.info("%s training enabled.", ft_type)

    LOG.info("Loading model: path=%s", args.model)
    new_vars = args.config.get("extra_variables")
    var_map, var_cfg = register_new_variables(new_vars or {})
    LOG.info("Variables to fine-tune: %s", var_map)
    cfg = args.config["aurora_config"] | var_cfg
    cfg["strict"] = not (lora := cfg.get("use_lora", False)) and (new_vars is None)
    model = load_model(args.model, train=True, **cfg)
    LOG.info("Loaded model using config: %s", cfg)

    if (epochs := args.config.get("epochs", 0)) < 1:
        msg = "Absent or invalid 'epochs' field, must be at least 1."
        raise ValueError(msg)

    batch_fn = validate_common_config(args.config)
    batch_fn = partial(batch_fn, data_path=args.data, **var_map)
    LOG.info("%s data mode enabled.", args.config["mode"])

    LOG.info("Loading model parameters and optimiser: lora=%s", lora)
    params = get_lora_params(model) if lora else list(model.parameters())
    optimiser = torch.optim.AdamW(
        params,
        lr=float(args.config.get("learning_rate", 3e-5)),
    )

    LOG.info(
        "Starting fine-tuning: start=%s, stop=%s, epochs=%d",
        args.start_datetime,
        args.end_datetime,
        epochs,
    )
    loss_history: list[float] = []
    for step in range(args.steps):
        # input for the model is the ERA5 data at start_datetime
        inputs = batch_fn(start_datetime=args.start_datetime)
        # target is the ERA5 data 6h in the future
        target = batch_fn(start_datetime=args.start_datetime + timedelta(hours=6))
        optimizer.zero_grad()
        prediction = model.forward(inputs)
        loss_value = weighted_mae_loss(prediction, target)
        loss_value.backward()
        optimizer.step()
        loss_history.append(float(loss_value.detach().cpu().item()))
        LOG.info(
            "Fine-tune step complete: no=%d, init_time=%s, pred_time=%s, loss=%.4f",
            step,
            inputs.metadata.time[0].isoformat(timespec="hours"),
            prediction.metadata.time[0].isoformat(timespec="hours"),
            loss_history[-1],
        )
    LOG.info("Completed %d fine-tuning steps.", args.steps)

    LOG.info("Writing results: loss=%s, prediction=%s", args.loss, args.prediction)
    np.save(args.loss, np.array(loss_history, dtype=float))
    ds = batch_to_xarray(prediction)
    ds.to_netcdf(args.prediction)

    LOG.info("Writing model: path=%s", args.finetuned)
    state = model.state_dict()
    if lora:
        LOG.info("Saving LoRA weights only.")
        state = {k: v for k, v in state.items() if "lora" in k.lower()}
    torch.save(state, args.finetuned)

    LOG.info("Done!")
