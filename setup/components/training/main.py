"""Aurora fine-tuning script.

This script takes a set of arguments through the command line to perform fine-tuning of
a pretrained Aurora model. The loss history of all fine-tuning epochs and the final
forecast made with the fine-tuned model are written to specified output paths.

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
from aurora import Aurora, Batch

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


def mae(pred: Batch, target: Batch) -> torch.Tensor:
    """Calculate MAE over all dynamic variables in the prediction.

    Parameters
    ----------
    pred : aurora.Batch
        Model prediction.
    target : aurora.Batch
        Target batch.

    Returns
    -------
    total : torch.Tensor
        Scalar MAE loss tensor.

    """
    device = next(iter(pred.surf_vars.values())).device
    total = torch.tensor(0.0, device=device)
    for k in pred.surf_vars:
        total = total + (pred.surf_vars[k] - target.surf_vars[k]).abs().mean()
    for k in pred.atmos_vars:
        total = total + (pred.atmos_vars[k] - target.atmos_vars[k]).abs().mean()
    return total


def get_lora_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Freeze all model parameters except LoRA adapters, returning the latter.

    Parameters
    ----------
    model : torch.nn.Module
        Model to freeze.

    Returns
    -------
    params : list[torch.nn.Parameter]
        List of trainable LoRA adapter parameters.

    Raises
    ------
    RuntimeError
        If no trainable parameters are found after freezing.

    """
    params = []
    for name, p in model.named_parameters():
        name_l = name.lower()
        p.requires_grad = "lora" in name_l
        if p.requires_grad:
            params.append(p)

    if not params:
        msg = "No trainable parameters found after LoRA-freeze."
        raise RuntimeError(msg)
    return params


def get_datetime_range(
    start: datetime,
    end: datetime,
    step: timedelta = timedelta(hours=6),
) -> list[datetime]:
    """Generate a list of datetimes at regular intervals between start and end.

    Parameters
    ----------
    start : datetime.datetime
        Range start datetime.
    end : datetime.datetime
        Range end datetime.
    step : datetime.timedelta, default = timedelta(hours=6)
        Time interval between generated datetimes.

    Returns
    -------
    list[datetime.datetime]
        List of generated datetimes.

    Raises
    ------
    ValueError
        If less than two timestamps are generated, i.e. start is not at least one step
        before end.

    """
    timestamps = [
        start + step * i for i
        in range(int((end - start).total_seconds() // step.total_seconds()) + 1)
    ]
    min_ts = 2
    if len(timestamps) < min_ts:
        msg = (
            "Less than two timestamps generated, check start datetime is at least "
            f"{step.total_seconds() / 3600} hours before end."
        )
        raise ValueError(msg)
    return timestamps


def finetune_short_lead(  # noqa: PLR0913
    model: Aurora,
    params: list[torch.nn.Parameter],
    optimiser: torch.optim.Optimizer,
    batch_fn: Callable[..., Batch],
    timestamps: list[datetime],
    epochs: int = 1,
    **_: dict,
) -> tuple[Batch, list[float]]:
    """Fine-tune a pre-trained Aurora model with short lead training.

    Parameters
    ----------
    model : aurora.Aurora
        Aurora model to fine-tune.
    params : list[torch.nn.Parameter]
        Model parameters to fine-tune.
    optimiser : torch.optim.Optimizer
        Optimiser for fine-tuning.
    batch_fn : collections.abc.Callable[..., aurora.Batch]
        Callable returning a batch for fine-tuning.
    timestamps : list[datetime.datetime]
        List of datetimes for fine-tuning data.
    epochs : int, default = 1
        Number of fine-tuning epochs.

    Returns
    -------
    pred : aurora.Batch
        Model prediction after the final epoch.
    loss_history : list[float]
        List of loss value floats.

    """
    use_ts = timestamps[:len(timestamps) - 1]
    _check_timestamps(use_ts, epochs)
    loss_history: list[float] = []
    step = timestamps[1] - timestamps[0]
    rng = np.random.Generator(np.random.PCG64())

    for epoch in range(epochs):
        start_datetime = use_ts.pop(rng.integers(0, len(use_ts)))
        init_batch = batch_fn(start_datetime=start_datetime)
        tgt_batch = batch_fn(start_datetime=init_batch.metadata.time[0] + step, times=1)
        optimiser.zero_grad(set_to_none=True)
        pred = model.forward(init_batch)
        loss_value = mae(pred, tgt_batch)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        loss_history.append(float(loss_value.detach().cpu().item()))
        _log_epoch(epoch, init_batch, pred, loss_history)

    return pred, loss_history


def finetune_autoregressive(  # noqa: PLR0913
    model: Aurora,
    params: list[torch.nn.Parameter],
    optimiser: torch.optim.Optimizer,
    batch_fn: Callable[..., Batch],
    timestamps: list[datetime],
    epochs: int = 1,
    rollout_steps: int = 4,
) -> tuple[Batch, list[float]]:
    """Fine-tune an Aurora model with autoregressive training.

    Loss calculation is performed on the final prediction to preserve memory relative to
    cumulative loss.

    Parameters
    ----------
    model : aurora.Aurora
        Aurora model to fine-tune.
    params : list[torch.nn.Parameter]
        Model parameters to fine-tune.
    optimiser : torch.optim.Optimizer
        Optimiser for fine-tuning.
    batch_fn : collections.abc.Callable[..., aurora.Batch]
        Callable returning a batch for fine-tuning.
    timestamps : list[datetime]
        List of datetimes for fine-tuning data.
    epochs : int, default = 1
        Number of fine-tuning epochs.
    rollout_steps : int, default = 4
        Number of autoregressive rollout steps per epoch.

    Returns
    -------
    pred : aurora.Batch
        Model prediction after the final epoch.
    loss_history : list[float]
        List of loss value floats.

    """
    use_ts = timestamps[:len(timestamps) - rollout_steps]
    _check_timestamps(use_ts, epochs)
    loss_history: list[float] = []
    step = timestamps[1] - timestamps[0]
    rng = np.random.Generator(np.random.PCG64())

    for epoch in range(epochs):
        optimiser.zero_grad(set_to_none=True)
        start_datetime = use_ts.pop(rng.integers(0, len(use_ts)))
        init_batch = batch_fn(start_datetime=start_datetime)

        with torch.no_grad():
            for ro_step in range(rollout_steps - 1):
                pred = model.forward(init_batch)
                LOG.info(
                    "Rollout step complete: no=%d, init_time=%s, pred_time=%s",
                    ro_step,
                    init_batch.metadata.time[0].isoformat(timespec="hours"),
                    pred.metadata.time[0].isoformat(timespec="hours"),
                )
                init_batch = update_batch(init_batch, pred)

        tgt_batch = batch_fn(
            start_datetime=init_batch.metadata.time[0] + step,
            times=1,
        )
        pred = model.forward(init_batch)
        loss_value = mae(pred, tgt_batch)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        loss_history.append(float(loss_value.detach().cpu().item()))
        _log_epoch(epoch, init_batch, pred, loss_history)

    return pred, loss_history


def _check_timestamps(timestamps: list[datetime], epochs: int) -> None:
    if epochs > len(timestamps):
        msg = (
            "Insufficient timestamps for epochs, reduce epochs or increase timestamp "
            f"range: usable_timestamps={len(timestamps)}, epochs={epochs}"
        )
        raise ValueError(msg)


def _log_epoch(
    epoch: int,
    init_batch: Batch,
    pred: Batch,
    loss_history: list[float],
) -> None:
    LOG.info(
        "Fine-tune epoch complete: no=%d, init_time=%s, pred_time=%s, loss=%.4f",
        epoch,
        init_batch.metadata.time[0].isoformat(timespec="hours"),
        pred.metadata.time[0].isoformat(timespec="hours"),
        loss_history[-1],
    )


def update_batch(init_batch: Batch, pred: Batch) -> Batch:
    """Update a batch with the latest prediction for autoregressive fine-tuning.

    Parameters
    ----------
    init_batch : aurora.Batch
        Input batch.
    pred : aurora.Batch
        Latest model prediction.

    Returns
    -------
    aurora.Batch
        Updated input batch for the next rollout step.

    """
    return dataclasses.replace(
        pred,
        surf_vars={
            k: torch.cat([init_batch.surf_vars[k][:, 1:], v], dim=1)
            for k, v in pred.surf_vars.items()
        },
        atmos_vars={
            k: torch.cat([init_batch.atmos_vars[k][:, 1:], v], dim=1)
            for k, v in pred.atmos_vars.items()
        },
    )


# mapping of fine-tuning modes to functions
FINETUNE_FNS: dict[str, Callable[..., tuple[Batch, list[float]]]] = {
    "short": finetune_short_lead,
    "rollout": finetune_autoregressive,
}

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
            "Start ISO 8601 format datetime e.g. 2025-01-01T00:00:00. "
            "This datetime and that -6 hours must be present in the data."
        ),
    )
    parser.add_argument(
        "--end_datetime",
        type=datetime.fromisoformat,
        help=(
            "End ISO 8601 format datetime e.g. 2025-01-31T23:00:00. "
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
    model = load_model(
        args.model,
        train=True,
        strict=not (lora := cfg.get("use_lora", False)) and (new_vars is None),
        **cfg,
    )
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
    timestamps = get_datetime_range(args.start_datetime, args.end_datetime)
    prediction, loss_history = finetune_fn(
        model=model,
        params=params,
        optimiser=optimiser,
        batch_fn=batch_fn,
        timestamps=timestamps,
        epochs=epochs,
        rollout_steps=args.config.get("rollout_steps", 4),
    )
    model = model.to("cpu")
    LOG.info("Completed %d fine-tuning epochs.", epochs)
    LOG.info("Writing results: loss=%s, prediction=%s", args.loss, args.prediction)
    np.save(args.loss, np.array(loss_history, dtype=float))
    ds = batch_to_xarray(prediction)
    ds.to_netcdf(args.prediction)

    LOG.info("Writing model: path=%s", args.finetuned)
    model_state = {
        k: (
            v.to(dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
            else v
        )
        for k, v in model.state_dict().items()
    }
    if lora:
        LOG.info("Updating model with LoRA: path=%s", args.finetuned)
        base_state = torch.load(args.model, map_location="cpu", weights_only=True)
        base_state.update(model_state)
        torch.save(base_state, args.finetuned)
    else:
        torch.save(model_state, args.finetuned)

    LOG.info("Done!")
