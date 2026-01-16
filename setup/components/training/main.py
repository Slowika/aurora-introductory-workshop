"""Aurora fine-tuning script.

This script takes a set of arguments through the command line to perform simple fine-
tuning (updating all weights) of a pretrained Aurora model with local initial state date
OR synthetic test data, a starting datetime, and the number of steps to perform. The
loss history of all fine-tuning steps and the final forecast made with the fine-tuned
model are written to specified output paths.

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

"""

import argparse
import dataclasses
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from functools import partial
from typing import Any

import numpy as np
import torch
from aurora import Aurora, Batch, normalisation

# NOTE: enable imports in local and remote environments
try:
    from common.constants import ATMOS_VAR_MAP, SFC_VAR_MAP, STATIC_VAR_MAP
    from common.utils import (
        BATCH_FNS,
        batch_to_xarray,
        create_logger,
        load_model,
    )
except ImportError:
    from setup.components.common.constants import (
        ATMOS_VAR_MAP,
        SFC_VAR_MAP,
        STATIC_VAR_MAP,
    )
    from setup.components.common.utils import (
        BATCH_FNS,
        batch_to_xarray,
        create_logger,
        load_model,
    )

LOG = create_logger()


def supervised_mae(pred: Batch, target: Batch) -> torch.Tensor:
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


def freeze_to_lora(model: torch.nn.Module) -> None:
    """Freeze all model parameters except LoRA adapters.

    Parameters
    ----------
    model : torch.nn.Module
        Model to freeze.

    """
    for name, p in model.named_parameters():
        name_l = name.lower()
        p.requires_grad = "lora" in name_l


def register_new_variable(
    name: str,
    info: dict[str, Any],
    var_map: dict[str, dict[str, str]],
    cfg: dict[str, Any],
) -> None:
    """Register a new variable to be added to the model and data.

    Parameters
    ----------
    name : str
        Long name of the new variable.
    info : dict[str, Any]
        Information about the new variable. Must include keys:
        - "kind": one of "surf", "static", or "atmos"
        - "key": variable shortname
        - "location": normalisation location statistic
        - "scale": normalisation scale statistic
    var_map : dict[str, dict[str, str]]
        Mapping of variable kinds to Aurora keys.
    cfg : dict[str, Any]
        Aurora model configuration dictionary to update.

    Raises
    ------
    KeyError
        If info["kind"] is not supported, currently "surf", "static", or "atmos".

    """
    try:
        type_var_map = var_map[info["kind"]]
    except KeyError as e:
        msg = f"Unknown variable kind, must be one of {list(var_map.keys())}."
        raise KeyError(msg) from e
    type_var_map[name] = info["key"]
    cfg[f"{info['kind']}_vars"] = tuple(type_var_map.values())
    normalisation.locations[info["key"]] = info["location"]
    normalisation.scales[info["key"]] = info["scale"]


def finetune_short_lead(  # noqa: PLR0913
    model: Aurora,
    params: list[torch.nn.Parameter],
    optimiser: torch.optim.Optimizer,
    batch_fn: Callable[..., Batch],
    start_datetime: datetime,
    steps: int = 1,
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
    start_datetime : datetime.datetime
        Starting datetime for fine-tuning data.
    steps : int, default = 1
        Number of fine-tuning steps to perform.

    Returns
    -------
    pred : aurora.Batch
        Model prediction after fine-tuning step.
    loss_history : list[float]
        List of loss value floats.

    """
    loss_history: list[float] = []

    for step in range(steps):
        init_batch = batch_fn(start_datetime=start_datetime + timedelta(hours=6) * step)
        tgt_batch = batch_fn(
            start_datetime=init_batch.metadata.time[0] + timedelta(hours=6),
            times=1,
        )
        optimiser.zero_grad(set_to_none=True)
        pred = model.forward(init_batch)
        loss_value = supervised_mae(pred, tgt_batch)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        loss_history.append(float(loss_value.detach().cpu().item()))
        _log_step(step, init_batch, pred, loss_history)

    return pred, loss_history


def finetune_autoregressive(  # noqa: PLR0913
    model: Aurora,
    params: list[torch.nn.Parameter],
    optimiser: torch.optim.Optimizer,
    batch_fn: Callable[..., Batch],
    start_datetime: datetime,
    steps: int = 1,
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
    start_datetime : datetime.datetime
        Starting datetime for fine-tuning data.
    steps : int, default = 1
        Number of fine-tuning steps to perform.
    rollout_steps : int, default = 4
        Number of autoregressive rollout steps per fine-tuning step.

    Returns
    -------
    pred : aurora.Batch
        Model prediction after fine-tuning step.
    loss_history : list[float]
        List of loss value floats.

    """
    loss_history: list[float] = []
    init_batch = batch_fn(start_datetime=start_datetime)

    for ft_step in range(steps):
        optimiser.zero_grad(set_to_none=True)
        rollout_batch = init_batch

        with torch.no_grad():
            for ro_step in range(rollout_steps - 1):
                tgt_batch = batch_fn(
                    start_datetime=rollout_batch.metadata.time[0] + timedelta(hours=6),
                    times=1,
                )
                pred = model.forward(rollout_batch)
                LOG.info(
                    "Rollout step: no=%d, init_time=%s, pred_time=%s, loss=%.4f",
                    ro_step,
                    rollout_batch.metadata.time[0].isoformat(timespec="hours"),
                    pred.metadata.time[0].isoformat(timespec="hours"),
                    loss_history[-1],
                )
                rollout_batch = update_batch(rollout_batch, pred)
                # NOTE: not even checking loss here?

        pred = model.forward(rollout_batch)
        loss_value = supervised_mae(pred, tgt_batch)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        loss_history.append(float(loss_value.detach().cpu().item()))
        _log_step(ft_step, init_batch, pred, loss_history)

    return pred, loss_history


def _log_step(
    step: int,
    init_batch: Batch,
    pred: Batch,
    loss_history: list[float],
) -> None:
    LOG.info(
        "Fine-tune step complete: no=%d, init_time=%s, pred_time=%s, loss=%.4f",
        step,
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
        help="JSON string of fine-tuning configuration.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        help="Path to which loss history will be written.",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        help="Path to which the final prediction will be written.",
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
    cfg = args.config["aurora_config"]
    var_map = {
        "surf": SFC_VAR_MAP.copy(),
        "static": STATIC_VAR_MAP.copy(),
        "atmos": ATMOS_VAR_MAP.copy(),
    }
    if (new_vars := args.config.get("extra_variables")) is not None:
        for longname, info in new_vars.items():
            register_new_variable(longname, info, var_map, cfg)
    cfg["strict"] = not (lora := cfg.get("use_lora", False)) and (new_vars is None)
    LOG.info("Using Aurora config: %s", cfg)
    model = load_model(args.model, **cfg)
    LOG.info("Variables to fine-tune: %s", var_map)

    try:
        mode = args.config["mode"]
        batch_fn = partial(
            BATCH_FNS[mode],
            data_path=args.data,
            surf_vars=var_map["surf"],
            static_vars=var_map["static"],
            atmos_vars=var_map["atmos"],
        )
    except KeyError as e:
        msg = (
            "Missing 'mode' field or invalid value, must be one of "
            f"{list(BATCH_FNS.keys())}."
        )
        raise KeyError(msg) from e
    LOG.info("%s data mode enabled.", mode)

    LOG.info("Loading model parameters and optimiser.")
    if lora is True:
        LOG.info("LoRA enabled, freezing all model parameters except LoRA adapters.")
        freeze_to_lora(model)
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        LOG.info("LoRA disabled, using all model parameters for training.")
        params = list(model.parameters())
    optimiser = torch.optim.AdamW(
        params,
        lr=float(args.config.get("learning_rate", 3e-5)),
    )

    ft_steps = args.config.get("steps", 1)
    LOG.info("Starting fine-tuning: start=%s, steps=%d", args.start_datetime, ft_steps)
    # TODO:
    # mlflow logging
    # register fine-tuned model in AML
    # re-work inference to allow new variables
    # test autoregressive and new variable, LoRA
    prediction, loss_history = finetune_fn(
        model=model,
        params=params,
        optimiser=optimiser,
        batch_fn=batch_fn,
        start_datetime=args.start_datetime,
        steps=ft_steps,
        rollout_steps=args.config.get("rollout_steps", 4),
    )
    LOG.info("Completed %d fine-tuning steps.", ft_steps)

    LOG.info("Writing results: loss=%s, prediction=%s", args.loss, args.prediction)
    np.save(args.loss, np.array(loss_history, dtype=float))
    ds = batch_to_xarray(prediction)
    ds.to_netcdf(args.prediction)

    LOG.info("Done!")
