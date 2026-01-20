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
    new_vars = args.config.get("extra_variables")
    var_map, var_cfg = register_new_variables(new_vars or {})
    LOG.info("Variables to fine-tune: %s", var_map)
    cfg = args.config["aurora_config"] | var_cfg
    cfg["strict"] = not (lora := cfg.get("use_lora", False)) and (new_vars is None)
    model = load_model(args.model, train=True, **cfg)
    LOG.info("Loaded model using config: %s", cfg)

    batch_fn, ft_steps = validate_common_config(args.config)
    batch_fn = partial(batch_fn, data_path=args.data, **var_map)
    LOG.info("%s data mode enabled.", args.config["mode"])

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

    LOG.info("Starting fine-tuning: start=%s, steps=%d", args.start_datetime, ft_steps)
    # TODO:
    # mlflow logging, add libs to dockerfile
    # register fine-tuned model in AML, write as comp output
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
