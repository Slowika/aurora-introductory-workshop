"""Utility functions for fine-tuning Microsoft Aurora."""

import dataclasses
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import torch
from aurora import Aurora, Batch

# NOTE: enable imports in local and remote environments
# alternatively, use sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
# and remove the try/except, retaining only the top common import block
try:
    from common.loss import weighted_mae
    from common.utils import create_logger
except ImportError:
    from setup.components.common.loss import weighted_mae
    from setup.components.common.utils import create_logger

LOG = create_logger(__name__)


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
    timestamps = pd.date_range(start=start, end=end, freq=step).to_pydatetime().tolist()
    min_timestamps = 2
    if len(timestamps) < min_timestamps:
        msg = (
            "Less than two timestamps generated, check start datetime is at least "
            f"{step.total_seconds() / 3600} hours before end."
        )
        raise ValueError(msg)
    return timestamps


def get_batches_sample_ts(
        timestamps: list[datetime],
        rng: np.random.Generator,
        batches_per_epoch: int | Literal["all"] = 1,
) -> list[datetime]:
    return (
        timestamps if batches_per_epoch == "all"
        else list(rng.choice(timestamps, batches_per_epoch))
    )

def finetune_short_lead(  # noqa: PLR0913
    model: Aurora,
    params: list[torch.nn.Parameter],
    optimiser: torch.optim.Optimizer,
    batch_fn: Callable[..., Batch],
    timestamps: list[datetime],
    epochs: int = 1,
    batches_per_epoch: int | Literal["all"] = 1,
    *,
    area_weighted: bool = False,
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
    batches_per_epoch : int | Literal["all"], default = 1
        How many batches are randomly sampled in each epoch.
    area_weighted : bool, default = false
        Whether the loss function being used is area-weighted.

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

    pred = None
    for epoch in range(epochs):
        LOG.info("Starting fine-tuning epoch: %d/%d", epoch + 1, epochs)
        chosen_ts = get_batches_sample_ts(use_ts, rng, batches_per_epoch)
        loss_value = 0.0
        for start_datetime in chosen_ts:
            init_batch = batch_fn(start_datetime=start_datetime)
            tgt_batch = batch_fn(start_datetime=init_batch.metadata.time[0] + step, times=1)
            optimiser.zero_grad(set_to_none=True)
            pred = model.forward(init_batch)
            loss_value += weighted_mae(pred, tgt_batch, area_weighted=area_weighted)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        loss_history.append(float(loss_value.detach().cpu().item()))
        _log_epoch_complete(epoch, init_batch, pred, loss_history)

    assert pred is not None, "No predictions generated during fine-tuning."
    return pred, loss_history


def finetune_autoregressive(  # noqa: PLR0913
    model: Aurora,
    params: list[torch.nn.Parameter],
    optimiser: torch.optim.Optimizer,
    batch_fn: Callable[..., Batch],
    timestamps: list[datetime],
    epochs: int = 1,
    batches_per_epoch: int | Literal["all"] = 1,
    rollout_steps: int = 4,
    *,
    area_weighted: bool = False,
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
    area_weighted : bool, default = false
        Whether the loss function being used is area-weighted.

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

    pred = None
    for epoch in range(epochs):
        LOG.info("Starting fine-tuning epoch: %d/%d", epoch + 1, epochs)
        optimiser.zero_grad(set_to_none=True)
        chosen_ts = get_batches_sample_ts(use_ts, rng, batches_per_epoch)
        loss_value = 0.0

        for start_datetime in chosen_ts:
            init_batch = batch_fn(start_datetime=start_datetime)

            with torch.no_grad():
                for istep in range(1, rollout_steps):
                    pred = model.forward(init_batch)
                    init_batch = update_batch(init_batch, pred)
                    LOG.info("Inference step complete: %d/%d", istep, rollout_steps)

            tgt_batch = batch_fn(start_datetime=init_batch.metadata.time[0] + step, times=1)
            pred = model.forward(init_batch)
            LOG.info("Inference step complete: %d/%d", *[rollout_steps] * 2)
            loss_value += weighted_mae(pred, tgt_batch, area_weighted=area_weighted)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        loss_history.append(float(loss_value.detach().cpu().item()))
        _log_epoch_complete(epoch, init_batch, pred, loss_history)

    assert pred is not None, "No predictions generated during fine-tuning."
    return pred, loss_history


def _check_timestamps(timestamps: list[datetime], epochs: int) -> None:
    if epochs > len(timestamps):
        msg = (
            "Insufficient timestamps for epochs, reduce epochs or increase timestamp "
            f"range: usable_timestamps={len(timestamps)}, epochs={epochs}"
        )
        raise ValueError(msg)


def _log_epoch_complete(
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
