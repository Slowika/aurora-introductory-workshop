"""Implementation of loss functions used in model training and evaluation."""

from typing import ClassVar

import torch
from aurora import Batch

try:
    from common.constants import ATMOS_VAR_MAP, SURF_VAR_MAP
except ImportError:
    from setup.components.common.constants import ATMOS_VAR_MAP, SURF_VAR_MAP

SURF_VARS = list(SURF_VAR_MAP.values())
SURF_VARS_COUNT = len(SURF_VARS)
ATMOS_VARS = list(ATMOS_VAR_MAP.values())
ATMOS_VARS_COUNT = len(ATMOS_VARS)


class ERA5FineTuningParams:
    """Fine-tuning parameters for ERA5 data.

    The values are taken from Bodnar et al. (2025), which are themselves based
    on the values given by Bi et al. (2023).
    """

    ERA5_WEIGHT = 2.0  # "gamma" from the paper
    SURF_LOSS_WEIGHT = 0.25  # "alpha" from the paper
    ATMOS_LOSS_WEIGHT = 1.0  # "beta" from the paper
    SURF_VAR_WEIGHTS: ClassVar[dict[str, float]] = {
        "2t": 3.0,  # 2m temperature
        "10u": 0.77,  # 10m u-component of wind
        "10v": 0.66,  # 10m v-component of wind
        "msl": 1.5,  # mean sea level pressure
    }
    ATMOS_VAR_WEIGHTS: ClassVar[dict[str, float]] = {
        "t": 1.7,  # temperature
        "u": 0.87,  # u-component of wind
        "v": 0.6,  # v-component of wind
        "q": 0.78,  # specific humidity
        "z": 2.8,  # geopotential
    }


# Convert the weights in the ERA5FineTuningParams dictionaries to tensors,
# while respecting the orders given in SURF_VARS and ATMOS_VARS.
# These are reshaped into 3d/4d tensors so they can be multiplied when computing
# the loss.
surf_var_weights = torch.tensor(
    [ERA5FineTuningParams.SURF_VAR_WEIGHTS[vid] for vid in SURF_VARS],
).view(-1, 1, 1)
atmos_var_weights = torch.tensor(
    [ERA5FineTuningParams.ATMOS_VAR_WEIGHTS[vid] for vid in ATMOS_VARS],
).view(-1, 1, 1, 1)


def surf_tensor(target: Batch) -> torch.Tensor:
    """Extract a surface variables tensor from a target Batch.

    Parameters
    ----------
    target : aurora.Batch
        Batch containing target surface variables.

    Returns
    -------
    torch.Tensor
        Stacked surface variables tensor.

    """
    return torch.stack([target.surf_vars[vid] for vid in SURF_VARS])


def atmos_tensor(batch: Batch) -> torch.Tensor:
    """Extract an atmospheric variables tensor from a Batch.

    Parameters
    ----------
    batch : aurora.Batch
        Batch containing atmospheric variables.

    Returns
    -------
    torch.Tensor
        Stacked atmospheric variables tensor.

    """
    return torch.stack([batch.atmos_vars[vid] for vid in ATMOS_VARS])


def weighted_mae_loss(pred: Batch, target: Batch) -> torch.Tensor:
    """Area-weighted mean absolute error, following Bodnar et al. (2025).

    Note: this does not currently account for new variables beyond those present in the
    pre-trained model, nor does it account for changes to variable names necessary to
    write data without naming collisions (e.g. z -> z_atmos).

    Parameters
    ----------
    pred : aurora.Batch
        Model prediction.
    target : aurora.Batch
        Ground truth.

    Returns
    -------
    total : torch.Tensor
        Scalar loss tensor.

    """
    surf_preds = surf_tensor(pred)
    surf_targets = surf_tensor(target)
    atmos_preds = atmos_tensor(pred)
    atmos_targets = atmos_tensor(target)
    surf_abs_err = (surf_preds - surf_targets).abs()
    atmos_abs_err = (atmos_preds - atmos_targets).abs()
    device = surf_preds.device
    surf_loss = (surf_var_weights.to(device) * surf_abs_err).sum()
    atmos_loss = (atmos_var_weights.to(device) * atmos_abs_err).sum()
    # normalisation denominator H x W
    surf_size = surf_loss.numel() / SURF_VARS_COUNT
    # normalisation denominator C x H x W
    atmos_size = atmos_loss.numel() / ATMOS_VARS_COUNT
    # total loss
    return (
        (
            ERA5FineTuningParams.SURF_LOSS_WEIGHT * surf_loss / surf_size
            + ERA5FineTuningParams.ATMOS_LOSS_WEIGHT * atmos_loss / atmos_size
        )
        * ERA5FineTuningParams.ERA5_WEIGHT
        / (SURF_VARS_COUNT + ATMOS_VARS_COUNT)
    )


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSE computed between two tensors, summed equally across all variables.

    Parameters
    ----------
    pred : torch.Tensor
        Single quantity (e.g. surface temperature) drawn from model output.
    target : torch.Tensor
        Single quantity (e.g. surface temperature), ground truth.

    Returns
    -------
    total : torch.Tensor
        Scalar loss tensor.

    """
    return torch.sqrt(torch.mean((pred - target) ** 2))
