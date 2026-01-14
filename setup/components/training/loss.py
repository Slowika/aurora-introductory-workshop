import torch

from aurora import Batch

SURF_VARS = ["2t", "10u", "10v", "msl"]
SURF_VARS_COUNT = 4
ATMOS_VARS = ["t", "u", "v", "q", "z"]
ATMOS_VARS_COUNT = 5
# Working around a quirk of the output prediction: keys in the output dictionary
# are slightly different.
SURF_VAR_RENAME = {"2t": "t2m", "10u": "u10", "10v" : "v10", "msl" : "msl"}

class ERA5FineTuningParams:
    """"Fine-tuning parameters for ERA5 data.
    The values are taken from Bodnar et al. (2025), which are themselves based
    on the values given by Bi et al. (2023)"""
    ERA5_WEIGHT = 2.0        # "gamma" from the paper
    SURF_LOSS_WEIGHT = 0.25  # "alpha" from the paper
    ATMOS_LOSS_WEIGHT = 1.0  # "beta" from the paper
    SURF_VAR_WEIGHTS = {
        "2t": 3.0,   # 2m temperature
        "10u": 0.77, # 10m u-component of wind
        "10v": 0.66, # 10m v-component of wind
        "msl": 1.5   # mean sea level pressure
    }
    ATMOS_VAR_WEIGHTS = {
        "t": 1.7,    # temperature
        "u": 0.87,   # u-component of wind
        "v": 0.6,    # v-component of wind
        "q": 0.78,   # specific humidity
        "z": 2.8     # geopotential
    }

# Convert the weights in the ERA5FineTuningParams dictionaries to tensors,
# while respecting the orders given in SURF_VARS and ATMOS_VARS.
# These are reshaped into 3d/4d tensors so they can be multiplied when computing
# the loss.
surf_var_weights  = (torch.tensor([ERA5FineTuningParams.SURF_VAR_WEIGHTS[id]
                                   for id in SURF_VARS])
                     .view(-1, 1, 1))
atmos_var_weights = (torch.tensor([ERA5FineTuningParams.ATMOS_VAR_WEIGHTS[id]
                                   for id in ATMOS_VARS])
                     .view(-1, 1, 1, 1))

def surf_target_tensor(target: Batch):
    """Extract a surface variables tensor from a target Batch."""
    surf_target_dict = target.surf_vars
    return torch.stack([surf_target_dict[id] for id in SURF_VARS])

def surf_pred_tensor(pred: Batch):
    """Extract a surface variables tensor from a prediction Batch.
    Note the name mangling."""
    surf_pred_dict = pred.surf_vars
    return torch.stack([surf_pred_dict[SURF_VAR_RENAME[id]]
                        for id in SURF_VARS])

def atmos_tensor(batch: Batch):
    """Extract an atmospheric variables tensor from a Batch."""
    atmos_dict = batch.atmos_vars
    return torch.stack([atmos_dict[id] for id in ATMOS_VARS])

def loss(pred: Batch, target: Batch) -> torch.Tensor:
    """Area-weighted mean absolute error, following Bodnar et al. (2025)

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
    surf_preds    = surf_pred_tensor(pred)
    surf_targets  = surf_target_tensor(target)
    atmos_preds   = atmos_tensor(pred)
    atmos_targets = atmos_tensor(target)
    surf_abs_err  = (surf_preds - surf_targets).abs()
    atmos_abs_err = (atmos_preds - atmos_targets).abs()
    device = surf_preds.device
    surf_loss  = (surf_var_weights.to(device) * surf_abs_err).sum()
    atmos_loss = (atmos_var_weights.to(device) * atmos_abs_err).sum()
    # normalisation denominator H x W
    surf_size  = surf_loss.numel() / SURF_VARS_COUNT
    # normalisation denominator C x H x W
    atmos_size = atmos_loss.numel() / ATMOS_VARS_COUNT
    # total loss
    total = ((ERA5FineTuningParams.SURF_LOSS_WEIGHT * surf_loss / surf_size +
              ERA5FineTuningParams.ATMOS_LOSS_WEIGHT * atmos_loss / atmos_size)
             * ERA5FineTuningParams.ERA5_WEIGHT
             / (SURF_VARS_COUNT + ATMOS_VARS_COUNT))
    return total