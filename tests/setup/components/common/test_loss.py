"""Test loss functions and tensor extraction utilities."""

import torch
from aurora import Batch

from setup.components.common.loss import (
    ATMOS_VARS,
    ATMOS_VARS_COUNT,
    SURF_VARS,
    SURF_VARS_COUNT,
    ERA5FineTuningParams,
    atmos_tensor,
    atmos_var_weights,
    rmse_loss,
    surf_tensor,
    surf_var_weights,
    weighted_mae,
)


def test_surf_var_weights_shape() -> None:
    """Verify surf weight tensor broadcasts over (var, lat, lon)."""
    assert surf_var_weights.shape == (SURF_VARS_COUNT, 1, 1)


def test_atmos_var_weights_shape() -> None:
    """Verify atmos weight tensor broadcasts over (var, level, lat, lon)."""
    assert atmos_var_weights.shape == (ATMOS_VARS_COUNT, 1, 1, 1)


def test_surf_var_weights_values() -> None:
    """Ensure weight tensor order matches SURF_VARS ordering."""
    expected = [ERA5FineTuningParams.SURF_VAR_WEIGHTS[v] for v in SURF_VARS]
    torch.testing.assert_close(surf_var_weights.flatten(), torch.tensor(expected))


def test_atmos_var_weights_values() -> None:
    """Ensure weight tensor order matches ATMOS_VARS ordering."""
    expected = [ERA5FineTuningParams.ATMOS_VAR_WEIGHTS[v] for v in ATMOS_VARS]
    torch.testing.assert_close(atmos_var_weights.flatten(), torch.tensor(expected))


def test_surf_tensor_shape(sample_batch: Batch) -> None:
    """Stack surface variables into (n_vars, batch, time, lat, lon)."""
    result = surf_tensor(sample_batch)
    assert result.shape == (SURF_VARS_COUNT, 1, 1, 4, 8)


def test_surf_tensor_order(sample_batch: Batch) -> None:
    """Preserve SURF_VARS ordering when stacking."""
    result = surf_tensor(sample_batch)
    for i, vid in enumerate(SURF_VARS):
        torch.testing.assert_close(result[i], sample_batch.surf_vars[vid])


def test_atmos_tensor_shape(sample_batch: Batch) -> None:
    """Stack atmospheric variables into (n_vars, batch, time, levels, lat, lon)."""
    result = atmos_tensor(sample_batch)
    assert result.shape == (ATMOS_VARS_COUNT, 1, 1, 3, 4, 8)


def test_atmos_tensor_order(sample_batch: Batch) -> None:
    """Preserve ATMOS_VARS ordering when stacking."""
    result = atmos_tensor(sample_batch)
    for i, vid in enumerate(ATMOS_VARS):
        torch.testing.assert_close(result[i], sample_batch.atmos_vars[vid])


def test_weighted_mae_zero_error(sample_batch: Batch) -> None:
    """Return zero loss when prediction equals target."""
    loss = weighted_mae(sample_batch, sample_batch, area_weighted=False)
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_weighted_mae_positive(sample_batch: Batch, target_batch: Batch) -> None:
    """Return positive loss for differing pred and target."""
    loss = weighted_mae(sample_batch, target_batch, area_weighted=False)
    assert loss.item() > 0


def test_weighted_mae_scalar(sample_batch: Batch, target_batch: Batch) -> None:
    """Produce a scalar (0-dim) loss tensor."""
    loss = weighted_mae(sample_batch, target_batch, area_weighted=True)
    assert loss.dim() == 0


def test_weighted_mae_area_weighted_differs(
    sample_batch: Batch,
    target_batch: Batch,
) -> None:
    """Produce different results with and without area weighting."""
    loss_flat = weighted_mae(sample_batch, target_batch, area_weighted=False)
    loss_aw = weighted_mae(sample_batch, target_batch, area_weighted=True)
    assert not torch.allclose(loss_flat, loss_aw)


def test_rmse_loss_zero_error() -> None:
    """Return zero RMSE for identical tensors."""
    t = torch.randn(4, 8)
    torch.testing.assert_close(rmse_loss(t, t), torch.tensor(0.0))


def test_rmse_loss_known_value() -> None:
    """Compute correct RMSE for a hand-calculated example."""
    pred = torch.tensor([3.0, 4.0])
    target = torch.tensor([0.0, 0.0])
    expected = torch.sqrt(torch.tensor((9.0 + 16.0) / 2))
    torch.testing.assert_close(rmse_loss(pred, target), expected)
