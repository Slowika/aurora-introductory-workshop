"""Aurora fine-tuning script."""

import argparse
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import torch
from aurora import Batch

# NOTE: enable imports in local and remote environments
try:
    from common.utils import (
        batch_to_xarray,
        create_logger,
        load_batch_from_asset,
        load_model,
        make_lowres_batch,
    )
except ImportError:
    from setup.components.common.utils import (
        batch_to_xarray,
        create_logger,
        load_batch_from_asset,
        load_model,
        make_lowres_batch,
    )

LOG = create_logger()


def loss(pred: Batch) -> torch.Tensor:
    """Very simple loss function: sum of squared values over all fields.

    In a real project you would replace this with a physically meaningful
    loss function.

    Parameters
    ----------
    pred : aurora.Batch
        Model prediction.

    Returns
    -------
    total : torch.Tensor
        Scalar loss tensor.

    """
    surf_values = pred.surf_vars.values()
    atmos_values = pred.atmos_vars.values()
    total = torch.tensor(0.0, device=next(iter(surf_values)).device)
    for x in [*surf_values, *atmos_values]:
        total = total + (x * x).sum()
    return total


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
        required=False,
        help="Path to the training data, if not a test run.",
    )
    parser.add_argument(
        "--start_datetime",
        type=datetime.fromisoformat,
        help="Start ISO 8601 format datetime e.g. 2026-01-01T00:00:00.",
    )
    parser.add_argument(
        "--steps",
        type=lambda x: max(int(x), 1),
        default=1,
        help="Number of fine-tuning steps to perform, default and minimum 1.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        help="Path to which loss history will be written.",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        help="Path to which the final 2t prediction will be written.",
    )
    args = parser.parse_args()

    LOG.info("Loading model and optimiser: path=%s", args.model)
    model = load_model(args.model, train=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    LOG.info("Loaded model and optimiser.")

    if args.data is None:
        LOG.info("No data argument provided, using synthetic test data.")
        batch_fn = make_lowres_batch
    else:
        LOG.info("Data argument provided, using real data: path=%s.", args.data)
        batch_fn = partial(
            load_batch_from_asset,
            data_path=args.data,
        )

    LOG.info(
        "Starting fine-tuning: start=%s, steps=%d",
        args.start_datetime,
        args.steps,
    )
    loss_history: list[float] = []
    for step in range(args.steps):
        batch = batch_fn(start_datetime=args.start_datetime + step * timedelta(hours=6))
        optimizer.zero_grad()
        prediction = model.forward(batch)
        loss_value = loss(prediction)
        loss_value.backward()
        optimizer.step()
        loss_history.append(float(loss_value.detach().cpu().item()))
        LOG.info(
            "Fine-tune step complete: no=%d, init_time=%s, pred_time=%s, loss=%.4f",
            step,
            batch.metadata.time[0].isoformat(timespec="hours"),
            prediction.metadata.time[0].isoformat(timespec="hours"),
            loss_history[-1],
        )
    LOG.info("Completed %d fine-tuning steps.", args.steps)

    LOG.info("Writing results: loss=%s, prediction=%s", args.loss, args.prediction)
    # NOTE: original wrote loss to JSON and npy, then all last pred to npz and 2t to npy
    # notebook only used loss npy, final 2t npy
    np.save(args.loss, np.array(loss_history, dtype=float))
    ds = batch_to_xarray(prediction)
    ds.to_netcdf(args.prediction)
    LOG.info("Wrote results: loss=%s, prediction=%s", args.loss, args.prediction)

    LOG.info("Done!")
