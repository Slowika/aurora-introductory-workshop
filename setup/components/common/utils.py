"""Common utility functions for inference and training."""
import logging
import sys
from datetime import datetime, timedelta, timezone  # UTC alias added py 3.11

import torch
import xarray as xr
from aurora import AuroraPretrained, Batch, Metadata


def load_model(model_path: str) -> AuroraPretrained:
    """Load the Aurora model from a local checkpoint.

    Parameters
    ----------
    model_path : str
        Path to the local model checkpoint.

    Returns
    -------
    aurora.AuroraPretrained
        Loaded Aurora model.

    """
    model = AuroraPretrained()
    model.load_checkpoint_local(model_path)
    model = model.to("cuda")
    model.eval()
    return model


def make_lowres_batch() -> Batch:
    """Create a small 17x32 batch on the given device."""
    return Batch(
        surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 17),
            lon=torch.linspace(0, 360, 32 + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, tzinfo=timezone.utc),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )


def load_batch_from_asset(data_path: str, start_datetime: datetime) -> Batch:
    """Load a Batch from a local Azure ML data asset.

    Parameters
    ----------
    data_path : str
        Path to the data asset, locally mounted or downloaded.
    start_datetime : datetime.datetime
        Start datetime for the batch.

    Returns
    -------
    aurora.Batch
        Loaded Batch object.

    """
    ds = xr.open_dataset(data_path, chunks={})
    prev_datetime = start_datetime - timedelta(hours=6)
    ds_sel = ds.sel(time=[prev_datetime, start_datetime])
    return Batch(
        surf_vars={},
        static_vars={},
        atmos_vars={},
        metadata=Metadata(
            lat="",
            lon="",
            atmos_levels=[],
        ),
    )


def create_logger() -> logging.Logger:
    """Create a configured logger.

    Returns
    -------
    logging.Logger
        Configured logger.

    """
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)
