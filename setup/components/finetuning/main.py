"""Aurora fine-tuning script.

This script takes a set of arguments through the command line to perform fine-tuning of
a pretrained Aurora model. The loss history of all fine-tuning epochs, the final
forecast made, and the fine-tuned model checkpoint are written to the specified output
paths.

Running locally:
    See notebooks/0_aurora_workshop_local.ipynb for example usage or run:
    `python -m setup.components.finetuning.main -h`

Running in Azure Machine Learning:
    See setup/components/finetuning/component.yaml for definition and
    notebooks/0_aurora_workshop.ipynb for example usage. Deploy with
    `az ml component create -f setup/components/finetuning/component.yaml -w <workspace>
    -g <resource_group>`

For configuration, see examples in notebooks/finetune_configs.yaml and the Pydantic
model definition in `setup.components.common.models.FinetuneConfig`.
"""
import argparse
from collections.abc import Callable
from functools import partial

import numpy as np
import torch
from aurora import Batch

# NOTE: enable imports in local and remote environments
# alternatively, use sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
# and remove the try/except, retaining only the top common import block
try:
    from common.models import FinetuneConfig
    from common.utils import (
        batch_to_xarray,
        create_logger,
        load_model,
        tz_naive_datetime,
    )
    from utils import (
        finetune_autoregressive,
        finetune_short_lead,
        get_datetime_range,
        get_lora_params,
    )
except ImportError:
    from setup.components.common.models import FinetuneConfig
    from setup.components.common.utils import (
        batch_to_xarray,
        create_logger,
        load_model,
        tz_naive_datetime,
    )
    from setup.components.finetuning.utils import (
        finetune_autoregressive,
        finetune_short_lead,
        get_datetime_range,
        get_lora_params,
    )

LOG = create_logger(__name__)
# mapping of fine-tuning modes to functions
FINETUNE_FNS: dict[str, Callable[..., tuple[Batch, list[float]]]] = {
    "short": finetune_short_lead,
    "rollout": finetune_autoregressive,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Aurora Fine-tuning")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "Path to the pre-trained Microsoft Aurora model checkpoint to use in "
            "inference."
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the data to fine-tune on. Ignored if configured mode is test.",
    )
    parser.add_argument(
        "--start_datetime",
        type=tz_naive_datetime,
        help=(
            "ISO 8601 format start datetime e.g. 2025-01-01T00:00:00. "
            "This datetime and that -6 hours must be present in the data."
        ),
    )
    parser.add_argument(
        "--end_datetime",
        type=tz_naive_datetime,
        help=(
            "ISO 8601 format end datetime e.g. 2025-01-31T23:00:00. "
            "This datetime is only possibly used as a target."
        ),
    )
    parser.add_argument(
        "--config",
        type=FinetuneConfig.model_validate_json,
        help="JSON-serialised string of fine-tuning configuration.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        help="Path to which a NumPy array of loss history will be written.",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        help="Path to which a NetCDF file of the final prediction will be written.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help=(
            "Path to which a checkpoint of the fine-tuned model state will be written."
        ),
    )
    args = parser.parse_args()
    cfg: FinetuneConfig = args.config
    LOG.info("Starting fine-tuning run with config: %s", cfg.model_dump())

    LOG.info("Loading model: path=%s", args.model)
    strict = not (lora := cfg.aurora_config.use_lora) and (cfg.extra_variables is None)
    model = load_model(args.model, train=True, strict=strict, **cfg.aurora_init_kwargs)
    LOG.info("Loaded model: strict=%s, kwargs=%s", strict, cfg.aurora_init_kwargs)

    LOG.info("Loading model parameters and optimiser: lora=%s", lora)
    params = get_lora_params(model) if lora else list(model.parameters())
    optimiser = torch.optim.AdamW(params, lr=float(cfg.learning_rate))
    batch_fn = partial(cfg.mode.batch_fn, data_path=args.data, **cfg.variable_map)

    LOG.info(
        "Starting fine-tuning: start=%s, stop=%s, epochs=%d",
        args.start_datetime,
        args.end_datetime,
        cfg.epochs,
    )
    timestamps = get_datetime_range(args.start_datetime, args.end_datetime)
    prediction, loss_history = FINETUNE_FNS[cfg.type](
        model=model,
        params=params,
        optimiser=optimiser,
        batch_fn=batch_fn,
        timestamps=timestamps,
        epochs=cfg.epochs,
        rollout_steps=cfg.rollout_steps,
        area_weighted=cfg.area_weighted,
    )
    model = model.to("cpu")

    LOG.info("Writing results: loss=%s, prediction=%s", args.loss, args.prediction)
    np.save(args.loss, np.array(loss_history, dtype=float))
    ds = batch_to_xarray(prediction)
    ds.to_netcdf(args.prediction)

    LOG.info(
        "Writing fine-tuned model checkpoint: path=%s, lora_only=%s",
        args.checkpoint,
        lora,
    )
    model_state = {
        k: (
            v.to(dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
            else v
        )
        for k, v in model.state_dict().items()
    }
    if lora:
        base_state = torch.load(args.model, map_location="cpu", weights_only=True)
        base_state.update(model_state)
        torch.save(base_state, args.checkpoint)
    else:
        torch.save(model_state, args.checkpoint)

    LOG.info("Done!")
