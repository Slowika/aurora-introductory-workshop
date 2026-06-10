"""Workshop setup constants."""

from pathlib import Path

_ROOT_DIR = Path(__file__).parents[2]

# checkpoint to load and where to retrieve it from
# DO NOT EDIT - workshop designed for this checkpoint
HF_REPOSITORY = "microsoft/aurora"
MODEL_FILENAME = "aurora-0.25-pretrained.ckpt"

# config files
INFERENCE_CONFIG_PATH = _ROOT_DIR / "notebooks/inference_configs.yaml"
FINETUNE_CONFIG_PATH = _ROOT_DIR / "notebooks/finetune_configs.yaml"

# input data URIs, paths, and params
GCP_ERA5_URI = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DATA_DIR = _ROOT_DIR / "data"
LOCAL_DATA_PATH = DATA_DIR / "era5_ar.zarr"
START_DATETIME = "2025-01-01T06:00:00"
END_DATETIME = "2025-01-31T23:00:00"
FREQUENCY = 6

# output data paths and names
OUTPUTS_DIR = DATA_DIR / "outputs"
INFERENCE_OUT_DIR = OUTPUTS_DIR / "inference"
FINETUNE_OUT_DIR = OUTPUTS_DIR / "finetuning"
INFERENCE_PREDS_FILENAME = "predictions.nc"
FINETUNE_PRED_FILENAME = "prediction.nc"
FINETUNE_LOSS_FILENAME = "loss.npy"
FINETUNE_CKPT_FILENAME = "finetuned.ckpt"

# ids under which to write and register loaded data and model as AML workspace assets
MODEL_ASSET_NAME = "aurora-0p25-pretrained"  # alnum chars and dashes only
DATA_ASSET_NAME = "gcp-era5-arco"
REMOTE_DIRNAME = "aurora-workshop"
