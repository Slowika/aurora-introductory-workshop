# Aurora Workshop: ERA5 Inference + Fine-Tuning on Azure ML

This repository contains a practical, workshop-friendly code for running **Microsoft Aurora** weather model **inference** and **fine-tuning** on **ERA5** data using **Azure Machine Learning Studio**.

It supports:

* ✅ **ERA5 inference** (short-lead & optional rollout inference)
* ✅ **ERA5 short-lead fine-tuning**
* ✅ **ERA5 rollout / autoregressive fine-tuning**
* ✅ **LoRA fine-tuning** (train LoRA only) or full fine-tune
* ✅ **MLflow logging** to AzureML (metrics + artifacts + images)
* ✅ Repeatable runs via profiles (env var presets)

The workflow is designed for training/education and experimentation: you choose a run profile in a “master notebook” and submit a single AzureML command job.

---

## Repository structure

Typical layout:

```
.
├── 0_aurora_workshop.ipynb          # Master notebook: pick profile & submit job to AzureML
├── run_aurora_job.py                # Job entrypoint: reads env vars, runs inference + finetune
├── aurora_demo_core.py              # Core logic: build model, create batches, finetune loops
├── era5_subsets/                    # (Local/dev) ERA5 zarr subset + static nc
```

### Key files

* **`0_aurora_workshop.ipynb`**
  The control panel. Defines run profiles (toy, era5_short, era5_rollout, lora variants), mounts inputs, sets env vars, submits a v2 `command()` job.

* **`run_aurora_job.py`**
  The job entrypoint executed on AzureML compute. It:

  * reads env vars
  * runs inference
  * optionally fine-tunes
  * saves outputs to `OUT_DIR/<participant_id>/...`
  * logs metrics & artifacts to MLflow (AzureML)

* **`aurora_demo_core.py`**
  The “engine”: model construction, loading checkpoints, creating Aurora `Batch`, ERA5 data loading, loss loops, short-lead and rollout fine-tuning implementations, LoRA hooks.

---

## What the job does

### Inference (ERA5)

1. Load an ERA5 batch from your mounted Zarr subset (dynamic vars).
2. Load static fields (lsm/slt/z) from a mounted NetCDF static file.
3. Run Aurora forward pass:

   * short-lead: produce next-step prediction
   * optional rollout inference: multiple autoregressive steps
4. Save outputs:

   * full prediction as `.npz`
   * 2m temperature (“2t”) as `.npy`
   * a PNG heatmap of `2t` (optional via MLflow artifact logging)

### Fine-tuning (ERA5)

Two modes:

#### 1) Short-lead fine-tuning (`FT_MODE=short`)

* Train on a single lead (e.g., 6 hours ahead).
* Steps controlled by `FINETUNE_STEPS`.

#### 2) Rollout/autoregressive fine-tuning (`FT_MODE=rollout`)

* Unroll multiple steps autoregressively.
* Inner horizon controlled by `ROLLOUT_HORIZON_STEPS`.
* Outer training iterations controlled by `FINETUNE_STEPS`.
* Rollout loss strategy controlled by `ROLLOUT_LOSS_ON` (`last` vs `sum`).

### LoRA

With LoRA enabled, you can:

* attach LoRA adapters to attention projections
* optionally freeze all base weights and train only LoRA parameters (`TRAIN_LORA_ONLY=1`)

---

## Data requirements (ERA5)

Aurora expects a **Batch** with:

* **Surface variables** (e.g., `2t`, `10u`, `10v`, `msl`, etc.)
* **Atmospheric variables** (pressure-level variables for supported levels)
* **Static variables** (must be 2D, **lat/lon only**):

  * `lsm` land-sea mask
  * `slt` soil type
  * `z` surface geopotential / orography

### Critical shape rule for statics

Static vars must be **2D** arrays `(H, W)`.

If your static NetCDF includes `time` or `valid_time` (common for CDS downloads), you must drop it:

* ✅ correct: `(latitude, longitude)`
* ❌ incorrect: `(time, latitude, longitude)` or `(1, H, W)`

---

## Running on Azure ML (v2)

### 1) Choose a profile

Profiles are simply environment-variable presets. Examples:

* `era5_short` – short-lead fine-tuning
* `era5_short_lora` – short-lead fine-tuning with LoRA
* `era5_rollout_no_lora` – autoregressive rollout fine-tuning without LoRA

The master notebook sets env vars like:

* `FLOW=era5`
* `FT_MODE=short` or `rollout`
* `FINETUNE_STEPS=...`
* `USE_LORA=0/1`, `TRAIN_LORA_ONLY=0/1`
* `ROLLOUT_HORIZON_STEPS=...`

### 2) Submit the job

The notebook submits a v2 `command()` job with:

* `inputs`:

  * `era5_zarr` (URI_FOLDER mount)
  * `era5_static` (URI_FILE mount)
* `outputs`:

  * `out_dir` (URI_FOLDER output mount)
* `environment_variables`:

  * run profile settings (above)
* `command`:

  * optional pip installs (zarr<3, azureml-mlflow, etc.)
  * exports mount paths into `ERA5_ZARR_PATH` and `ERA5_STATIC_NC`
  * runs `python run_aurora_job.py`

---

## MLflow logging (AzureML)

### What is logged

* **Metrics**

  * `inference_seconds`
  * `finetune_seconds`
  * `train_loss` (curve per step)
  * `train_loss_best`, `train_loss_final`
  * `params_total`, `params_trainable`, `params_lora`
  * `*_min/max/mean` stats for plotted arrays

* **Artifacts**

  * `inference_full_prediction.npz`
  * `inference_2t.npy`
  * `inference_2t.png`
  * `finetune_last_prediction.npz`
  * `finetune_last_2t.npy`
  * `finetune_last_2t.png`
  * `finetune_losses.json` / `.npy`
  * `finetuned_state_dict.pt`

### Important AzureML UI note

In AzureML v2, MLflow metrics/artifacts often appear under the **MLflow run record**, which can show up as a separate entry in job's list. This is expected behavior for MLflow tracking.

### Naming the MLflow run to match the command job

Set an env var in the notebook:

```python
env_vars["PARENT_JOB_NAME"] = JOB_NAME
```

Then `_start_mlflow()` sets:

```
mlflow.runName = f"{PARENT_JOB_NAME}_mlflow"
```

This produces a clear relationship in the UI between:

* the command job name
* the MLflow tracking run name

---

## Output files

All outputs go under:

```
OUT_DIR/<participant_id>/
```

### Inference outputs

* `inference_full_prediction.npz`
  Compressed arrays of predicted fields.

* `inference_2t.npy`
  2m temperature array (often stored as `(B,T,H,W)`).

* `inference_2t.png`
  2D heatmap image derived from the array.

### Fine-tuning outputs

* `finetune_last_prediction.npz`
* `finetune_last_2t.npy`
* `finetune_last_2t.png`
* `finetune_losses.json`
* `finetune_losses.npy`
* `finetuned_state_dict.pt`

  * LoRA-only run save only `lora_*` params.

---

## Configuration reference (env vars)

### Core

* `FLOW`: `toy` | `era5`
* `DEVICE`: `cuda` | `cpu`
* `PARTICIPANT_ID`: output subfolder name
* `OUT_DIR`: output root folder

### Fine-tuning

* `FINETUNE_STEPS`: number of optimizer steps (outer loop)
* `FT_MODE`: `short` | `rollout`
* `LR`: learning rate
* `AUTOCAST`: `1/0` to use torch autocast

### ERA5 selection

* `ERA5_LEAD_HOURS`: lead time for prediction/fine-tuning (e.g. 6)
* `ERA5_TIME_INDEX`: selects which time index from your subset to use
* `ERA5_CROP_LAT`, `ERA5_CROP_LON`: spatial crop size

### Rollout/autoregressive

* `ROLLOUT_HORIZON_STEPS`: how many steps to unroll
* `ROLLOUT_LOSS_ON`: `last` or `sum`
* `INFER_ROLLOUT_STEPS`: rollout steps during inference (optional)

### LoRA

* `USE_LORA`: `1/0`
* `TRAIN_LORA_ONLY`: `1/0` (freeze base weights)
* `LORA_MODE`: `single` (or other supported modes)
* `LORA_STEPS`: model config knob (not training loop length)

### MLflow

* `LOG_MLFLOW`: `1/0`
* `RUN_NAME`: optional label
* `PARENT_JOB_NAME`: recommended for UI naming (`JOB_NAME` from notebook)

---

## Troubleshooting

### 1) Static variables are NaN or model crashes

* Ensure your static file contains `lsm`, `slt`, `z` with no NaNs.
* Ensure statics are **2D** (drop `time`/`valid_time`).
* If model errors:
  `repeat dims can not be smaller than number of dimensions`
  → at least one static var is not 2D.

### 2) LoRA checkpoint loading fails

If you enable LoRA but load a base checkpoint with strict loading, you’ll see missing keys like `lora_A/lora_B`.
Solution: load base weights with `strict=False` when LoRA modules are present.

### 3) `Invalid shape (1,1,H,W) for image data`

Your `2t` arrays are `(B,T,H,W)`. Slice before plotting:

* `arr2 = arr[0, -1]`

### 4) MLflow “Run not found”

If `MLFLOW_RUN_ID` is set to a human string, clear it:

* `os.environ.pop("MLFLOW_RUN_ID", None)`


