"""Aurora workshop job runner (Azure ML entrypoint).

This is the script Azure ML executes on the GPU node.

Think of it like this:
- The **notebook** is where you choose what to run (toy vs ERA5, short vs rollout, LoRA on/off).
- This script reads those choices from environment variables and runs the right code in
  `aurora_demo_core.py`.

Two big modes (FLOW):
- FLOW=toy   → quick “does everything work?” demo (random data + dummy loss)
- FLOW=era5  → real fine‑tuning on your local ERA5 subset (Zarr + static.nc)

Nothing here downloads ERA5 from the internet.
If FLOW=era5, the job expects Azure ML to mount two inputs:
- ERA5_ZARR_PATH   : path to the local Zarr folder (dynamic vars)
- ERA5_STATIC_NC   : path to the local NetCDF file (static vars: lsm/slt/z)

Outputs
-------
We write all outputs under:  OUT_DIR / PARTICIPANT_ID

This keeps workshop runs tidy because each person gets their own folder."""

from __future__ import annotations

import json
import os
import atexit
import time
import re



# -----------------------------------------------------------------------------
# Environment variables this script understands (the master notebook sets these)
# -----------------------------------------------------------------------------
# Required for all runs:
#   OUT_DIR           Where to write outputs (Azure ML mounts this as an output folder)
#   PARTICIPANT_ID    A short name so each attendee gets their own output folder
#   DEVICE            'cuda' or 'cpu' (normally 'cuda')
#   FINETUNE_STEPS    0 = inference only, >0 = run fine‑tuning as well
#
# Flow selection:
#   FLOW              'toy' or 'era5'
#
# ERA5 inputs (only when FLOW=era5):
#   ERA5_ZARR_PATH    Path to ERA5 subset Zarr folder
#   ERA5_STATIC_NC    Path to static NetCDF with variables lsm/slt/z
#
# Fine‑tuning mode (When FLOW=era5):
#   FT_MODE           'short' or 'rollout'
#   LEAD_HOURS        6 by default (so training is [t-6h,t] -> t+6h)
#   ERA5_TIME_INDEX   Which time index to use inside the month (We are using September 2025 data)
#   ERA5_CROP_LAT     Crop height (We are cropping data as global grid will be too big)
#   ERA5_CROP_LON     Crop width  (We are cropping data as global grid will be too big. The lon should be multiple of 4)
#
# Rollout extras (FT_MODE=rollout):
#   ROLLOUT_HORIZON_STEPS   How many steps ahead to roll (e.g., 8 = 2 days at 6h)
#   ROLLOUT_LOSS_ON         'last' (cheap) or 'sum' (heavier)
#
# LoRA knobs (optional):
#   USE_LORA          1 or 0
#   TRAIN_LORA_ONLY   1 = freeze base model, train LoRA params only
#   LORA_MODE         'single' | 'from_second' | 'all' 
#   LORA_STEPS        how many rollout steps LoRA should cover
#
# Add ONE extra variable (optional):
#   EXTRA_KIND        'surf' | 'static' | 'atmos'
#   EXTRA_KEY         Aurora key name for the new variable (e.g. 'sp')
#   EXTRA_SRC         Dataset variable name to read (must exist in your Zarr/static.nc)
#   EXTRA_LOCATION    normalisation centre (float)
#   EXTRA_SCALE       normalisation scale  (float)
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from aurora_demo_core import (
    ExtraVar,
    run_finetuning,
    run_finetuning_era5_rollout,
    run_finetuning_era5_short_lead,
    run_inference,
    run_inference_era5,
)


def batch_to_npz(prediction, path: Path) -> None:
    """Save a single aurora.Batch prediction to NPZ."""
    data = {}

    # Surface variables
    for name, tensor in prediction.surf_vars.items():
        data[f"surf_{name}"] = tensor.detach().cpu().numpy()

    # Static variables (if present)
    if hasattr(prediction, "static_vars"):
        for name, tensor in prediction.static_vars.items():
            data[f"static_{name}"] = tensor.detach().cpu().numpy()

    # Atmospheric variables
    for name, tensor in prediction.atmos_vars.items():
        data[f"atmos_{name}"] = tensor.detach().cpu().numpy()

    # Metadata
    if hasattr(prediction, "metadata"):
        md = prediction.metadata
        if hasattr(md, "lat") and md.lat is not None:
            data["lat"] = md.lat.detach().cpu().numpy() if torch.is_tensor(md.lat) else np.array(md.lat)
        if hasattr(md, "lon") and md.lon is not None:
            data["lon"] = md.lon.detach().cpu().numpy() if torch.is_tensor(md.lon) else np.array(md.lon)
        if hasattr(md, "atmos_levels") and md.atmos_levels is not None:
            data["atmos_levels"] = (
                md.atmos_levels.detach().cpu().numpy()
                if torch.is_tensor(md.atmos_levels)
                else np.array(md.atmos_levels)
            )
        if hasattr(md, "time") and md.time is not None:
            data["time"] = np.array([t.isoformat() for t in md.time])

    np.savez(path, **data)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, str(int(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _parse_extra_var_from_env() -> Optional[ExtraVar]:
    key = os.getenv("EXTRA_KEY", "").strip()
    if not key:
        return None
    kind = os.getenv("EXTRA_KIND", "surf").strip()
    src = os.getenv("EXTRA_SRC", "").strip()
    if not src:
        raise ValueError("EXTRA_KEY set but EXTRA_SRC is empty.")
    loc = _env_float("EXTRA_LOCATION", 0.0)
    scale = _env_float("EXTRA_SCALE", 1.0)
    return ExtraVar(kind=kind, key=key, src=src, location=loc, scale=scale)

def _start_mlflow(config: dict, tags: dict):
    """AML-only MLflow setup.
    """
    if not _env_bool("LOG_MLFLOW", False):
        print("[aurora-demo][mlflow] LOG_MLFLOW disabled -> skipping")
        return None

    try:
        import mlflow
    except Exception as e:
        print(f"[aurora-demo][mlflow] import mlflow failed: {e}")
        return None

    # AzureML MLflow integration must be present for azureml://
    try:
        import azureml.mlflow 
    except Exception as e:
        print(f"[aurora-demo][mlflow] azureml-mlflow missing/unavailable: {e}")
        return None

    env_tracking = os.getenv("MLFLOW_TRACKING_URI")
    rid = (os.getenv("MLFLOW_RUN_ID") or "").strip()
    print("[aurora-demo][mlflow] tracking_uri_env =", env_tracking)
    print("[aurora-demo][mlflow] run_id_env =", rid)

    # Confirm AML tracking
    tracking_uri = (env_tracking or mlflow.get_tracking_uri() or "").strip()
    is_azureml = tracking_uri.lower().startswith("azureml")
    print("[aurora-demo][mlflow] tracking_uri_resolved =", tracking_uri)
    if not is_azureml:
        print("[aurora-demo][mlflow] ERROR: Not using azureml:// tracking. Aborting MLflow setup.")
        return None

    if rid and not re.fullmatch(r"[0-9a-f]{32}", rid.lower()):
        print(f"[aurora-demo][mlflow] Clearing invalid MLFLOW_RUN_ID='{rid}'")
        os.environ.pop("MLFLOW_RUN_ID", None)

    for k in ("MLFLOW_REGISTRY_URI", "MLFLOW_MODEL_REGISTRY_URI"):
        v = (os.getenv(k) or "").strip().lower()
        if v.startswith("azureml"):
            print(f"[aurora-demo][mlflow] Clearing {k}='{v}'")
            os.environ.pop(k, None)

    parent_job = (os.getenv("PARENT_JOB_NAME") or "").strip()
    run_name = f"{parent_job}_mlflow" if parent_job else ((os.getenv("RUN_NAME") or "").strip() or "aurora_mlflow")

    try:
        # Standard tag used for display name
        mlflow.set_tag("mlflow.runName", run_name)
        print(f"[aurora-demo][mlflow] set mlflow.runName = {run_name}")
    except Exception as e:
        print(f"[aurora-demo][mlflow] failed to set runName tag: {e}")

    # Log params + tags + a smoke metric
    try:
        mlflow.log_params({k: str(v) for k, v in config.items() if v is not None})
        for k, v in tags.items():
            mlflow.set_tag(k, str(v))
        mlflow.log_metric("mlflow_smoke", 1.0)
        print("[aurora-demo][mlflow] params/tags/metric logged OK")
    except Exception as e:
        print(f"[aurora-demo][mlflow] logging failed: {e}")

    # Debug: show active run id if available
    try:
        ar = mlflow.active_run()
        print("[aurora-demo][mlflow] active_run =", None if ar is None else ar.info.run_id)
    except Exception as e:
        print(f"[aurora-demo][mlflow] active_run() failed: {e}")

    return mlflow






def _log_2t_plot(mlflow, npy_path: Path, out_dir: Path, prefix: str) -> None:
    """Create a PNG heatmap from a saved .npy field and log it to MLflow."""
    if not mlflow:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arr = np.load(npy_path)

        # --- FIX: convert to 2D for plotting ---
        if arr.ndim == 4:        # (B, T, H, W)
            arr2 = arr[0, -1]    # batch 0, last time
        elif arr.ndim == 3:      # (T, H, W)
            arr2 = arr[-1]       # last time
        elif arr.ndim == 2:      # (H, W)
            arr2 = arr
        else:
            raise ValueError(f"Invalid 2t array shape {arr.shape} for plotting")

        # Log scalar stats from the 2D field
        mlflow.log_metric(f"{prefix}_min", float(np.nanmin(arr2)))
        mlflow.log_metric(f"{prefix}_max", float(np.nanmax(arr2)))
        mlflow.log_metric(f"{prefix}_mean", float(np.nanmean(arr2)))

        fig = plt.figure()
        plt.imshow(arr2)
        plt.colorbar()
        plt.title(prefix)
        plt.tight_layout()

        png_path = out_dir / f"{prefix}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(str(png_path))
    except Exception as e:
        print(f"[aurora-demo] 2t plot logging failed (continuing): {e}")




def main() -> None:
    # ------------------------------------------------------------------
    # Read configuration from environment
    # ------------------------------------------------------------------
    device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    finetune_steps = _env_int("FINETUNE_STEPS", 0)

    flow = os.getenv("FLOW", "toy").strip().lower()
    if flow not in ("toy", "era5"):
        raise ValueError("FLOW must be 'toy' or 'era5'.")

    # Base output directory mounted by Azure ML
    base_out_dir = Path(os.getenv("OUT_DIR", "./outputs")).resolve()

    # Per-participant subfolder (set PARTICIPANT_ID in the notebook)
    participant_id = os.getenv("PARTICIPANT_ID", "unknown")
    out_dir = base_out_dir / participant_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[aurora-demo] flow={flow}, participant_id={participant_id}, "
        f"device={device}, finetune_steps={finetune_steps}"
    )
    print(f"[aurora-demo] Writing outputs under: {out_dir}")

    # ERA5 config (used when FLOW=era5)
    era5_zarr_path = os.getenv("ERA5_ZARR_PATH", "").strip()
    era5_static_nc = os.getenv("ERA5_STATIC_NC", "").strip()
    era5_time_index = _env_int("ERA5_TIME_INDEX", 10)
    era5_crop_lat = _env_int("ERA5_CROP_LAT", 128)
    era5_crop_lon = _env_int("ERA5_CROP_LON", 256)
    era5_lead_hours = _env_int("ERA5_LEAD_HOURS", 6)

    infer_rollout_steps_int = _env_int("INFER_ROLLOUT_STEPS", 0)
    infer_rollout_steps_int = infer_rollout_steps_int or None

    ft_mode = os.getenv("FT_MODE", "short").strip().lower()  # short | rollout

    # LoRA config
    use_lora = _env_bool("USE_LORA", default=(ft_mode == "rollout"))
    train_lora_only = _env_bool("TRAIN_LORA_ONLY", False)
    lora_mode = os.getenv("LORA_MODE", "all" if ft_mode == "rollout" else "single").strip()
    lora_steps = _env_int("LORA_STEPS", 40)

    stabilise_level_agg = _env_bool("STABILISE_LEVEL_AGG", False)

    # Rollout FT config
    rollout_horizon = _env_int("ROLLOUT_HORIZON_STEPS", 8)
    rollout_loss_on = os.getenv("ROLLOUT_LOSS_ON", "last").strip().lower()  # last|sum

    extra_var = _parse_extra_var_from_env()

    # ------------------------------------------------------------------
    # MLflow: log params/tags for experiment comparison
    # ------------------------------------------------------------------
    config = {
        "flow": flow,
        "ft_mode": ft_mode,
        "device": device,
        "participant_id": participant_id,
        "finetune_steps": finetune_steps,

        # ERA5 settings (still safe to log even if FLOW=toy)
        "era5_lead_hours": era5_lead_hours,
        "era5_time_index": era5_time_index,
        "era5_crop_lat": era5_crop_lat,
        "era5_crop_lon": era5_crop_lon,
        "autocast": _env_bool("AUTOCAST", True),
        "lr": _env_float("LR", 3e-5),

        # LoRA settings
        "use_lora": use_lora,
        "train_lora_only": train_lora_only,
        "lora_mode": lora_mode,
        "lora_steps": lora_steps,

        # Rollout settings
        "rollout_horizon": rollout_horizon,
        "rollout_loss_on": rollout_loss_on,
        "infer_rollout_steps": infer_rollout_steps_int,

        # Extra var info (just enough to filter runs)
        "extra_key": os.getenv("EXTRA_KEY", "").strip() or None,
        "extra_kind": os.getenv("EXTRA_KIND", "").strip() or None,
        "extra_src": os.getenv("EXTRA_SRC", "").strip() or None,
    }

    tags = {
        "flow": flow,
        "ft_mode": ft_mode,
        "participant_id": participant_id,
        "use_lora": use_lora,
    }

    mlflow = _start_mlflow(config, tags)


    # ------------------------------------------------------------------
    # 1) Inference
    # ------------------------------------------------------------------
    print(f"[aurora-demo] Running inference (flow={flow})")

    t0_inf = time.time()

    if flow == "toy":
        prediction = run_inference(device=device, rollout_steps=infer_rollout_steps_int)
    else:
        if not era5_zarr_path or not era5_static_nc:
            raise ValueError("FLOW=era5 requires ERA5_ZARR_PATH and ERA5_STATIC_NC.")
        prediction = run_inference_era5(
            era5_zarr_path=era5_zarr_path,
            era5_static_nc=era5_static_nc,
            device=device,
            time_index=era5_time_index,
            lead_hours=era5_lead_hours,
            crop_lat=era5_crop_lat,
            crop_lon=era5_crop_lon,
            rollout_steps=infer_rollout_steps_int,
            use_lora=False,  # inference doesn't need LoRA fine-tuning
            stabilise_level_agg=stabilise_level_agg,
            extra_var=extra_var,
        )

    inf_seconds = time.time() - t0_inf
    print(f"[aurora-demo] inference_seconds={inf_seconds:.3f}")
    if mlflow:
        try:
            mlflow.log_metric("inference_seconds", float(inf_seconds))
        except Exception as e:
            print(f"[aurora-demo] MLflow inference_seconds failed (continuing): {e}")

    # If rollout inference, save last step like the normal path, and optionally save the whole rollout.
    if isinstance(prediction, list):
        pred_last = prediction[-1]
        # Save 2t for every step
        if hasattr(pred_last, "surf_vars") and "2t" in pred_last.surf_vars:
            t2_all = np.stack([p.surf_vars["2t"].detach().cpu().numpy() for p in prediction], axis=0)
            np.save(out_dir / "inference_rollout_2t.npy", t2_all)
            print("[aurora-demo] Saved inference_rollout_2t.npy")
            if mlflow:
                try:
                    mlflow.log_artifact(str(out_dir / "inference_rollout_2t.npy"))
                except Exception as e:
                    print(f"[aurora-demo] MLflow log_artifact(inference_rollout_2t) failed (continuing): {e}")

        prediction_to_save = pred_last
    else:
        prediction_to_save = prediction

    # Save FULL inference prediction (all vars + metadata)
    inf_full_path = out_dir / "inference_full_prediction.npz"
    batch_to_npz(prediction_to_save, inf_full_path)
    print(f"[aurora-demo] Saved full inference prediction to {inf_full_path}")

    if mlflow:
        try:
            mlflow.log_artifact(str(inf_full_path))
        except Exception as e:
            print(f"[aurora-demo] MLflow log_artifact(inference_full) failed (continuing): {e}")


    # Save 2m temperature field only
    if hasattr(prediction_to_save, "surf_vars") and "2t" in prediction_to_save.surf_vars:
        t2m = prediction_to_save.surf_vars["2t"].detach().cpu().numpy()
        inf_2t_path = out_dir / "inference_2t.npy"
        np.save(inf_2t_path, t2m)
        print(f"[aurora-demo] Saved inference 2m temperature to {inf_2t_path}")
        _log_2t_plot(mlflow, inf_2t_path, out_dir, prefix="inference_2t")

    else:
        print("[aurora-demo] WARNING: '2t' not found in inference prediction; skipping inference_2t.npy")

    # ------------------------------------------------------------------
    # 2) Fine-tuning
    # ------------------------------------------------------------------
    if finetune_steps > 0:
        print(f"[aurora-demo] Running fine-tuning (flow={flow}, mode={ft_mode})")
        t0_ft = time.time()


        finetuned_model = None
        if flow == "toy":
            last_pred, loss_history = run_finetuning(steps=finetune_steps, device=device)
        else:
            if not era5_zarr_path or not era5_static_nc:
                raise ValueError("FLOW=era5 requires ERA5_ZARR_PATH and ERA5_STATIC_NC.")
            if ft_mode == "short":
                finetuned_model, last_pred, loss_history = run_finetuning_era5_short_lead(
                    steps=finetune_steps,
                    device=device,
                    era5_zarr_path=era5_zarr_path,
                    era5_static_nc=era5_static_nc,
                    time_index=era5_time_index,
                    lead_hours=era5_lead_hours,
                    crop_lat=era5_crop_lat,
                    crop_lon=era5_crop_lon,
                    lr=_env_float("LR", 3e-5),
                    use_lora=use_lora,
                    train_lora_only=train_lora_only,
                    lora_mode=lora_mode,
                    lora_steps=lora_steps,
                    stabilise_level_agg=stabilise_level_agg,
                    autocast=_env_bool("AUTOCAST", True),
                    extra_var=extra_var,
                )
            elif ft_mode == "rollout":
                finetuned_model, last_pred, loss_history = run_finetuning_era5_rollout(
                    steps=finetune_steps,
                    device=device,
                    era5_zarr_path=era5_zarr_path,
                    era5_static_nc=era5_static_nc,
                    time_index=era5_time_index,
                    lead_hours=era5_lead_hours,
                    rollout_horizon=rollout_horizon,
                    rollout_loss_on=rollout_loss_on,
                    crop_lat=era5_crop_lat,
                    crop_lon=era5_crop_lon,
                    lr=_env_float("LR", 3e-5),
                    use_lora=use_lora,
                    train_lora_only=train_lora_only,
                    lora_mode=lora_mode,
                    lora_steps=lora_steps,
                    stabilise_level_agg=stabilise_level_agg,
                    autocast=_env_bool("AUTOCAST", True),
                    extra_var=extra_var,
                )
            else:
                raise ValueError("FT_MODE must be 'short' or 'rollout'.")
            
        ft_seconds = time.time() - t0_ft
        print(f"[aurora-demo] finetune_seconds={ft_seconds:.3f}")
        if mlflow:
            try:
                mlflow.log_metric("finetune_seconds", float(ft_seconds))
            except Exception as e:
                print(f"[aurora-demo] MLflow finetune_seconds failed (continuing): {e}")


        # Save FULL final fine-tune prediction (all vars + metadata)
        fin_full_path = out_dir / "finetune_last_prediction.npz"
        batch_to_npz(last_pred, fin_full_path)
        print(f"[aurora-demo] Saved final fine-tune prediction to {fin_full_path}")
        if mlflow:
            try:
                mlflow.log_artifact(str(fin_full_path))
            except Exception as e:
                print(f"[aurora-demo] MLflow log_artifact(finetune_pred) failed (continuing): {e}")


        # Save 2m temperature from final prediction (for plotting)
        if hasattr(last_pred, "surf_vars") and "2t" in last_pred.surf_vars:
            t2m_last = last_pred.surf_vars["2t"].detach().cpu().numpy()
            fin_2t_path = out_dir / "finetune_last_2t.npy"
            np.save(fin_2t_path, t2m_last)
            print(f"[aurora-demo] Saved final fine-tune 2m temperature to {fin_2t_path}")
            _log_2t_plot(mlflow, fin_2t_path, out_dir, prefix="finetune_last_2t")

        else:
            print("[aurora-demo] WARNING: '2t' not found in final prediction; skipping finetune_last_2t.npy")

        # Save loss history
        losses_json_path = out_dir / "finetune_losses.json"
        losses_npy_path = out_dir / "finetune_losses.npy"
        losses_json_path.write_text(json.dumps(loss_history, indent=2))
        np.save(losses_npy_path, np.array(loss_history, dtype=float))
        print(f"[aurora-demo] Saved loss history to {losses_json_path} and {losses_npy_path}")
        if mlflow:
            try:
                mlflow.log_artifact(str(losses_json_path))
                mlflow.log_artifact(str(losses_npy_path))
            except Exception as e:
                print(f"[aurora-demo] MLflow log_artifact(losses) failed (continuing): {e}")


        # ------------------------------------------------------------------
        # MLflow metrics: log loss curve for easy run comparison
        # ------------------------------------------------------------------
        if mlflow and loss_history:
            try:
                for step, loss in enumerate(loss_history):
                    mlflow.log_metric("train_loss", float(loss), step=step)

                mlflow.log_metric("train_loss_final", float(loss_history[-1]))
                mlflow.log_metric("train_loss_best", float(min(loss_history)))
                mlflow.log_metric("train_steps", float(len(loss_history)))
            except Exception as e:
                print(f"[aurora-demo] MLflow logging failed (continuing): {e}")


        # Save fine-tuned weights (ERA5 flow only)
        if finetuned_model is not None:
            weights_path = out_dir / "finetuned_state_dict.pt"

            # The full Aurora model is large. If you are doing LoRA-only fine-tuning,
            # saving the full state_dict can be slow and will upload a big artifact.
            #
            # For this workshops we prefer to save only the LoRA weights,
            # and keep the base checkpoint unchanged.
            state = finetuned_model.state_dict()
            if use_lora and train_lora_only:
                state = {k: v for k, v in state.items() if "lora" in k.lower()}
            if mlflow:
                try:
                    total = sum(p.numel() for p in finetuned_model.parameters())
                    trainable = sum(p.numel() for p in finetuned_model.parameters() if p.requires_grad)
                    lora = sum(p.numel() for n, p in finetuned_model.named_parameters() if "lora" in n.lower())
                    mlflow.log_metric("params_total", float(total))
                    mlflow.log_metric("params_trainable", float(trainable))
                    mlflow.log_metric("params_lora", float(lora))

                    if torch.cuda.is_available():
                        mlflow.log_metric("cuda_max_mem_mb", float(torch.cuda.max_memory_allocated() / (1024**2)))
                        mlflow.set_tag("gpu_name", torch.cuda.get_device_name(0))
                except Exception as e:
                    print(f"[aurora-demo] MLflow model stats failed (continuing): {e}")

            torch.save(state, weights_path)
            print(f"[aurora-demo] Saved fine-tuned weights to {weights_path}")
            if mlflow:
                try:
                    mlflow.log_artifact(str(weights_path))
                except Exception as e:
                    print(f"[aurora-demo] MLflow log_artifact(weights) failed (continuing): {e}")


    else:
        print("[aurora-demo] FINETUNE_STEPS=0 – skipping fine-tuning")

    if mlflow and getattr(mlflow, "_AURORA_STARTED_RUN", False):
        try:
            mlflow.end_run()
        except Exception:
            pass


    print(f"[aurora-demo] Done. Outputs are in {out_dir}")


if __name__ == "__main__":
    main()