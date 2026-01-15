"""Aurora workshop core (toy + ERA5) — model logic only.

- **This file is just the “engine”.**
  It knows how to build Aurora batches, run inference, and run fine‑tuning steps.

- **The master notebook (`0_aurora_workshop.ipynb`) is the “remote control”.**
    it submits an Azure ML job and sets
  a few environment variables (FLOW, FT_MODE, etc.). `run_aurora_job.py` reads
  those variables and calls the right functions in this file.

Two flows are supported:

1) **Toy flow** (warm‑up / sanity check)
   - Creates a tiny random batch (17×32 grid) so we can prove:
     “the environment works, the model loads, GPU runs, outputs get saved”.
   - The “loss” here is deliberately fake. It’s only to prove backprop works.

2) **ERA5 flow** (the real workshop content)
   - Uses your **already-downloaded** ERA5 subset:
       • a local **Zarr folder** for dynamic variables (time‑varying)
       • a local **NetCDF file** for static variables (lsm/slt/z)
   - This flow never downloads from the internet. It only reads local paths (AML data assets) that
     Azure ML mounts into the job.

Quick terms explnation:

- **Short‑lead fine‑tuning**: learn the next step
    Input history: [t-6h, t]  →  Target: [t+6h]

- **Long‑lead / rollout fine‑tuning**: learn to stay stable across many steps
    Model predicts step‑by‑step, feeding its own output back in.

- **LoRA**: a lightweight way to fine‑tune
    We freeze the big model weights and train tiny adapter weights instead.

Batch shapes (handy when debugging):

- surface vars:  (batch=1, history=2, lat, lon)
- atmos vars:    (batch=1, history=2, level, lat, lon)
- static vars:   (lat, lon)
- target batch:  same, but history=1 (only the “next” time)
- Keep `crop_lat` / `crop_lon` small (e.g., 128×256) to avoid OOM on shared GPUs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, Sequence, Tuple

import dataclasses
import torch

from aurora import AuroraPretrained, Batch, Metadata, rollout

# -----------------------------------------------------------------------------
# Workshop quick start (how this file is used)
# -----------------------------------------------------------------------------
# You normally don’t call these functions manually in the workshop.
# Instead:
#   1) You run `0_aurora_workshop.ipynb` (the master notebook).
#   2) The notebook submits an Azure ML job that runs `run_aurora_job.py`.
#   3) `run_aurora_job.py` reads env vars like FLOW / FT_MODE / USE_LORA and then
#      calls the right functions below.
#
# If you *do* want to run locally (no Azure ML), you can run:
#   FLOW=era5 FT_MODE=short ERA5_ZARR_PATH=/path/to/subset.zarr ERA5_STATIC_NC=/path/to/era5_static.nc \
#   python run_aurora_job.py
#
# The toy flow (FLOW=toy) is kept on purpose. It’s a reliable “does my env work?” check.
# The ERA5 flow (FLOW=era5) is where the real learning happens.
# -----------------------------------------------------------------------------



# =============================================================================
# toy demo
# =============================================================================

def make_lowres_batch(device: str = "cpu") -> Batch:
    """Create a small 17x32 random batch."""
    return Batch(
        surf_vars={k: torch.randn(1, 2, 17, 32, device=device)
                   for k in ("2t", "10u", "10v", "msl")},
        static_vars={k: torch.randn(17, 32, device=device)
                     for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 4, 17, 32, device=device)
                    for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 17, device=device),
            lon=torch.linspace(0, 360, 32 + 1, device=device)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )


def loss(pred: Batch) -> torch.Tensor:
    """Toy loss: sum of squared values over all predicted fields."""
    surf_values = pred.surf_vars.values()
    atmos_values = pred.atmos_vars.values()
    total = torch.tensor(0.0, device=next(iter(surf_values)).device)
    for x in list(surf_values) + list(atmos_values):
        total = total + (x * x).sum()
    return total


def run_inference(device: str = "cpu", rollout_steps: Optional[int] = None):
    """ 
        Toy inference: load checkpoint and run on a random low-res batch.

        If rollout_steps is provided, performs an autoregressive rollout.
    """
    model = AuroraPretrained()
    model.load_checkpoint()
    model = model.to(device)
    model.eval()

    batch = make_lowres_batch(device=device)

    with torch.inference_mode():
        if rollout_steps is None:
            prediction = model(batch)
            return prediction.to("cpu")
        preds = [pred.to("cpu") for pred in rollout(model, batch, steps=rollout_steps)]
        return preds


def run_finetuning(steps: int = 10, device: str = "cuda"):
    """Toy fine-tuning loop on random LOW-RES data (17x32)."""
    model = AuroraPretrained()
    model.load_checkpoint()
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    last_prediction = None
    loss_history: list[float] = []

    for step in range(steps):
        batch = make_lowres_batch(device=device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss_value = loss(prediction)
        loss_value.backward()
        optimizer.step()

        last_prediction = prediction
        last_loss_value = float(loss_value.detach().cpu().item())
        loss_history.append(last_loss_value)

        print(f"[toy][step {step}] loss = {last_loss_value:.4f}")

    return last_prediction, loss_history


# =============================================================================
# ERA5 flow
# =============================================================================

# What the ERA5 flow expects
# ------------------------------------------
# You should already have:
#
#   1) Dynamic subset (Zarr folder) -  time-varying fields
#        Example: era5_aurora_2025-09_6hourly.zarr
#        This is what you created with your “download subset” notebook cell (Will already be done before Workshop as part of ENV setups)
#
#   2) Static file (NetCDF): Constants that do not change with time
#        Example: era5_static.nc containing variables:
#            - lsm (land/sea mask)
#            - slt (soil type)
#            - z   (surface geopotential / orography)
#
# Why keep static separate?
#   - It’s tiny and you load it once.
#   - Your dynamic Zarr stays focused on the time series (easy to version by month).
#
# A few practical details that matter a lot:
#   - Aurora expects latitude in **descending** order and longitude in **0..360** order.
#   - This code standardises those coordinates before creating the Batch.
#   - For workshops, we crop a small center patch from the global grid. Otherwise you will
#     run out of GPU memory.
#


# Standard Aurora meteorology variable mapping (ERA5 descriptive names -> Aurora keys)
ERA5_SURF_SRC = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}

ERA5_ATMOS_SRC = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}

# Aurora pressure levels used in examples (hPa)
AURORA_LEVELS: Tuple[int, ...] = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
)


@dataclass(frozen=True)
class ExtraVar:
    """Configuration for adding ONE extra variable to the model + data.

    kind:
      - "surf": dynamic surface variable (time, lat, lon)
      - "atmos": dynamic atmospheric variable (time, level, lat, lon)
      - "static": static variable (lat, lon) read from static_nc
    key:
      Aurora key (must be unique)
    src:
      variable name in the ERA5 Zarr (for surf/atmos) or static NetCDF (for static)
    location/scale:
      normalisation statistics to register in aurora.normalisation.locations/scales
      (required when introducing a new variable).
    """
    kind: Literal["surf", "atmos", "static"]
    key: str
    src: str
    location: float
    scale: float


def _lazy_import_xarray():
    import xarray as xr
    return xr


def _lazy_import_numpy():
    import numpy as np
    return np


def _standardise_latlon(ds):
    """Ensure lon in [0,360) increasing and lat decreasing."""
    xr = _lazy_import_xarray()
    np = _lazy_import_numpy()

    if "longitude" in ds.coords:
        ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby("longitude")

    if "latitude" in ds.coords:
        lat = ds.latitude.values
        if lat[0] < lat[-1]:
            ds = ds.sortby("latitude", ascending=False)

    return ds


def _crop_center(ds, crop_lat: int, crop_lon: int):
    """Crop a center patch to keep runs small in workshops."""
    if crop_lat <= 0 or crop_lon <= 0:
        return ds

    H = ds.sizes["latitude"]
    W = ds.sizes["longitude"]
    lat0 = max(0, (H - crop_lat) // 2)
    lon0 = max(0, (W - crop_lon) // 2)
    return ds.isel(latitude=slice(lat0, lat0 + crop_lat), longitude=slice(lon0, lon0 + crop_lon))


def _infer_time_step_hours(ds) -> int:
    np = _lazy_import_numpy()
    if ds.sizes["time"] < 2:
        raise ValueError("Need at least two timesteps in the ERA5 subset.")
    dt = (ds.time.values[1] - ds.time.values[0]) / np.timedelta64(1, "h")
    return int(dt)


def _lead_to_index_step(dt_hours: int, lead_hours: int) -> int:
    if lead_hours % dt_hours != 0:
        raise ValueError(f"lead_hours={lead_hours} not divisible by dt_hours={dt_hours}.")
    return lead_hours // dt_hours


def _open_era5_dynamic_zarr(zarr_path: str, crop_lat: int, crop_lon: int):
    print(f"Attempting to open ERA5 Zarr at: {zarr_path}")
    xr = _lazy_import_xarray()
    ds = xr.open_zarr(zarr_path, consolidated=True)
    ds = _standardise_latlon(ds)

    # keep only requested levels if present
    if "level" in ds.coords:
        wanted = [l for l in AURORA_LEVELS if l in set(ds.level.values.tolist())]
        if wanted:
            ds = ds.sel(level=wanted)

    ds = _crop_center(ds, crop_lat=crop_lat, crop_lon=crop_lon)
    return ds


def _open_static_nc(static_nc: str, lat_vals, lon_vals):
    xr = _lazy_import_xarray()
    ds = xr.open_dataset(static_nc, engine="netcdf4")
    ds = _standardise_latlon(ds)
    # align to the dynamic grid (important if crop was applied)
    ds = ds.sel(latitude=lat_vals, longitude=lon_vals)
    return ds


def _get_static_arrays(static_ds) -> dict[str, torch.Tensor]:
    """static_ds must contain variables: lsm, slt, z (2D)."""
    stat = {}
    for key in ("lsm", "slt", "z"):
        if key not in static_ds.data_vars:
            raise KeyError(f"Static NetCDF missing variable '{key}'. Expected lsm/slt/z.")
        da = static_ds[key]
        if "time" in da.dims:
            da = da.isel(time=0, drop=True)
        if "level" in da.dims:
            raise ValueError(f"Static var '{key}' unexpectedly has a level dimension.")
        stat[key] = torch.from_numpy(da.astype("float32").values)
    return stat


def supervised_mae(pred: Batch, target: Batch) -> torch.Tensor:
    """MAE over all surf + atmos variables in the prediction."""
    device = next(iter(pred.surf_vars.values())).device
    total = torch.tensor(0.0, device=device)

    for k in pred.surf_vars:
        total = total + (pred.surf_vars[k] - target.surf_vars[k]).abs().mean()
    for k in pred.atmos_vars:
        total = total + (pred.atmos_vars[k] - target.atmos_vars[k]).abs().mean()
    return total


def _extend_var_tuples(
    extra: Optional[ExtraVar],
    base_surf: Tuple[str, ...] = ("2t", "10u", "10v", "msl"),
    base_static: Tuple[str, ...] = ("lsm", "z", "slt"),
    base_atmos: Tuple[str, ...] = ("z", "u", "v", "t", "q"),
) -> tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    surf_vars, static_vars, atmos_vars = base_surf, base_static, base_atmos
    if extra is None:
        return surf_vars, static_vars, atmos_vars
    if extra.kind == "surf":
        surf_vars = surf_vars + (extra.key,)
    elif extra.kind == "static":
        static_vars = static_vars + (extra.key,)
    elif extra.kind == "atmos":
        atmos_vars = atmos_vars + (extra.key,)
    else:
        raise ValueError(f"Unknown extra.kind={extra.kind}")
    return surf_vars, static_vars, atmos_vars


def _register_new_var_stats(extra: Optional[ExtraVar]) -> None:
    """Register normalisation stats for a newly introduced variable."""
    if extra is None:
        return
    from aurora.normalisation import locations, scales  # type: ignore
    locations[extra.key] = float(extra.location)
    scales[extra.key] = float(extra.scale)



# -----------------------------------------------------------------------------
# LoRA + "add one new variable"
# -----------------------------------------------------------------------------
# These helpers keep the workshop flexible:
#
# - If `use_lora=True`, we ask Aurora to enable its built‑in LoRA adapters.
#   Then we freeze the base model and train only LoRA params.
#
# - If you pass `extra_var`, we extend the model to accept ONE new variable.
#   Important: your dataset must contain that variable, otherwise we
#   can’t build a Batch.
# -----------------------------------------------------------------------------


def build_model(
    *,
    use_lora: bool,
    lora_mode: Literal["single", "from_second", "all"],
    lora_steps: int,
    autocast: bool,
    stabilise_level_agg: bool,
    extra_var: Optional[ExtraVar] = None,
) -> AuroraPretrained:
    """Construct and load an Aurora model, optionally extended with one new variable.

    - LoRA parameters come from the Aurora constructor.
    - Adding new variables requires strict=False and registering locations/scales.
    """
    surf_vars, static_vars, atmos_vars = _extend_var_tuples(extra_var)

    model = AuroraPretrained(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        stabilise_level_agg=stabilise_level_agg,
        use_lora=use_lora,
        lora_steps=lora_steps,
        lora_mode=lora_mode,
        autocast=autocast,
    )

    # If we extended vars, checkpoint won't match exactly.

    # If LoRA is enabled, the base checkpoint will NOT contain LoRA weights -> must be strict=False.
    strict = (extra_var is None) and (not use_lora)
    model.load_checkpoint(strict=strict)

    
    _register_new_var_stats(extra_var)
    return model


def freeze_to_lora_only(model: torch.nn.Module) -> None:
    """Freeze all params except LoRA params"""
    for name, p in model.named_parameters():
        name_l = name.lower()
        p.requires_grad = ("lora" in name_l)

    # safety: ensure at least something is trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("No trainable parameters found after LoRA-freeze.")


def make_era5_short_lead_pair(
    *,
    era5_zarr_path: str,
    era5_static_nc: str,
    time_index: int,
    lead_hours: int,
    crop_lat: int,
    crop_lon: int,
    device: str,
    extra_var: Optional[ExtraVar] = None,
) -> tuple[Batch, Batch]:
    """Build (x, y) where x=[t-lead, t] and y=[t+lead]."""
    np = _lazy_import_numpy()

    ds = _open_era5_dynamic_zarr(era5_zarr_path, crop_lat=crop_lat, crop_lon=crop_lon)
    static_ds = _open_static_nc(era5_static_nc, ds.latitude, ds.longitude)
    stat = _get_static_arrays(static_ds)

    dt_hours = _infer_time_step_hours(ds)
    step = _lead_to_index_step(dt_hours, lead_hours)

    # choose safe indices
    idx = max(step, min(time_index, ds.sizes["time"] - step - 1))
    hist = [idx - step, idx]
    tgt = idx + step

    # build mappings including extra var
    surf_src = dict(ERA5_SURF_SRC)
    atmos_src = dict(ERA5_ATMOS_SRC)
    static_keys = ["lsm", "slt", "z"]

    if extra_var is not None:
        if extra_var.kind == "surf":
            surf_src[extra_var.key] = extra_var.src
        elif extra_var.kind == "atmos":
            atmos_src[extra_var.key] = extra_var.src
        elif extra_var.kind == "static":
            static_keys.append(extra_var.key)
        else:
            raise ValueError(extra_var.kind)

    def surf_hist(name):  # (1,2,H,W)
        arr = ds[name].isel(time=hist).astype("float32").values
        return torch.from_numpy(arr)[None].to(device)

    def atmos_hist(name):  # (1,2,C,H,W)
        arr = ds[name].isel(time=hist).astype("float32").values
        return torch.from_numpy(arr)[None].to(device)

    def surf_tgt(name):  # (1,1,H,W)
        arr = ds[name].isel(time=slice(tgt, tgt + 1)).astype("float32").values
        return torch.from_numpy(arr)[None].to(device)

    def atmos_tgt(name):  # (1,1,C,H,W)
        arr = ds[name].isel(time=slice(tgt, tgt + 1)).astype("float32").values
        return torch.from_numpy(arr)[None].to(device)

    current_time = ds.time.values[idx].astype("datetime64[s]").tolist()
    target_time = ds.time.values[tgt].astype("datetime64[s]").tolist()
    levels = tuple(int(x) for x in ds.level.values) if "level" in ds.coords else AURORA_LEVELS

    md_x = Metadata(
        lat=torch.from_numpy(ds.latitude.values.astype("float32")).to(device),
        lon=torch.from_numpy(ds.longitude.values.astype("float32")).to(device),
        time=(current_time,),
        atmos_levels=levels,
    )
    md_y = Metadata(
        lat=md_x.lat,
        lon=md_x.lon,
        time=(target_time,),
        atmos_levels=levels,
    )

    static_vars = {
        "lsm": stat["lsm"].to(device),
        "slt": stat["slt"].to(device),
        "z": stat["z"].to(device),
    }

    if extra_var is not None and extra_var.kind == "static":
        if extra_var.src not in static_ds.data_vars and extra_var.key not in static_ds.data_vars:
            raise KeyError(
                f"Static nc does not contain '{extra_var.src}' (or '{extra_var.key}')."
            )
        da = static_ds[extra_var.src] if extra_var.src in static_ds.data_vars else static_ds[extra_var.key]
        if "time" in da.dims:
            da = da.isel(time=0, drop=True)
        static_vars[extra_var.key] = torch.from_numpy(da.astype("float32").values).to(device)

    x = Batch(
        surf_vars={k: surf_hist(v) for k, v in surf_src.items()},
        static_vars=static_vars,
        atmos_vars={k: atmos_hist(v) for k, v in atmos_src.items()},
        metadata=md_x,
    )

    y = Batch(
        surf_vars={k: surf_tgt(v) for k, v in surf_src.items()},
        static_vars=static_vars,
        atmos_vars={k: atmos_tgt(v) for k, v in atmos_src.items()},
        metadata=md_y,
    )
    return x, y


def run_inference_era5(
    *,
    era5_zarr_path: str,
    era5_static_nc: str,
    device: str,
    time_index: int = 10,
    lead_hours: int = 6,
    crop_lat: int = 128,
    crop_lon: int = 256,
    rollout_steps: Optional[int] = None,
    use_lora: bool = False,
    lora_mode: Literal["single", "from_second", "all"] = "single",
    lora_steps: int = 40,
    stabilise_level_agg: bool = False,
    extra_var: Optional[ExtraVar] = None,
):
    """Inference on ERA5 data (one-step or autoregressive rollout)."""
    model = build_model(
        use_lora=use_lora,
        lora_mode=lora_mode,
        lora_steps=lora_steps,
        autocast=False,
        stabilise_level_agg=stabilise_level_agg,
        extra_var=extra_var,
    ).to(device).eval()

    x, _ = make_era5_short_lead_pair(
        era5_zarr_path=era5_zarr_path,
        era5_static_nc=era5_static_nc,
        time_index=time_index,
        lead_hours=lead_hours,
        crop_lat=crop_lat,
        crop_lon=crop_lon,
        device=device,
        extra_var=extra_var,
    )

    with torch.inference_mode():
        if rollout_steps is None:
            return model(x).to("cpu")
        preds = [p.to("cpu") for p in rollout(model, x, steps=rollout_steps)]
        return preds


def run_finetuning_era5_short_lead(
    *,
    steps: int,
    device: str,
    era5_zarr_path: str,
    era5_static_nc: str,
    time_index: int = 10,
    lead_hours: int = 6,
    crop_lat: int = 128,
    crop_lon: int = 256,
    lr: float = 3e-5,
    use_lora: bool = False,
    train_lora_only: bool = True,
    lora_mode: Literal["single", "from_second", "all"] = "single",
    lora_steps: int = 40,
    stabilise_level_agg: bool = False,
    autocast: bool = True,
    extra_var: Optional[ExtraVar] = None,
):
    """Short-lead supervised fine-tuning on one ERA5 sample."""
    model = build_model(
        use_lora=use_lora,
        lora_mode=lora_mode,
        lora_steps=lora_steps,
        autocast=autocast,
        stabilise_level_agg=stabilise_level_agg,
        extra_var=extra_var,
    ).to(device).train()

    if use_lora and train_lora_only:
        freeze_to_lora_only(model)

    # Optional: activation checkpointing if available
    if hasattr(model, "configure_activation_checkpointing"):
        model.configure_activation_checkpointing()

    x, y = make_era5_short_lead_pair(
        era5_zarr_path=era5_zarr_path,
        era5_static_nc=era5_static_nc,
        time_index=time_index,
        lead_hours=lead_hours,
        crop_lat=crop_lat,
        crop_lon=crop_lon,
        device=device,
        extra_var=extra_var,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)

    last_pred = None
    loss_history: list[float] = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss_value = supervised_mae(pred, y)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        last_pred = pred
        last_loss = float(loss_value.detach().cpu().item())
        loss_history.append(last_loss)
        print(f"[era5-short][step {step}] loss = {last_loss:.4f}")

    return model, last_pred, loss_history


def _make_target_at_time_index(
    ds,
    static_vars: dict[str, torch.Tensor],
    atmos_levels: tuple[int, ...],
    idx: int,
    device: str,
    surf_src: dict[str, str],
    atmos_src: dict[str, str],
    time_value,
) -> Batch:
    """Create a t=1 target Batch at ds.time[idx]."""
    def surf_tgt(name):
        arr = ds[name].isel(time=slice(idx, idx + 1)).astype("float32").values
        return torch.from_numpy(arr)[None].to(device)

    def atmos_tgt(name):
        arr = ds[name].isel(time=slice(idx, idx + 1)).astype("float32").values
        return torch.from_numpy(arr)[None].to(device)

    md = Metadata(
        lat=torch.from_numpy(ds.latitude.values.astype("float32")).to(device),
        lon=torch.from_numpy(ds.longitude.values.astype("float32")).to(device),
        time=(time_value,),
        atmos_levels=atmos_levels,
    )
    return Batch(
        surf_vars={k: surf_tgt(v) for k, v in surf_src.items()},
        static_vars=static_vars,
        atmos_vars={k: atmos_tgt(v) for k, v in atmos_src.items()},
        metadata=md,
    )


def run_finetuning_era5_rollout(
    *,
    steps: int,
    device: str,
    era5_zarr_path: str,
    era5_static_nc: str,
    time_index: int = 10,
    lead_hours: int = 6,
    rollout_horizon: int = 8,
    rollout_loss_on: Literal["last", "sum"] = "last",
    crop_lat: int = 128,
    crop_lon: int = 256,
    lr: float = 3e-5,
    use_lora: bool = True,
    train_lora_only: bool = True,
    lora_mode: Literal["single", "from_second", "all"] = "all",
    lora_steps: int = 40,
    stabilise_level_agg: bool = False,
    autocast: bool = True,
    extra_var: Optional[ExtraVar] = None,
):
    """Rollout (autoregressive) fine-tuning on one ERA5 sequence.

    rollout_loss_on:
      - "sum": loss at every step (heavier as builds a bigger graph)
      - "last": run first K-1 steps without grads i-e backprop only through final step (lighter)
    """
    np = _lazy_import_numpy()

    model = build_model(
        use_lora=use_lora,
        lora_mode=lora_mode,
        lora_steps=lora_steps,
        autocast=autocast,
        stabilise_level_agg=stabilise_level_agg,
        extra_var=extra_var,
    ).to(device).train()

    if use_lora and train_lora_only:
        freeze_to_lora_only(model)

    if hasattr(model, "configure_activation_checkpointing"):
        model.configure_activation_checkpointing()

    # Prepare data
    ds = _open_era5_dynamic_zarr(era5_zarr_path, crop_lat=crop_lat, crop_lon=crop_lon)
    static_ds = _open_static_nc(era5_static_nc, ds.latitude, ds.longitude)
    stat = _get_static_arrays(static_ds)

    dt_hours = _infer_time_step_hours(ds)
    step = _lead_to_index_step(dt_hours, lead_hours)

    idx = max(step, min(time_index, ds.sizes["time"] - (rollout_horizon * step) - 1))
    hist = [idx - step, idx]

    # mappings including extra var
    surf_src = dict(ERA5_SURF_SRC)
    atmos_src = dict(ERA5_ATMOS_SRC)
    static_vars = {
        "lsm": stat["lsm"].to(device),
        "slt": stat["slt"].to(device),
        "z": stat["z"].to(device),
    }

    if extra_var is not None:
        if extra_var.kind == "surf":
            surf_src[extra_var.key] = extra_var.src
        elif extra_var.kind == "atmos":
            atmos_src[extra_var.key] = extra_var.src
        elif extra_var.kind == "static":
            da = static_ds[extra_var.src] if extra_var.src in static_ds.data_vars else static_ds[extra_var.key]
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            static_vars[extra_var.key] = torch.from_numpy(da.astype("float32").values).to(device)

    def surf_hist(name):
        arr = ds[name].isel(time=hist).astype("float32").values
        return torch.from_numpy(arr)[None].to(device)

    def atmos_hist(name):
        arr = ds[name].isel(time=hist).astype("float32").values
        return torch.from_numpy(arr)[None].to(device)

    current_time = ds.time.values[idx].astype("datetime64[s]").tolist()
    levels = tuple(int(x) for x in ds.level.values) if "level" in ds.coords else AURORA_LEVELS

    x0 = Batch(
        surf_vars={k: surf_hist(v) for k, v in surf_src.items()},
        static_vars=static_vars,
        atmos_vars={k: atmos_hist(v) for k, v in atmos_src.items()},
        metadata=Metadata(
            lat=torch.from_numpy(ds.latitude.values.astype("float32")).to(device),
            lon=torch.from_numpy(ds.longitude.values.astype("float32")).to(device),
            time=(current_time,),
            atmos_levels=levels,
        ),
    )

    # Build targets for each rollout step
    target_batches: list[Batch] = []
    for s in range(1, rollout_horizon + 1):
        tidx = idx + s * step
        tval = ds.time.values[tidx].astype("datetime64[s]").tolist()
        target_batches.append(
            _make_target_at_time_index(
                ds, static_vars, levels, tidx, device, surf_src, atmos_src, tval
            )
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)

    last_pred = None
    loss_history: list[float] = []

    for it in range(steps):
        optimizer.zero_grad(set_to_none=True)

        if rollout_loss_on == "sum":
            batch_state = x0
            total_loss = torch.tensor(0.0, device=device)

            # Similar to aurora.rollout source, but we keep the graph.
            for step_i in range(rollout_horizon):
                pred = model.forward(batch_state)
                total_loss = total_loss + supervised_mae(pred, target_batches[step_i])
                last_pred = pred

                # Update history for next step (same logic as aurora.rollout)
                batch_state = dataclasses.replace(
                    pred,
                    surf_vars={
                        k: torch.cat([batch_state.surf_vars[k][:, 1:], v], dim=1)
                        for k, v in pred.surf_vars.items()
                    },
                    atmos_vars={
                        k: torch.cat([batch_state.atmos_vars[k][:, 1:], v], dim=1)
                        for k, v in pred.atmos_vars.items()
                    },
                )

            loss_value = total_loss

        elif rollout_loss_on == "last":
            # Run K-1 steps without grads to get the last state.
            batch_state = x0
            with torch.no_grad():
                for step_i in range(rollout_horizon - 1):
                    pred_ng = model.forward(batch_state)
                    batch_state = dataclasses.replace(
                        pred_ng,
                        surf_vars={
                            k: torch.cat([batch_state.surf_vars[k][:, 1:], v], dim=1)
                            for k, v in pred_ng.surf_vars.items()
                        },
                        atmos_vars={
                            k: torch.cat([batch_state.atmos_vars[k][:, 1:], v], dim=1)
                            for k, v in pred_ng.atmos_vars.items()
                        },
                    )

            # Final step WITH grads
            pred = model.forward(batch_state)
            last_pred = pred
            loss_value = supervised_mae(pred, target_batches[-1])

        else:
            raise ValueError(f"Unknown rollout_loss_on={rollout_loss_on}")

        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        last_loss = float(loss_value.detach().cpu().item())
        loss_history.append(last_loss)
        print(f"[era5-rollout][iter {it}] loss = {last_loss:.4f}")

    return model, last_pred, loss_history