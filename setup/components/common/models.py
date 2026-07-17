"""Pydantic models for inference and fine-tuning job configurations.

Located here for better availability across notebooks and components. When placed in
setup/common, remote component import paths become challenging to resolve.
"""

import warnings
from collections.abc import Callable, Generator
from enum import StrEnum
from typing import Any, Literal, Self

from aurora import Batch, normalisation
from aurora.model.lora import LoRAMode
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .constants import (
    ATMOS_VAR_MAP,
    STATIC_VAR_MAP,
    SURF_VAR_MAP,
)
from .utils import load_batch_from_asset, make_lowres_batch


class AuroraConfig(BaseModel):
    """Configuration parameters for the Aurora model.

    Can be extended to include any Aurora model constructor parameters, alternatively
    from the `Aurora` object's signature itself.
    """

    model_config = ConfigDict(extra="forbid")

    use_lora: bool = False
    lora_steps: int = 40
    lora_mode: LoRAMode = "single"
    autocast: bool = False


class ExtraVariable(BaseModel):
    """Definition of an extra, fine-tuned variable to include."""

    kind: Literal["surf_vars", "static_vars", "atmos_vars"]
    """Type of variable per Aurora's variable types."""
    key: str
    """Model-expected variable short name."""
    location: float = 0.0
    """Location parameter for normalisation."""
    scale: float = 1.0
    """Scale parameter for normalisation."""


class ExtraVariables(BaseModel):
    """Collection of extra, fine-tuned variables with uniqueness validation."""

    model_config = ConfigDict(extra="forbid")

    variables: dict[str, ExtraVariable] = {}
    """Mapping of variable long names to their definitions."""

    @model_validator(mode="after")
    def _check_unique_keys(self) -> Self:
        """Ensure no duplicate short keys within the same variable type.

        In keeping with the Aurora API, permits duplicate short keys across variable
        types (e.g. atmos_vars and static_vars "z" for geopotential).
        """
        by_kind: dict[str, list[str]] = {}
        for ev in self.variables.values():
            by_kind.setdefault(ev.kind, []).append(ev.key)
        for kind, keys in by_kind.items():
            if len(keys) != len(set(keys)):
                duplicates = {k for k in keys if keys.count(k) > 1}
                msg = f"Duplicate extra variable short keys in {kind}: {duplicates}"
                raise ValueError(msg)
        return self

    def __iter__(self) -> Generator[tuple[str, ExtraVariable]]:
        """Iterate over (long name, ExtraVariable) pairs."""
        yield from self.variables.items()

    def __len__(self) -> int:
        """Return the number of extra variables defined."""
        return len(self.variables)


class DataMode(StrEnum):
    """Data mode for aurora.Batch creation."""

    TEST = "test"
    ERA5 = "era5"

    @property
    def batch_fn(self) -> Callable[..., Batch]:
        """Return the aurora.Batch creation function for a mode.

        Returns
        -------
        Callable[..., aurora.Batch]
            Batch creation function corresponding to the mode.

        """
        return {"test": make_lowres_batch, "era5": load_batch_from_asset}[self.value]


class BaseConfig(BaseModel):
    """Base configuration for inference and fine-tuning jobs."""

    mode: DataMode
    """Data mode to use for aurora.Batch creation."""
    aurora_config: AuroraConfig = AuroraConfig()
    """Dictionary of Aurora model constructor parameters."""
    extra_variables: ExtraVariables | None = None
    """Additional fine-tuned variables to include, if any.

    Should not include the standard pre-trained variables unless modifying normalisation
    parameters.
    """

    # computed from extra_variables, excluded from serialisation
    _var_map: dict[str, dict[str, str]] = PrivateAttr(default_factory=dict)
    _var_cfg: dict[str, tuple[str, ...]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _: None) -> None:
        """Compute variable mappings and config overrides from extra_variables."""
        var_map = {
            "surf_vars": SURF_VAR_MAP.copy(),
            "static_vars": STATIC_VAR_MAP.copy(),
            "atmos_vars": ATMOS_VAR_MAP.copy(),
        }
        var_cfg: dict[str, tuple[str, ...]] = {}
        if self.extra_variables is not None:
            for longname, ev in self.extra_variables:
                type_var_map = var_map[ev.kind]
                type_var_map[longname] = ev.key
                var_cfg[ev.kind] = tuple(type_var_map.values())
                normalisation.locations[ev.key] = ev.location
                normalisation.scales[ev.key] = ev.scale
        self._var_map = var_map
        self._var_cfg = var_cfg

    @property
    def variable_map(self) -> dict[str, dict[str, str]]:
        """Mapping of variable long names to short names for each variable type.

        Returns
        -------
        dict[str, dict[str, str]]
            Mapping of variable long names to short names for each variable type,
            including standard pre-trained and extra fine-tuned variables, if any.

        """
        return {k: v.copy() for k, v in self._var_map.items()}

    @property
    def aurora_init_kwargs(self) -> dict[str, Any]:
        """Aurora constructor kwargs with extra variable tuples merged in.

        Returns
        -------
        dict[str, Any]
            Aurora constructor kwargs with extra fine-tuned variable tuples merged in.

        """
        return self.aurora_config.model_dump() | self._var_cfg


class InferenceConfig(BaseConfig):
    """Configuration for inference jobs."""

    steps: int = Field(ge=1)
    """Number of autoregressive inference steps to perform."""


class FinetuneConfig(BaseConfig):
    """Configuration for fine-tuning jobs."""

    type: Literal["short", "rollout"]
    """Type of fine-tuning to perform."""
    epochs: int = Field(ge=1)
    """Number of training epochs."""
    learning_rate: float = 3e-5
    """Learning rate for the optimiser."""
    rollout_steps: int | None = Field(default=None, ge=1)
    """Number of autoregressive steps for rollout fine-tuning."""
    area_weighted: bool = False
    """Whether the MAE loss is area-weighted."""
    batches_per_epoch: int | Literal["all"] = 1
    """How many batches are sampled in each epoch (integer or 'all')."""

    @model_validator(mode="after")
    def _validate_rollout_steps(self) -> Self:
        """Validate rollout_steps against fine-tuning type."""
        if self.type == "rollout" and self.rollout_steps is None:
            msg = "rollout_steps is required when type='rollout'."
            raise ValueError(msg)
        if self.type != "rollout" and self.rollout_steps is not None:
            warnings.warn(
                f"rollout_steps={self.rollout_steps} will be ignored for "
                f"type='{self.type}' fine-tuning.",
                stacklevel=2,
            )
        return self
