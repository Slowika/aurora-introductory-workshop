"""Test Pydantic configuration models and their custom validators."""

import warnings
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from setup.components.common.models import (
    BaseConfig,
    DataMode,
    ExtraVariable,
    ExtraVariables,
    FinetuneConfig,
)
from setup.components.common.utils import load_batch_from_asset, make_lowres_batch


def test_extra_variables_iteration() -> None:
    """Yield (long name, ExtraVariable) pairs in insertion order."""
    variables={
        "total_precipitation": ExtraVariable(kind="surf_vars", key="tp"),
        "sea_surface_temp": ExtraVariable(kind="surf_vars", key="sst"),
    }
    evs = ExtraVariables(variables=variables)
    pairs = list(evs)
    assert len(pairs) == len(variables)
    for i, (k, v) in enumerate(variables.items()):
        assert pairs[i][0] == k
        assert pairs[i][1] == v


def test_extra_variables_duplicate_short_keys_same_kind_rejected() -> None:
    """Reject duplicate short keys within the same variable category."""
    kind = "surf_vars"
    dupe_key = "dupe"
    with pytest.raises(
        ValidationError,
        match=f"Duplicate extra variable short keys in {kind}",
    ):
        ExtraVariables(
            variables={
                "var_a": ExtraVariable(kind=kind, key=dupe_key),
                "var_b": ExtraVariable(kind=kind, key=dupe_key),
            },
        )


def test_extra_variables_same_key_different_kind_allowed() -> None:
    """Allow the same short key across different variable categories."""
    key = "z"
    variables={
        "geopotential_static": ExtraVariable(kind="static_vars", key=key),
        "geopotential_atmos": ExtraVariable(kind="atmos_vars", key=key),
    }
    evs = ExtraVariables(variables=variables)
    assert len(evs) == len(variables)


def test_data_mode_batch_fn_test() -> None:
    """Map TEST mode to make_lowres_batch."""
    assert DataMode.TEST.batch_fn is make_lowres_batch


def test_data_mode_batch_fn_era5() -> None:
    """Map ERA5 mode to load_batch_from_asset."""
    assert DataMode.ERA5.batch_fn is load_batch_from_asset


def test_base_config_variable_map_returns_copy() -> None:
    """Return a defensive copy so mutations don't affect internal state."""
    cfg = BaseConfig(mode=DataMode.TEST)
    vm = cfg.variable_map
    kind = "surf_vars"
    longname = "new_longname"
    vm[kind][longname] = "new_short"
    assert longname not in cfg.variable_map[kind]


def test_base_config_extra_variables_extend_map() -> None:
    """Append extra variables to the corresponding category map."""
    longname = "total_precipitation"
    kind = "surf_vars"
    key = "tp"
    extra = ExtraVariables(variables={longname: ExtraVariable(kind=kind, key=key)})
    cfg = BaseConfig(mode=DataMode.TEST, extra_variables=extra)
    assert longname in cfg.variable_map[kind]
    assert cfg.variable_map[kind][longname] == key


@patch("setup.components.common.models.normalisation")
def test_base_config_extra_variables_register_normalisation(
    mock_norm: MagicMock,
) -> None:
    """Register location and scale parameters in the normalisation module."""
    mock_norm.locations = {}
    mock_norm.scales = {}
    key = "tp"
    loc = 1.5
    scale = 0.5
    extra = ExtraVariables(
        variables={
            "total_precipitation": ExtraVariable(
                kind="surf_vars",
                key=key,
                location=loc,
                scale=scale,
            ),
        },
    )
    BaseConfig(mode=DataMode.TEST, extra_variables=extra)
    assert mock_norm.locations[key] == loc
    assert mock_norm.scales[key] == scale


def test_base_config_aurora_init_kwargs_merges_var_cfg() -> None:
    """Merge extra variable tuples into Aurora constructor kwargs."""
    kind = "surf_vars"
    key = "tp"
    extra = ExtraVariables(
        variables={"total_precipitation": ExtraVariable(kind=kind, key=key)},
    )
    cfg = BaseConfig(mode=DataMode.TEST, extra_variables=extra)
    assert key in cfg.aurora_init_kwargs[kind]


def test_finetune_config_rollout_requires_rollout_steps() -> None:
    """Require rollout_steps when type is 'rollout'."""
    with pytest.raises(ValidationError, match="rollout_steps is required"):
        FinetuneConfig(mode=DataMode.TEST, type="rollout", epochs=5)


def test_finetune_config_short_with_rollout_steps_warns() -> None:
    """Warn when rollout_steps is provided but type is not 'rollout'."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        FinetuneConfig(mode=DataMode.TEST, type="short", epochs=5, rollout_steps=3)
    assert len(w) == 1
    assert "will be ignored" in str(w[0].message)
