"""Unit tests for fine-tuning utilities.

Covers only functionality not exercised by component integration tests and not worth
additional permutations or fixtures.
"""

from datetime import UTC, datetime, timedelta

import pytest
import torch

from setup.components.finetuning.utils import get_datetime_range, get_lora_params


def test_get_datetime_range_raises() -> None:
    """Test get_datetime_range raises for insufficient timestamps."""
    with pytest.raises(ValueError, match="Less than two timestamps generated"):
        get_datetime_range(
            start=datetime(2025, 1, 1, 0, tzinfo=UTC),
            end=datetime(2025, 1, 1, 1, tzinfo=UTC),
            step=timedelta(hours=6),
        )


def test_get_lora_params_raises_no_lora() -> None:
    """Test get_lora_params raises when model has no LoRA parameters."""
    model = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="No trainable parameters found"):
        get_lora_params(model)
