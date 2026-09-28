"""Numerical checks for the integrated sparse EMA implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "nerfstudio"
    / "data"
    / "ema_hardness.py"
)
SPEC = importlib.util.spec_from_file_location("egsa_ema_hardness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EMA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EMA
SPEC.loader.exec_module(EMA)


def sequential_reference(hardness, y, x, observations, decay, scale=1.0):
    result = hardness.clone()
    for yy, xx, value in zip(y.tolist(), x.tolist(), observations.tolist()):
        result[yy, xx] = decay * result[yy, xx] + (1.0 - decay) * scale * value
    return result


def test_sparse_ema_matches_sequential_reference():
    cases = (
        (
            torch.tensor([0, 0, 1, 2]),
            torch.tensor([0, 2, 1, 3]),
        ),
        (
            torch.tensor([0, 1, 0, 2, 0, 1]),
            torch.tensor([2, 1, 2, 3, 2, 1]),
        ),
    )
    observations = torch.tensor([0.2, 0.8, 0.4, 1.2, 0.6, 0.1])

    for y, x in cases:
        values = observations[: y.numel()]
        for decay in (0.0, 0.5, 0.9, 1.0):
            for scale in (0.5, 1.0, 1.7):
                initial = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
                expected = sequential_reference(initial, y, x, values, decay, scale)
                actual = initial.clone()
                EMA.update_sparse_ema_map(
                    actual,
                    y,
                    x,
                    values,
                    decay=decay,
                    scale=scale,
                )
                torch.testing.assert_close(actual, expected)


def test_probability_map_stays_consistent():
    hardness = torch.ones((3, 4), dtype=torch.float32)
    prior = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    probability = prior * hardness
    y = torch.tensor([0, 2, 0, 1])
    x = torch.tensor([1, 3, 1, 0])
    observations = torch.tensor([0.2, 0.9, 0.6, 0.3])

    EMA.update_sampler_image_state(
        hardness=hardness,
        prior=prior,
        probability=probability,
        y=y,
        x=x,
        per_ray_loss=observations,
        decay=0.9,
    )
    torch.testing.assert_close(probability, prior * hardness)


def test_disabled_update_is_noop():
    hardness = torch.ones((2, 2), dtype=torch.float32)
    before = hardness.clone()
    result = EMA.update_sparse_ema_map(
        hardness,
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([3.0]),
        decay=0.9,
        enabled=False,
    )
    torch.testing.assert_close(hardness, before)
    assert result.y.numel() == 0


if __name__ == "__main__":
    test_sparse_ema_matches_sequential_reference()
    test_probability_map_stays_consistent()
    test_disabled_update_is_noop()
    print("EMA numerical checks passed")
