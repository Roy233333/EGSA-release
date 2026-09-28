"""Correct and efficient sparse EMA updates for adaptive pixel hardness maps."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseEMAResult:
    """Unique updated coordinates and their old/new values."""

    y: torch.Tensor
    x: torch.Tensor
    old: torch.Tensor
    new: torch.Tensor

    @property
    def delta(self) -> torch.Tensor:
        return self.new - self.old


@torch.no_grad()
def update_sparse_ema_map(
    hardness: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    observations: torch.Tensor,
    *,
    decay: float,
    scale: float = 1.0,
    enabled: bool = True,
) -> SparseEMAResult:
    """Update visited pixels with exact sequential EMA semantics.

    The common case, in which every sampled coordinate is unique, uses a
    direct vectorised update. Repeated coordinates use an exact grouped path
    that preserves the order of their observations. For observations
    ``l_1, ..., l_k`` of one pixel, the result is exactly:

    ``h_k = decay**k * h_0 + (1-decay) * sum(decay**(k-j) * scale*l_j)``.

    ``enabled=False`` is the explicit no-hardness ablation. ``decay=0`` remains
    a valid EMA setting and means "replace by the latest observation".
    """

    if hardness.ndim != 2:
        raise ValueError("hardness must be a 2D map")
    if not 0.0 <= float(decay) <= 1.0:
        raise ValueError("decay must lie in [0, 1]")

    y = y.reshape(-1).long().detach().cpu()
    x = x.reshape(-1).long().detach().cpu()
    values = observations.reshape(-1).float().detach().cpu()
    if not (y.numel() == x.numel() == values.numel()):
        raise ValueError("y, x, and observations must have equal lengths")

    empty_long = torch.empty(0, dtype=torch.long)
    empty_value = torch.empty(0, dtype=hardness.dtype)
    if y.numel() == 0 or not enabled:
        return SparseEMAResult(empty_long, empty_long.clone(), empty_value, empty_value.clone())

    height, width = hardness.shape
    if torch.any(y < 0) or torch.any(y >= height) or torch.any(x < 0) or torch.any(x >= width):
        raise IndexError("EMA coordinates are outside the hardness map")

    linear = y * width + x
    order = torch.argsort(linear)
    linear_sorted = linear[order]

    # Most ray batches contain no repeated pixel coordinates. Avoid the more
    # expensive grouped-power/scatter calculation in that common case while
    # preserving exactly the same one-observation EMA arithmetic.
    if linear_sorted.numel() < 2 or not torch.any(linear_sorted[1:] == linear_sorted[:-1]):
        unique_y = y[order]
        unique_x = x[order]
        old = hardness[unique_y, unique_x].clone()
        new = float(decay) * old
        new = new + (1.0 - float(decay)) * (values[order] * float(scale)).to(old)
        hardness[unique_y, unique_x] = new
        return SparseEMAResult(unique_y, unique_x, old, new)

    # PyTorch 1.12 does not expose ``stable=True`` for argsort. Sorting a
    # compound (pixel, original-position) key preserves observation order for
    # repeated pixels on both old and new PyTorch versions.
    original_position = torch.arange(linear.numel(), dtype=torch.long)
    sort_key = linear * (linear.numel() + 1) + original_position
    order = torch.argsort(sort_key)
    linear_sorted = linear[order]
    values_sorted = values[order] * float(scale)

    unique_linear, counts = torch.unique_consecutive(linear_sorted, return_counts=True)
    group_starts = torch.cumsum(counts, dim=0) - counts
    repeated_starts = torch.repeat_interleave(group_starts, counts)
    within_group = torch.arange(linear.numel(), dtype=torch.long) - repeated_starts
    repeated_counts = torch.repeat_interleave(counts, counts)
    exponent = repeated_counts - 1 - within_group

    decay_value = float(decay)
    weights = torch.pow(torch.full_like(values_sorted, decay_value), exponent)
    group_ids = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    weighted_observations = torch.zeros(counts.numel(), dtype=values_sorted.dtype)
    weighted_observations.scatter_add_(0, group_ids, values_sorted * weights)

    unique_y = torch.div(unique_linear, width, rounding_mode="floor")
    unique_x = unique_linear.remainder(width)
    old = hardness[unique_y, unique_x].clone()
    new = torch.pow(torch.full_like(old, decay_value), counts) * old
    new = new + (1.0 - decay_value) * weighted_observations.to(old)
    hardness[unique_y, unique_x] = new

    return SparseEMAResult(unique_y, unique_x, old, new)


@torch.no_grad()
def update_sampler_image_state(
    *,
    hardness: torch.Tensor,
    prior: torch.Tensor,
    probability: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    per_ray_loss: torch.Tensor,
    decay: float,
    scale: float = 1.0,
    enabled: bool = True,
) -> SparseEMAResult:
    """Update hardness and keep the pixel probability map exactly consistent."""

    result = update_sparse_ema_map(
        hardness,
        y,
        x,
        per_ray_loss,
        decay=decay,
        scale=scale,
        enabled=enabled,
    )
    if result.y.numel() == 0:
        return result

    probability[result.y, result.x] = prior[result.y, result.x] * result.new
    return result
