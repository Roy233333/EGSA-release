"""Utilities for closing SDFStudio's adaptive-sampling feedback loop.

This module extracts detached per-ray errors and submits them to the sampler.
It is independent of SDFStudio imports so that its numerical behavior can be
tested separately.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch


DEFAULT_PREDICTION_KEYS: tuple[str, ...] = (
    "rgb",
    "rgb_fine",
    "rgb_coarse",
)


def _first_tensor(mapping: Optional[Mapping[str, Any]], keys: Sequence[str]) -> Optional[torch.Tensor]:
    if mapping is None:
        return None
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def extract_per_ray_l2(
    model_outputs: Optional[Mapping[str, Any]],
    batch: Mapping[str, Any],
    loss_dict: Optional[Mapping[str, Any]] = None,
    *,
    explicit_key: str = "per_ray_l2_vec",
    target_key: str = "image",
    prediction_keys: Sequence[str] = DEFAULT_PREDICTION_KEYS,
) -> Optional[torch.Tensor]:
    """Return one detached RGB mean-squared error value per sampled ray.

    An explicitly supplied per-ray vector takes precedence.  Otherwise the
    vector is derived from rendered RGB and the sampled ground-truth pixels.
    The returned tensor is detached because it updates sampler state and must
    not create a second autograd path.

    ``None`` is returned when the required tensors are unavailable or their
    shapes are incompatible.  This makes the helper safe across SDFStudio
    model variants without silently broadcasting mismatched tensors.
    """

    explicit = _first_tensor(loss_dict, (explicit_key,))
    if explicit is None:
        explicit = _first_tensor(model_outputs, (explicit_key,))
    if explicit is not None:
        return explicit.reshape(-1).detach()

    prediction = _first_tensor(model_outputs, prediction_keys)
    target = batch.get(target_key)
    if prediction is None or not isinstance(target, torch.Tensor):
        return None

    # Some datasets retain an alpha channel in the sampled target.
    if target.shape[-1:] == (4,) and prediction.shape[-1:] == (3,):
        target = target[..., :3]

    if prediction.shape != target.shape or prediction.ndim < 2:
        return None

    per_ray = (prediction.float() - target.to(prediction).float()).square().mean(dim=-1)
    return per_ray.reshape(-1).detach()


@torch.no_grad()
def submit_adaptive_feedback(
    datamanager: Any,
    batch: Mapping[str, Any],
    model_outputs: Optional[Mapping[str, Any]],
    loss_dict: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Compute and submit per-ray hardness feedback to a compatible manager.

    Returns ``True`` only when feedback was actually submitted.  Exceptions
    raised by the datamanager are intentionally not swallowed here: a broken
    feedback path should be visible rather than silently
    disabling EMA updates.
    """

    update = getattr(datamanager, "update_from_per_ray_loss", None)
    if not callable(update):
        return False

    # Avoid even computing the auxiliary RGB error for non-adaptive samplers.
    if hasattr(datamanager, "train_pixel_sampler"):
        sampler = datamanager.train_pixel_sampler
        if sampler is None or not callable(getattr(sampler, "update", None)):
            return False
        # Avoid the rendered-RGB reduction and GPU-to-CPU transfer entirely
        # for the explicit no-hardness ablation.
        if not getattr(sampler, "hardness_enabled", True):
            return False

    per_ray = extract_per_ray_l2(model_outputs, batch, loss_dict)
    if per_ray is None:
        return False

    indices = batch.get("indices")
    if not isinstance(indices, torch.Tensor) or indices.shape[0] != per_ray.numel():
        return False

    update(batch, {"per_ray_l2_vec": per_ray}, key="per_ray_l2_vec")
    return True
