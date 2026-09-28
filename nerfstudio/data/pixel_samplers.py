# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Uniform, equirectangular and edge-guided adaptive pixel samplers.

- PixelSampler: uniform per-pixel sampling
- EquirectangularPixelSampler: uniform on sphere
- AdaptivePixelSampler: edge prior + EMA hardness + uniform mix with true blockwise multinomial

"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from nerfstudio.data.ema_hardness import update_sampler_image_state
from nerfstudio.data.edge_priors import compute_edge_map_cpu
from nerfstudio.utils.images import BasicImages

# =========================================================
# Collate helpers (kept from upstream behavior)
# =========================================================


def collate_image_dataset_batch(
    batch: Dict[str, Any],
    num_rays_per_batch: int,
    keep_full_image: bool = False,
) -> Dict[str, Any]:
    """Operate on a batch of images and sample pixels uniformly."""
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape

    if "mask" in batch:
        nonzero_indices = torch.nonzero(batch["mask"][..., 0].to(device), as_tuple=False)
        chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_per_batch)
        indices = nonzero_indices[chosen_indices]
    else:
        indices = torch.floor(
            torch.rand((num_rays_per_batch, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

    collated_batch: Dict[str, Any] = {
        key: value[c, y, x]
        for key, value in batch.items()
        if key not in ("image_idx", "src_imgs", "src_idxs", "sparse_sfm_points") and value is not None
    }

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    if "sparse_sfm_points" in batch:
        collated_batch["sparse_sfm_points"] = batch["sparse_sfm_points"].images[c[0]]

    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


def collate_image_dataset_batch_list(
    batch: Dict[str, Any],
    num_rays_per_batch: int,
    keep_full_image: bool = False,
) -> Dict[str, Any]:
    """Same as collate_image_dataset_batch, but input tensors may be ragged (lists of images/masks)."""
    device = batch["image"][0].device
    num_images = len(batch["image"])

    all_indices: List[torch.Tensor] = []
    all_images: List[torch.Tensor] = []
    all_fg_masks: List[torch.Tensor] = []

    if "mask" in batch:
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
            nonzero_indices = batch["mask"][i]
            chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_in_batch)
            indices = nonzero_indices[chosen_indices]
            indices = torch.cat([torch.full((num_rays_in_batch, 1), i, device=device), indices], dim=-1)
            all_indices.append(indices)
            all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
            if "fg_mask" in batch:
                all_fg_masks.append(batch["fg_mask"][i][indices[:, 1], indices[:, 2]])
    else:
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            H, W, _ = batch["image"][i].shape
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
            indices = torch.floor(
                torch.rand((num_rays_in_batch, 3), device=device) * torch.tensor([1, H, W], device=device)
            ).long()
            indices[:, 0] = i
            all_indices.append(indices)
            all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
            if "fg_mask" in batch:
                all_fg_masks.append(batch["fg_mask"][i][indices[:, 1], indices[:, 2]])

    indices = torch.cat(all_indices, dim=0)
    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

    collated_batch: Dict[str, Any] = {
        key: value[c, y, x]
        for key, value in batch.items()
        if key not in ("image_idx", "image", "mask", "fg_mask", "sparse_pts") and value is not None
    }

    collated_batch["image"] = torch.cat(all_images, dim=0)

    if len(all_fg_masks) > 0:
        collated_batch["fg_mask"] = torch.cat(all_fg_masks, dim=0)

    if "sparse_pts" in batch:
        rand_idx = random.randint(0, num_images - 1)
        collated_batch["sparse_pts"] = batch["sparse_pts"][rand_idx]

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


def collate_image_dataset_batch_equirectangular(
    batch: Dict[str, Any],
    num_rays_per_batch: int,
    keep_full_image: bool = False,
) -> Dict[str, Any]:
    """Sample equirectangular images uniformly on the sphere."""
    device = batch["image"].device
    num_images, H, W, _ = batch["image"].shape

    if "mask" in batch:
        raise NotImplementedError("Masking not implemented for equirectangular images.")

    num_images_rand = torch.rand(num_rays_per_batch, device=device)
    phi_rand = torch.acos(1 - 2 * torch.rand(num_rays_per_batch, device=device)) / torch.pi
    theta_rand = torch.rand(num_rays_per_batch, device=device)

    indices = torch.floor(
        torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
        * torch.tensor([num_images, H, W], device=device)
    ).long()

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

    collated_batch = {key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None}
    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


# =========================================================
# Base samplers
# =========================================================


class PixelSampler:
    """Uniform pixel sampler."""

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False) -> None:
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image

    def set_num_rays_per_batch(self, num_rays_per_batch: int) -> None:
        self.num_rays_per_batch = num_rays_per_batch

    def sample(self, image_batch: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())
            pixel_batch = collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], BasicImages):
            image_batch = dict(image_batch.items())
            image_batch["image"] = image_batch["image"].images
            if "mask" in image_batch:
                image_batch["mask"] = image_batch["mask"].images
            if "fg_mask" in image_batch:
                image_batch["fg_mask"] = image_batch["fg_mask"].images
            if "sparse_pts" in image_batch:
                image_batch["sparse_pts"] = image_batch["sparse_pts"].images
            pixel_batch = collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch


class EquirectangularPixelSampler(PixelSampler):
    """Uniform sphere sampling for equirectangular images."""

    def sample(self, image_batch: Dict[str, Any]) -> Dict[str, Any]:
        return collate_image_dataset_batch_equirectangular(
            image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
        )


# =========================================================
# AdaptivePixelSampler: CPU sampling and blockwise allocation
# =========================================================


class AdaptivePixelSampler(PixelSampler):
    """
    Edge prior + EMA hardness + uniform mix.

    Key speedups:
    - CPU-side sampling (only N pixels copied to GPU).
    - True blockwise: sample cells first (q_cells), then sample inside the cell.
    - Incremental updates to q (per-pixel) and q_cells (per-cell) after EMA update.

    """

    def __init__(
        self,
        num_rays_per_batch: int,
        keep_full_image: bool = False,
        *,
        ema_decay: float = 0.9,
        hardness_enabled: bool = True,
        uniform_mix: float = 0.6,
        edge_weight: float = 1.3,
        edge_type: str = "sobel",  # "sobel" | "canny" (requires Kornia; no fallback)
        device: Union[str, torch.device] = "cuda",
        # blockwise
        block_size: int = 8,  # 0/1: pixel-level; >1: cell side length
        blockwise_exact: bool = False,  # True: multinomial in cell; False: uniform-in-cell (faster)
    ) -> None:
        super().__init__(num_rays_per_batch=num_rays_per_batch, keep_full_image=keep_full_image)

        self.device = torch.device(device)

        self.ema = float(ema_decay)
        if not 0.0 <= self.ema <= 1.0:
            raise ValueError("ema_decay must lie in [0, 1]")
        self.hardness_enabled = bool(hardness_enabled)
        self.uniform_mix = float(uniform_mix)
        self.edge_weight = float(edge_weight)
        self.edge_type = str(edge_type).lower()
        if self.edge_type not in ("sobel", "canny"):
            raise ValueError("edge_type must be 'sobel' or 'canny'")

        self.block_size = int(block_size) if block_size is not None else 0
        self.blockwise_exact = bool(blockwise_exact)

        # Per-image CPU caches
        self.mask_maps: Dict[int, torch.Tensor] = {}   # (H,W) float32 {0,1}
        self.edge_maps: Dict[int, torch.Tensor] = {}   # (H,W) [0,1]
        self.prior_maps: Dict[int, torch.Tensor] = {}  # (H,W)
        self.loss_weights: Dict[int, torch.Tensor] = {}  # (H,W)
        self.prob_maps: Dict[int, torch.Tensor] = {}   # (H,W)
        self.prob_sums: Dict[int, float] = {}          # scalar

        # Block level
        self.block_sums: Dict[int, torch.Tensor] = {}  # (Hc,Wc)
        self.block_shape: Dict[int, Tuple[int, int]] = {}
        self.shape_hw: Dict[int, Tuple[int, int]] = {}

        self.feedback_update_calls = 0

        msg = (
            f"[AdaptivePixelSampler] edge={self.edge_type}, edge_weight={self.edge_weight}, "
            f"uniform_mix={self.uniform_mix}, ema={self.ema}, "
            f"hardness_enabled={self.hardness_enabled}, block_size={self.block_size}, "
            f"blockwise_exact={self.blockwise_exact}"
        )
        print(msg)

    # ---------------- CPU-side state builders ----------------

    @torch.no_grad()
    def _ensure_state(self, gid: int, img_hwc_cpu: torch.Tensor, mask_hw_cpu: Optional[torch.Tensor]) -> None:
        if gid in self.prob_maps:
            return
        assert img_hwc_cpu.device.type == "cpu"
        H, W = img_hwc_cpu.shape[:2]
        self.shape_hw[gid] = (H, W)

        # mask
        if mask_hw_cpu is None:
            m = torch.ones((H, W), dtype=torch.float32)
        else:
            m = mask_hw_cpu
            if m.ndim == 3 and m.shape[-1] == 1:
                m = m[..., 0]
            m = (m > 0.5).to(torch.float32).cpu()
        self.mask_maps[gid] = m

        # edge & prior
        if self.edge_weight == 0.0:
            e = torch.ones((H, W), dtype=torch.float32)
        else:
            e = self._compute_edge_map_cpu(img_hwc_cpu)
        self.edge_maps[gid] = e

        prior = torch.pow(e.clamp_min(1e-6), self.edge_weight) * m
        self.prior_maps[gid] = prior

        # hardness & prob
        h = torch.ones_like(prior)
        q = prior * h
        Z = float(q.sum().item())

        self.loss_weights[gid] = h
        self.prob_maps[gid] = q
        self.prob_sums[gid] = max(Z, 1e-12)

        if self.block_size and self.block_size > 1:
            self._rebuild_block_sums(gid)

    @torch.no_grad()
    def _rebuild_block_sums(self, gid: int) -> None:
        s = self.block_size
        q = self.prob_maps[gid]  # (H,W)
        H, W = q.shape
        Hc = (H + s - 1) // s
        Wc = (W + s - 1) // s
        pad_h = Hc * s - H
        pad_w = Wc * s - W
        q_pad = F.pad(q, (0, pad_w, 0, pad_h))
        q_cells = q_pad.view(Hc, s, Wc, s).sum(3).sum(1).contiguous()
        self.block_sums[gid] = q_cells
        self.block_shape[gid] = (Hc, Wc)

    @torch.no_grad()
    def _update_block_sums_incremental(self, gid: int, ys: torch.Tensor, xs: torch.Tensor, dq: torch.Tensor) -> None:
        if not (self.block_size and self.block_size > 1):
            return
        s = self.block_size
        Hc, Wc = self.block_shape[gid]
        cy = torch.div(ys, s, rounding_mode="floor").clamp(0, Hc - 1)
        cx = torch.div(xs, s, rounding_mode="floor").clamp(0, Wc - 1)
        key = cy * Wc + cx
        uniq, inv = torch.unique(key, return_inverse=True)
        add = torch.zeros_like(uniq, dtype=dq.dtype)
        add.index_add_(0, inv, dq)
        cy_u = (uniq // Wc).long()
        cx_u = (uniq % Wc).long()
        self.block_sums[gid][cy_u, cx_u] += add

    @torch.no_grad()
    def _compute_edge_map_cpu(self, img_hwc: torch.Tensor) -> torch.Tensor:
        """Compute the selected Sobel or Canny-magnitude response."""
        return compute_edge_map_cpu(img_hwc, self.edge_type)

    @torch.no_grad()
    def _extract_mask_full(self, batch: Dict[str, Any], b: int) -> Optional[torch.Tensor]:
        key = "mask" if "mask" in batch else ("fg_mask" if "fg_mask" in batch else None)
        if key is None:
            return None
        m = batch[key]
        if hasattr(m, "images"):
            m = m.images
        return m[b].cpu()

    # ---------------- sampling (CPU) ----------------

    @torch.no_grad()
    def sample(self, image_batch: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize inputs to tensors on CPU
        if isinstance(image_batch["image"], list):
            return collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], BasicImages):
            image_batch = dict(image_batch.items())
            image_batch["image"] = image_batch["image"].images
            if "mask" in image_batch:
                image_batch["mask"] = image_batch["mask"].images
            if "fg_mask" in image_batch:
                image_batch["fg_mask"] = image_batch["fg_mask"].images
            if "sparse_pts" in image_batch:
                image_batch["sparse_pts"] = image_batch["sparse_pts"].images
        elif not isinstance(image_batch["image"], torch.Tensor):
            raise ValueError("image_batch['image'] must be a list, BasicImages, or torch.Tensor")

        imgs_cpu = image_batch["image"].cpu()
        B, _, _, _ = imgs_cpu.shape
        img_indices = image_batch["image_idx"].view(B).long().cpu()

        # 1) Build state & mixture masses
        masses: List[float] = []
        for b in range(B):
            gid = int(img_indices[b].item())
            mask_hw = self._extract_mask_full(image_batch, b)
            self._ensure_state(gid, imgs_cpu[b], mask_hw)
            masses.append(self.prob_sums[gid])
        masses_t = torch.tensor(masses, dtype=torch.float32)
        masses_t = masses_t / max(masses_t.sum().item(), 1e-12)
        if self.uniform_mix > 0:
            masses_t = (1 - self.uniform_mix) * masses_t + self.uniform_mix * (torch.ones_like(masses_t) / B)

        # 2) Allocations per image
        N = self.num_rays_per_batch
        img_choices = torch.multinomial(masses_t, N, replacement=True)
        counts = torch.bincount(img_choices, minlength=B)

        # 3) Per-image sampling
        sel_b, sel_y, sel_x = [], [], []
        for b in range(B):
            k = int(counts[b].item())
            if k == 0:
                continue
            gid = int(img_indices[b].item())
            H_, W_ = self.shape_hw[gid]

            if self.block_size and self.block_size > 1:
                q_cells = self.block_sums[gid]
                q_vec = q_cells.reshape(-1)
                Zc = float(q_vec.sum().item())
                cell_p = (torch.ones_like(q_vec) / q_vec.numel()) if Zc <= 0 else (q_vec / Zc)
                if self.uniform_mix > 0:
                    cell_p = (1 - self.uniform_mix) * cell_p + self.uniform_mix * (torch.ones_like(cell_p) / cell_p.numel())
                cell_p = cell_p / max(cell_p.sum().item(), 1e-12)

                cell_ids = torch.multinomial(cell_p.clamp_min(1e-12), k, replacement=True)
                Hc, Wc = q_cells.shape
                cy = torch.div(cell_ids, Wc, rounding_mode="floor").long()
                cx = (cell_ids % Wc).long()

                if not self.blockwise_exact:
                    ry = torch.randint(0, self.block_size, (k,))
                    rx = torch.randint(0, self.block_size, (k,))
                    ys = (cy * self.block_size + ry).clamp_max(H_ - 1)
                    xs = (cx * self.block_size + rx).clamp_max(W_ - 1)
                else:
                    ys = torch.empty(k, dtype=torch.long)
                    xs = torch.empty(k, dtype=torch.long)
                    q = self.prob_maps[gid]
                    for i in range(k):
                        y0 = int(cy[i].item()) * self.block_size
                        x0 = int(cx[i].item()) * self.block_size
                        y1 = min(y0 + self.block_size, H_)
                        x1 = min(x0 + self.block_size, W_)
                        q_local = q[y0:y1, x0:x1].reshape(-1)
                        off = torch.randint(0, q_local.numel(), (1,)).item() if float(q_local.sum().item()) <= 0 else torch.multinomial(
                            (q_local / q_local.sum().clamp_min(1e-12)).clamp_min(1e-12), 1
                        ).item()
                        dy = off // (x1 - x0)
                        dx = off % (x1 - x0)
                        ys[i] = y0 + dy
                        xs[i] = x0 + dx
            else:
                q = self.prob_maps[gid]
                p = q.reshape(-1)
                p = (torch.ones_like(p) / p.numel()) if float(p.sum().item()) <= 0 else (p / p.sum())
                if self.uniform_mix > 0:
                    p = (1 - self.uniform_mix) * p + self.uniform_mix * (torch.ones_like(p) / p.numel())
                p = p / p.sum().clamp_min(1e-12)
                ids = torch.multinomial(p.clamp_min(1e-12), k, replacement=True)
                ys = (ids // W_).long()
                xs = (ids % W_).long()

            sel_b.append(torch.full((k,), b, dtype=torch.long))
            sel_y.append(ys)
            sel_x.append(xs)

        if len(sel_b) == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=self.device)
            return {"indices": torch.stack([empty, empty, empty], dim=-1), "image": torch.zeros((0, 3), device=self.device)}

        b_idx = torch.cat(sel_b, dim=0)
        ys = torch.cat(sel_y, dim=0)
        xs = torch.cat(sel_x, dim=0)

        # 4) Gather to device
        rgb = imgs_cpu[b_idx, ys, xs, :].to(self.device)

        collated: Dict[str, Any] = {}
        for key, value in image_batch.items():
            if key in ("image_idx", "src_imgs", "src_idxs", "sparse_sfm_points", "sparse_pts"):
                continue
            if isinstance(value, torch.Tensor) and value.ndim >= 3:
                sel = value[b_idx, ys, xs]
                collated[key] = sel.to(self.device)
            else:
                collated[key] = value
        collated["image"] = rgb

        indices = torch.stack([b_idx.long(), ys, xs], dim=-1)
        if indices.numel() > 0:
            indices[:, 0] = image_batch["image_idx"][b_idx]
        collated["indices"] = indices.to(self.device)

        if self.keep_full_image:
            collated["full_image"] = image_batch["image"]

        return collated

    # ---------------- EMA update + incremental maintenance (CPU) ----------------

    @torch.no_grad()
    def update(self, indices: Optional[torch.Tensor], per_ray_loss: Optional[torch.Tensor], alpha: float = 1.0) -> None:
        if not self.hardness_enabled or indices is None or per_ray_loss is None or indices.numel() == 0:
            return

        idx = indices.long().detach().cpu()
        loss = per_ray_loss.reshape(-1).float().detach().cpu()
        if idx.ndim != 2 or idx.shape[1] < 3:
            raise ValueError("indices must have shape [num_rays, >=3]")
        if idx.shape[0] != loss.numel():
            raise ValueError("indices and per_ray_loss must contain the same number of rays")

        # Group rays by image once. This avoids scanning the complete ray batch
        # with a Boolean mask for every selected image while preserving the
        # original within-image observation order.
        positions = torch.arange(idx.shape[0], dtype=torch.long)
        order = torch.argsort(idx[:, 0] * (idx.shape[0] + 1) + positions)
        idx = idx[order]
        loss = loss[order]
        gids, counts = torch.unique_consecutive(idx[:, 0], return_counts=True)

        updated_unique_pixels = 0
        absolute_hardness_delta = 0.0
        offset = 0

        for gid, count in zip(gids.tolist(), counts.tolist()):
            part = idx[offset : offset + count]
            losses = loss[offset : offset + count]
            offset += count
            gid_i = int(gid)

            w = self.loss_weights[gid_i]
            prior = self.prior_maps[gid_i]
            q = self.prob_maps[gid_i]

            yy = part[:, 1].clamp(0, w.shape[0] - 1)
            xx = part[:, 2].clamp(0, w.shape[1] - 1)

            result = update_sampler_image_state(
                hardness=w,
                prior=prior,
                probability=q,
                y=yy,
                x=xx,
                per_ray_loss=losses,
                decay=self.ema,
                scale=float(alpha),
                enabled=True,
            )
            if result.y.numel() == 0:
                continue

            updated_unique_pixels += int(result.y.numel())
            if self.feedback_update_calls == 0:
                absolute_hardness_delta += float(result.delta.abs().sum().item())

            # q = prior * hardness, so each unique pixel contributes exactly
            # one probability-mass delta even when sampled repeatedly.
            dq = prior[result.y, result.x] * result.delta
            self.prob_sums[gid_i] = float(max(self.prob_sums[gid_i] + dq.sum().item(), 1e-12))

            self._update_block_sums_incremental(gid_i, result.y, result.x, dq)

        if updated_unique_pixels > 0:
            self.feedback_update_calls += 1
            if self.feedback_update_calls == 1:
                mean_abs_delta = absolute_hardness_delta / updated_unique_pixels
                print(
                    "[AdaptivePixelSampler] EMA feedback connected: "
                    f"rays={loss.numel()}, unique_pixels={updated_unique_pixels}, "
                    f"mean_abs_hardness_delta={mean_abs_delta:.6g}, "
                    f"decay={self.ema}, hardness_enabled={self.hardness_enabled}"
                )
