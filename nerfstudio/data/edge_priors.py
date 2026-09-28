"""CPU edge responses used by EGSA's static sampling prior."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_edge_map_cpu(img_hwc: torch.Tensor, edge_type: str = "sobel") -> torch.Tensor:
    """Return a min-max-normalised HxW response with a 1e-6 floor.

    Canny uses the non-maximum-suppressed *magnitude* output, not its binary
    hysteresis edge map. Missing/broken Canny dependencies never select Sobel
    silently. Sobel retains the existing CPU implementation and arithmetic.
    """
    edge_type = str(edge_type).lower()
    if edge_type not in ("sobel", "canny"):
        raise ValueError(f"Unsupported edge_type: {edge_type!r}; use 'sobel' or 'canny'")
    img = img_hwc.cpu()
    if img.dtype != torch.float32:
        img = img.float()
    if float(img.max().item()) > 1.0:
        img = img / 255.0
    gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    gray = gray.unsqueeze(0).unsqueeze(0)

    if edge_type == "canny":
        try:
            import kornia.filters as KF
        except ImportError as exc:
            raise ImportError(
                "EGSA edge_type='canny' requires Kornia. From the EGSA repository "
                "root, complete setup with: python -m pip install -r requirements.txt. "
                "Sobel fallback is intentionally disabled."
            ) from exc
        # Explicit Kornia 0.6.12 parameters for a reproducible edge response.
        magnitude, _ = KF.canny(
            gray, low_threshold=0.1, high_threshold=0.2,
            kernel_size=(5, 5), sigma=(1.0, 1.0), hysteresis=True, eps=1e-6,
        )
        magnitude = magnitude[0, 0]
        magnitude = magnitude - magnitude.min()
        return (magnitude / magnitude.max().clamp(min=1e-12)).clamp_min(1e-6).cpu()

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy)[0, 0]
    mag = mag - mag.min()
    mag = mag / max(mag.max().item(), 1e-12)
    return mag.clamp_min(1e-6).cpu()
