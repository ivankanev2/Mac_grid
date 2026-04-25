"""HSV-based fluid segmentation.

For our chosen videoplayback-3 (olive oil pouring into a glass bottle on
a clean white background), simple HSV thresholding for yellow is enough
to isolate the fluid pixels.  This module is intentionally minimal —
when we move to harder fluids (transparent water, tinted in a complex
scene, etc.) we'll swap in a learned segmenter (SAM, etc.).

OpenCV HSV ranges (note: H is 0–180, not 0–360):
    Yellow   : H ≈ 15–45
    Orange   : H ≈ 5–20
    Red      : H ≈ 0–10  *and* H ≈ 170–180  (wraparound)
    Green    : H ≈ 40–80
    Blue     : H ≈ 100–130
    Purple   : H ≈ 130–160
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def mask_yellow_fluid(
    image_rgb: np.ndarray,
    hue_min: int = 15,
    hue_max: int = 45,
    sat_min: int = 60,
    val_min: int = 60,
    morph_kernel: int = 5,
) -> np.ndarray:
    """Boolean (H, W) mask of pixels classified as fluid by HSV thresholding.

    Parameters
    ----------
    image_rgb : (H, W, 3) uint8 RGB image.
    hue_min, hue_max : int (0–180)
        OpenCV hue range for the fluid colour.  Defaults are tuned for
        olive-oil yellow.
    sat_min, val_min : int (0–255)
        Reject pixels too desaturated or too dark — keeps glints,
        backgrounds, and very dim fluid edges out of the mask.
    morph_kernel : int
        Kernel size for morphological open + close.  Removes specular
        speckles inside the fluid and bridges thin breaks at edges.
        Set to 0 to disable cleanup entirely.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB image, got {image_rgb.shape}")
    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    raw = (
        (h >= int(hue_min)) & (h <= int(hue_max)) &
        (s >= int(sat_min)) & (v >= int(val_min))
    )

    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(morph_kernel), int(morph_kernel))
        )
        m = (raw.astype(np.uint8) * 255)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        return m > 0

    return raw


def save_mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    overlay_color_rgb: tuple = (255, 0, 0),
    alpha: float = 0.5,
) -> None:
    """Save a debug visualisation: original image with mask tinted on top."""
    if image_rgb.shape[:2] != mask.shape:
        raise ValueError("image and mask must share H and W")

    overlay = image_rgb.copy()
    color = np.array(overlay_color_rgb, dtype=np.float32)
    if mask.any():
        mix = alpha * color + (1.0 - alpha) * overlay[mask].astype(np.float32)
        overlay[mask] = np.clip(mix, 0, 255).astype(np.uint8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
