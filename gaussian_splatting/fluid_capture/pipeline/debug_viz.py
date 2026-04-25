"""Debug visualisations for each pipeline stage.

Saving these PNGs is the cheapest way to figure out where things went
wrong if the lifted point cloud looks bad:

* depth heatmap                 — is the fluid actually being detected
                                   as foreground vs the white background?
* optical-flow HSV visualisation — is RAFT picking up the pour motion?
* mask overlay                  — is the segmentation grabbing the right
                                   pixels (and not also grabbing reflections)?

Each function takes the array(s) it needs plus an output path and writes
a single PNG.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_depth_heatmap(
    depth: np.ndarray,
    out_path: str | Path,
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> None:
    """Render a (H, W) depth array as a colour heatmap PNG.

    The depth is normalised to its own [min, max] range first, so the
    output highlights *relative* depth even when the raw values are in
    arbitrary units (which they are for Depth Anything V2).
    """
    if depth.ndim != 2:
        raise ValueError(f"Expected (H, W) depth, got {depth.shape}")
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError("Depth array contains no finite values")
    d_min = float(depth[finite].min())
    d_max = float(depth[finite].max())
    if d_max - d_min < 1e-12:
        norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        norm = ((depth - d_min) / (d_max - d_min) * 255.0).clip(0, 255).astype(np.uint8)
    norm[~finite] = 0
    heat = cv2.applyColorMap(norm, colormap)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), heat)


def save_flow_hsv(
    flow: np.ndarray,
    out_path: str | Path,
    max_magnitude: float | None = None,
) -> None:
    """Standard optical-flow visualisation: hue = direction, value = magnitude.

    If ``max_magnitude`` is None, the magnitude is normalised to its own
    99th percentile so that a few outlier vectors don't blow out the
    contrast.
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected (H, W, 2) flow, got {flow.shape}")
    H, W, _ = flow.shape
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    ang = np.arctan2(fy, fx)               # radians, range (-pi, pi]
    if max_magnitude is None:
        clip = float(np.percentile(mag, 99.0)) if mag.size > 0 else 0.0
        if clip <= 1e-9:
            clip = 1.0
    else:
        clip = float(max_magnitude)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180.0).astype(np.uint8)  # 0..179
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(mag / clip, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)
