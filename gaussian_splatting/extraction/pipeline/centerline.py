"""Axis endpoints → length + direction + start.

Given a fitted straight cylinder and its inlier points, produce the
parameters the ``.pipe`` blueprint needs:

    start       — one of the two endpoints on the axis.
    direction   — normalised vector from start to the other endpoint.
    length      — distance between the two endpoints.

We derive endpoints by projecting inliers onto the axis and taking the
min / max extents, then trimming a small fraction at each end to reject
fuzzy end-cap noise (the splash field tends to leak a few points past
the true cap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Centerline:
    start: np.ndarray       # (3,) world-space start position
    end: np.ndarray         # (3,) world-space end position
    direction: np.ndarray   # (3,) unit vector from start to end
    length: float           # metres


def derive_centerline(
    inlier_points: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    trim_frac: float = 0.02,
) -> Centerline:
    """Compute start / end / length from inliers projected onto the axis.

    Parameters
    ----------
    inlier_points : (M, 3) ndarray
        The points classified as pipe-surface by :func:`fit_straight_cylinder`.
    axis_point, axis_dir : (3,) ndarray
        From the cylinder fit.  ``axis_dir`` must be unit length.
    trim_frac : float
        Trim this fraction of the axial extent from each end when choosing
        the endpoints.  Defaults to 0.02 — small enough to be a no-op in
        clean scenes and big enough to reject the thin spray of points
        that survives denoising at the caps.
    """
    if not np.isfinite(inlier_points).all():
        raise ValueError("inlier_points contain non-finite values")
    if inlier_points.shape[0] < 10:
        raise ValueError(
            f"Not enough inliers to derive the centreline: {inlier_points.shape[0]}"
        )

    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
    t = (inlier_points - axis_point) @ axis_dir  # axial coordinate per inlier

    if trim_frac > 0.0:
        lo = float(np.quantile(t, trim_frac))
        hi = float(np.quantile(t, 1.0 - trim_frac))
    else:
        lo = float(t.min())
        hi = float(t.max())

    if hi <= lo:
        # Fall back to untrimmed extents (degenerate, but don't crash).
        lo = float(t.min())
        hi = float(t.max())
        if hi <= lo:
            raise ValueError("Inliers collapse to a single point on the axis")

    start = axis_point + lo * axis_dir
    end = axis_point + hi * axis_dir
    length = float(hi - lo)
    direction = (end - start) / (length + 1e-12)

    return Centerline(start=start, end=end, direction=direction, length=length)
