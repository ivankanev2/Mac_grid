"""Straight-cylinder fitting.

After an AABB crop + denoise, the remaining points should be dominated by
the pipe's outer surface.  We exploit that with a simple, robust recipe
rather than a full 7-parameter RANSAC:

    1. PCA:        first principal direction ≈ pipe axis; centroid is the
                   axis anchor point.  Works because the pipe is much
                   longer than it is wide, so axial variance dominates.
    2. Radius:     median orthogonal distance from points to the axis.
                   Median is robust against the few floor-splash points
                   that survive the crop.
    3. Refine:     drop points whose |r − r_median| exceeds
                   ``radial_tol * MAD(r)``, re-centre, re-PCA, re-fit.
                   Iterate a handful of times.

We return the axis anchor, unit direction, radius, the inlier mask over
the input points, and a small diagnostics dict for printing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CylinderFit:
    axis_point: np.ndarray      # (3,) any point on the axis (we use the centroid)
    axis_dir: np.ndarray        # (3,) unit vector along the axis
    radius: float               # metres
    inlier_mask: np.ndarray     # (N,) bool, over the *input* points
    diagnostics: dict


def _pca_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (centroid, unit axis direction) via PCA of points."""
    centroid = points.mean(axis=0)
    centred = points - centroid
    # SVD is numerically friendlier than eig on the covariance for thin
    # clouds; the first right-singular vector is the dominant direction.
    _u, _s, vh = np.linalg.svd(centred, full_matrices=False)
    axis = vh[0]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return centroid, axis


def _radial_distances(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
) -> np.ndarray:
    """Orthogonal distance from each point to the line (axis_point, axis_dir)."""
    rel = points - axis_point
    # project onto the axis, subtract to get the radial component.
    axial = rel @ axis_dir
    orth = rel - np.outer(axial, axis_dir)
    return np.linalg.norm(orth, axis=1)


def fit_straight_cylinder(
    points: np.ndarray,
    radial_tol: float = 2.0,
    max_iters: int = 6,
    min_inliers: int = 200,
) -> CylinderFit:
    """Fit a straight cylinder to a point cloud.

    Parameters
    ----------
    points : (N, 3) ndarray
        The cropped, denoised point cloud that's dominated by the pipe.
    radial_tol : float
        In each refinement pass, points with ``|r − r_median| > radial_tol *
        MAD(r)`` are treated as outliers.  Larger → more permissive.
    max_iters : int
        Upper bound on refinement passes; we also stop early when the
        inlier set stops shrinking.
    min_inliers : int
        Safety net: never iterate down below this count.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3); got {points.shape}")
    if len(points) < max(min_inliers, 10):
        raise ValueError(
            f"Too few points to fit a cylinder: {len(points)} "
            f"(need ≥ {max(min_inliers, 10)})"
        )

    N = len(points)
    inlier_mask = np.ones(N, dtype=bool)
    last_count = N
    radius = 0.0
    axis_point = np.zeros(3)
    axis_dir = np.array([0.0, 0.0, 1.0])
    iters_run = 0

    for it in range(max_iters):
        iters_run = it + 1
        working = points[inlier_mask]
        axis_point, axis_dir = _pca_axis(working)

        r = _radial_distances(working, axis_point, axis_dir)
        radius = float(np.median(r))
        mad = float(np.median(np.abs(r - radius)))
        if mad < 1e-9:
            # Perfectly circular at this resolution — nothing to refine.
            break

        # Update inlier_mask from the working-mask-space back to full-points-space.
        keep_working = np.abs(r - radius) <= radial_tol * mad
        new_mask = np.zeros(N, dtype=bool)
        full_indices = np.where(inlier_mask)[0]
        new_mask[full_indices[keep_working]] = True

        new_count = int(new_mask.sum())
        if new_count < min_inliers:
            # Back off — the threshold was too aggressive this round.
            break
        if new_count == last_count:
            inlier_mask = new_mask
            break
        inlier_mask = new_mask
        last_count = new_count

    # Final fit on the refined inliers.
    working = points[inlier_mask]
    if len(working) >= 3:
        axis_point, axis_dir = _pca_axis(working)
        r = _radial_distances(working, axis_point, axis_dir)
        radius = float(np.median(r))

    diagnostics = {
        "input_points": N,
        "inliers": int(inlier_mask.sum()),
        "radius_median": radius,
        "radius_mad": float(
            np.median(
                np.abs(
                    _radial_distances(points[inlier_mask], axis_point, axis_dir)
                    - radius
                )
            )
        )
        if int(inlier_mask.sum()) > 1
        else 0.0,
        "iters_run": iters_run,
    }

    return CylinderFit(
        axis_point=axis_point,
        axis_dir=axis_dir,
        radius=radius,
        inlier_mask=inlier_mask,
        diagnostics=diagnostics,
    )
