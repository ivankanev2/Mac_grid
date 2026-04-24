"""Interactive 3D crop.

Opens an Open3D editing viewer.  The user Shift + Left-Clicks points to
define an axis-aligned bounding box around the pipe, then closes the
window.  We tightening the crop to the AABB of those picked points and
return the filtered geometry.

Why an AABB rather than an oriented box?  Because we only need a *rough*
ROI — downstream PCA + RANSAC tighten the fit to the true axis.  An AABB
is fast, easy to explain and hard to mess up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


@dataclass
class CropResult:
    """Geometry after the crop, plus the AABB used (for logging / reuse)."""

    points: o3d.geometry.PointCloud
    bbox_min: np.ndarray  # shape (3,)
    bbox_max: np.ndarray  # shape (3,)


def _aabb_from_corners(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> o3d.geometry.AxisAlignedBoundingBox:
    lo = np.minimum(bbox_min, bbox_max)
    hi = np.maximum(bbox_min, bbox_max)
    return o3d.geometry.AxisAlignedBoundingBox(lo, hi)


def crop_interactive(
    pcd: o3d.geometry.PointCloud,
    window_name: str = "Pick >= 2 corners around the pipe, then close",
) -> CropResult:
    """Open the editing viewer and crop the cloud to the AABB of the
    picked points.

    Raises
    ------
    RuntimeError
        If the user closes the window without picking at least 2 points.
    """
    if len(pcd.points) == 0:
        raise ValueError("Cannot crop an empty point cloud")

    print("=" * 72)
    print("  Interactive crop")
    print("-" * 72)
    print("  • Shift + Left-Click picks a 3D point on the surface.")
    print("  • Pick at least 2 points that together bracket the pipe")
    print("    (opposite corners of a loose box is enough — the fit")
    print("     downstream does the fine work).")
    print("  • Close the window when you're done.")
    print("=" * 72)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name, width=1280, height=800)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked_idx = vis.get_picked_points()
    if picked_idx is None or len(picked_idx) < 2:
        raise RuntimeError(
            "Need at least 2 picked points to define a crop bounding box"
        )

    pts = np.asarray(pcd.points)[picked_idx]
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)

    cropped = pcd.crop(_aabb_from_corners(bbox_min, bbox_max))
    print(
        f"  Crop AABB: min={np.round(bbox_min, 4).tolist()}  "
        f"max={np.round(bbox_max, 4).tolist()}"
    )
    print(f"  Points after crop: {len(cropped.points)}")

    return CropResult(points=cropped, bbox_min=bbox_min, bbox_max=bbox_max)


def crop_by_bbox(
    pcd: o3d.geometry.PointCloud,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> CropResult:
    """Non-interactive crop.  Use for scripted runs (``--bbox`` CLI flag)."""
    cropped = pcd.crop(_aabb_from_corners(bbox_min, bbox_max))
    return CropResult(
        points=cropped,
        bbox_min=np.asarray(bbox_min, dtype=float),
        bbox_max=np.asarray(bbox_max, dtype=float),
    )


def parse_bbox_string(s: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a ``"x0,y0,z0,x1,y1,z1"`` CLI bbox string."""
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"--bbox expects 6 comma-separated numbers, got {len(parts)}"
        )
    lo = np.array(parts[:3], dtype=float)
    hi = np.array(parts[3:], dtype=float)
    return np.minimum(lo, hi), np.maximum(lo, hi)
