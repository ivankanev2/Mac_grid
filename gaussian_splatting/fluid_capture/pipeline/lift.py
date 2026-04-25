"""Lift 2D depth + optical flow into a 3D point cloud with per-point velocities.

The recipe:

1. **Lift positions.**  For each fluid pixel ``(r, c)`` with depth ``z``
   from frame A, back-project through the pinhole model:
       X = (c - cx) * z / fx
       Y = -(r - cy) * z / fy        # negate Y so up is +Y, not +row
       Z = z

2. **Find the same material point in frame B.**  Optical flow gives
   ``(du, dv)`` — the pixel-space displacement of that fluid material
   point from frame A to frame B.  Sample ``depth_b`` bilinearly at
   ``(c + du, r + dv)`` to get its frame-B depth.  Lift again with the
   same intrinsics.

3. **Velocity = (P_b − P_a) / dt.**

We optionally drop points whose flow exits the frame, whose flow points
into a region with zero or extreme depth, or whose final velocity
magnitude is unreasonably large (clipped at a percentile of the
distribution to kill outliers).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from .frames import CameraIntrinsics


@dataclass
class FluidState:
    """Output of the lift step.

    All arrays are aligned: row ``i`` of each is the same fluid surface
    sample.  Coordinates are in the camera frame in arbitrary units (the
    same units as ``depth``).
    """
    points: np.ndarray              # (N, 3) float32, 3D positions at frame A
    velocities: np.ndarray          # (N, 3) float32, 3D velocity vectors
    colors: np.ndarray              # (N, 3) uint8, per-point RGB sampled from frame A
    pixel_indices: np.ndarray       # (N, 2) int32, (row, col) in frame A
    intrinsics: CameraIntrinsics    # camera model used for back-projection
    dt: float                       # seconds between frame A and frame B


def _lift_pixels(
    rows: np.ndarray,
    cols: np.ndarray,
    depths: np.ndarray,
    K: CameraIntrinsics,
) -> np.ndarray:
    """Pinhole back-projection of pixel arrays with corresponding depths.

    Returns (N, 3) float32 in the camera frame: +X right, +Y up,
    +Z forward (away from the camera).
    """
    X = (cols - K.cx) * depths / K.fx
    Y = -(rows - K.cy) * depths / K.fy
    Z = depths
    return np.stack([X, Y, Z], axis=1).astype(np.float32)


def _bilinear_sample(field: np.ndarray, rows_f: np.ndarray, cols_f: np.ndarray) -> np.ndarray:
    """Bilinear sample of a 2-D scalar field at floating-point pixel coordinates.

    Out-of-bounds samples are clamped to the edge.
    """
    H, W = field.shape
    cols_clamped = np.clip(cols_f, 0.0, W - 1.0)
    rows_clamped = np.clip(rows_f, 0.0, H - 1.0)

    c0 = np.floor(cols_clamped).astype(np.int32)
    c1 = np.minimum(c0 + 1, W - 1)
    r0 = np.floor(rows_clamped).astype(np.int32)
    r1 = np.minimum(r0 + 1, H - 1)
    fc = (cols_clamped - c0).astype(np.float32)
    fr = (rows_clamped - r0).astype(np.float32)

    f00 = field[r0, c0]
    f01 = field[r0, c1]
    f10 = field[r1, c0]
    f11 = field[r1, c1]
    return (
        (1 - fr) * ((1 - fc) * f00 + fc * f01)
        + fr * ((1 - fc) * f10 + fc * f11)
    ).astype(np.float32)


def lift_to_fluid_state(
    image_a_rgb: np.ndarray,
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    flow_ab: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    dt: float,
    velocity_clip_pct: float = 99.0,
) -> FluidState:
    """Run the full lift step and return a :class:`FluidState`.

    Parameters
    ----------
    image_a_rgb : (H, W, 3) uint8
        Frame A.  Used only to colour the output point cloud.
    depth_a : (H, W) float32
        Per-pixel depth at frame A.
    depth_b : (H, W) float32
        Per-pixel depth at frame B.  Used to lift flow-warped pixels.
    flow_ab : (H, W, 2) float32
        Optical flow from A to B, in pixels.  ``flow_ab[..., 0]`` is column
        displacement, ``flow_ab[..., 1]`` is row displacement.
    mask : (H, W) bool
        Which pixels in frame A are fluid.
    intrinsics : :class:`CameraIntrinsics`
        Pinhole model used for back-projection.
    dt : float
        Time gap between frame A and frame B in seconds.
    velocity_clip_pct : float
        Drop the top ``100 - velocity_clip_pct`` percent of velocity
        magnitudes as outliers (clamped to the percentile, not zeroed).
        Set to 100 to disable clipping.
    """
    if depth_a.shape != depth_b.shape:
        raise ValueError(
            f"depth_a {depth_a.shape} and depth_b {depth_b.shape} must match"
        )
    if image_a_rgb.shape[:2] != depth_a.shape:
        raise ValueError("image and depth must share H and W")
    if mask.shape != depth_a.shape:
        raise ValueError("mask and depth must share H and W")
    if flow_ab.shape[:2] != depth_a.shape or flow_ab.shape[2] != 2:
        raise ValueError(f"flow_ab must be (H, W, 2); got {flow_ab.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")

    H, W = depth_a.shape
    rows_full, cols_full = np.indices((H, W))
    rows_m = rows_full[mask].astype(np.float32)
    cols_m = cols_full[mask].astype(np.float32)
    depth_a_m = depth_a[mask].astype(np.float32)

    # Drop pixels with non-finite or zero depth — these can't be lifted.
    finite = np.isfinite(depth_a_m) & (depth_a_m > 0.0)
    rows_m = rows_m[finite]
    cols_m = cols_m[finite]
    depth_a_m = depth_a_m[finite]

    # Position at frame A.
    points_a = _lift_pixels(rows_m, cols_m, depth_a_m, intrinsics)

    # Position at frame B = lift the flow-warped pixel using depth_b at that
    # location.
    flow_m = flow_ab[mask][finite]  # (N, 2)
    new_cols = cols_m + flow_m[..., 0]
    new_rows = rows_m + flow_m[..., 1]
    depth_b_at_flow = _bilinear_sample(depth_b, new_rows, new_cols)

    valid_b = np.isfinite(depth_b_at_flow) & (depth_b_at_flow > 0.0)
    # We don't drop these — we just zero the velocity for invalid samples,
    # so the point cloud stays the same size as the position list.
    points_b = _lift_pixels(new_rows, new_cols, depth_b_at_flow, intrinsics)
    velocities = (points_b - points_a) / float(dt)
    velocities[~valid_b] = 0.0

    # Outlier clipping by percentile.  Anything above that magnitude is
    # almost certainly a flow / depth error, not real motion.
    if velocity_clip_pct < 100.0:
        vmag = np.linalg.norm(velocities, axis=1)
        if vmag.size > 0:
            threshold = float(np.percentile(vmag, velocity_clip_pct))
            if threshold > 0:
                scale = np.where(vmag > threshold, threshold / np.maximum(vmag, 1e-9), 1.0)
                velocities = velocities * scale[:, None]

    colors_m = image_a_rgb[mask][finite]
    pixel_indices = np.stack(
        [rows_m.astype(np.int32), cols_m.astype(np.int32)],
        axis=1,
    )

    return FluidState(
        points=points_a,
        velocities=velocities.astype(np.float32),
        colors=colors_m.astype(np.uint8),
        pixel_indices=pixel_indices,
        intrinsics=intrinsics,
        dt=float(dt),
    )


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def save_rgb_point_cloud(path: str | Path, state: FluidState) -> None:
    """Save the lifted points coloured by the original image RGB."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(state.points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(state.colors.astype(np.float64) / 255.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    if not ok:
        raise IOError(f"Failed to write {path}")


def save_velocity_magnitude_point_cloud(path: str | Path, state: FluidState) -> None:
    """Save the lifted points coloured by velocity magnitude (red=fast, blue=slow)."""
    vmag = np.linalg.norm(state.velocities, axis=1)
    if vmag.size == 0:
        # Empty cloud — write an empty PLY anyway so downstream tools don't break.
        pcd = o3d.geometry.PointCloud()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
        return
    vmax = float(vmag.max())
    if vmax <= 1e-12:
        normalised = np.zeros_like(vmag)
    else:
        normalised = vmag / vmax
    colors = np.zeros((len(vmag), 3), dtype=np.float64)
    colors[:, 0] = normalised             # red ramp with speed
    colors[:, 2] = 1.0 - normalised       # blue ramp inversely

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(state.points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    if not ok:
        raise IOError(f"Failed to write {path}")


def save_state_npz(path: str | Path, state: FluidState) -> None:
    """Save the raw arrays for Milestone 2 to consume without re-running models."""
    K = state.intrinsics
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        points=state.points.astype(np.float32),
        velocities=state.velocities.astype(np.float32),
        colors=state.colors.astype(np.uint8),
        pixel_indices=state.pixel_indices.astype(np.int32),
        intrinsics=np.array(
            [K.fx, K.fy, K.cx, K.cy, K.width, K.height], dtype=np.float64
        ),
        dt=np.float32(state.dt),
    )
