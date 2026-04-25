"""Bridge from camera-frame capture to simulator-frame world coordinates.

We do three things here, in order:

1. **Orient.**  Estimate gravity direction from the captured velocity
   field — the falling stream's mean velocity is the best signal we have
   for "down."  We rotate the cloud so gravity aligns with the simulator's
   convention (gravity = -Z).

2. **Scale.**  The captured cloud is up to a single global factor.  Scale
   so its tallest world-space extent matches a target height (default
   15 cm — the user's choice for "real bottle scale").

3. **Translate.**  Place the lowest point of the fluid at z = 0 and centre
   the cloud in x and y, so it lives at a sensible spot in the simulator.

Output is the bundled sim-frame point cloud + velocity vectors, ready to
be voxelised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class WorldFluid:
    """Captured fluid expressed in the simulator's world frame."""
    points: np.ndarray       # (N, 3) float32 metres
    velocities: np.ndarray   # (N, 3) float32 metres / second
    colors: np.ndarray       # (N, 3) uint8
    transform: np.ndarray    # (4, 4) float32 — full transform from capture frame to world
    scale_factor: float      # scalar applied during scaling step
    gravity_dir_capture: np.ndarray  # (3,) the direction we identified as "down" in capture frame
    bounds_min: np.ndarray   # (3,) world-frame AABB min
    bounds_max: np.ndarray   # (3,) world-frame AABB max


# -----------------------------------------------------------------------------
# Step 1: orient (estimate gravity in capture frame, build a rotation to align with -Z)
# -----------------------------------------------------------------------------

def estimate_gravity_direction(
    velocities: np.ndarray,
    speed_percentile: float = 90.0,
) -> np.ndarray:
    """Return the unit vector pointing in the gravity direction in the
    capture frame.

    Heuristic: take the top percentile of velocities by magnitude (the
    fast-moving fluid is overwhelmingly the falling pour stream); their
    mean direction is the gravity direction.  Robust to noise from
    pool-surface micro-motion and segmentation artefacts.
    """
    vmag = np.linalg.norm(velocities, axis=1)
    if vmag.size == 0 or vmag.max() <= 1e-9:
        # No motion in the cloud — fall back to "down is -Y in image-space",
        # which is the assumption baked into our pinhole y-flip.
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)

    threshold = float(np.percentile(vmag, speed_percentile))
    fast_mask = vmag >= max(threshold, 1e-9)
    if fast_mask.sum() < 10:
        fast_mask = vmag > 0.0
    fast = velocities[fast_mask]

    mean_dir = fast.mean(axis=0)
    n = float(np.linalg.norm(mean_dir))
    if n < 1e-9:
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    return (mean_dir / n).astype(np.float32)


def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix that maps unit vector ``a`` onto unit vector ``b``.

    Uses the Rodrigues construction; safe when ``a`` and ``b`` are colinear
    (returns identity if same, π-rotation about an arbitrary perpendicular
    axis if antiparallel).
    """
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a /= np.linalg.norm(a) + 1e-12
    b /= np.linalg.norm(b) + 1e-12
    dot = float(np.dot(a, b))

    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float32)
    if dot < -1.0 + 1e-9:
        # Antiparallel — pick any axis perpendicular to a.
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis) + 1e-12
        # 180° rotation
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ], dtype=np.float64)
        R = np.eye(3) + 2.0 * (K @ K)  # cos(π) = -1, sin(π) = 0
        return R.astype(np.float32)

    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)
    R = np.eye(3) + K + K @ K * ((1.0 - dot) / (s * s))
    return R.astype(np.float32)


# -----------------------------------------------------------------------------
# Step 2 + 3: scale + translate to put the cloud at the user's target size
# -----------------------------------------------------------------------------

def transform_to_world(
    points: np.ndarray,
    velocities: np.ndarray,
    colors: np.ndarray,
    target_height_m: float = 0.15,
) -> WorldFluid:
    """Run the full orient + scale + translate pipeline.

    Parameters
    ----------
    points, velocities : (N, 3) float
        From the capture step, in the camera frame (arbitrary units).
    colors : (N, 3) uint8
        Per-point RGB (passed through unchanged).
    target_height_m : float
        World-space height the tallest extent of the cloud should map to.

    Returns
    -------
    WorldFluid
        Same point/velocity/colour arrays expressed in metres in the
        simulator's world frame, plus the transform that was applied (so
        downstream voxelisation can record provenance).
    """
    if points.shape != velocities.shape or points.shape[1] != 3:
        raise ValueError(f"Expected matching (N,3) shapes; got {points.shape} / {velocities.shape}")
    if len(points) == 0:
        raise ValueError("Empty input cloud — nothing to bridge.")

    # 1. Identify gravity direction in the capture frame.
    g_capture = estimate_gravity_direction(velocities)

    # 2. Build rotation that maps that gravity direction to world -Z.
    target_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    R = rotation_aligning(g_capture.astype(np.float64), target_gravity).astype(np.float32)

    pts_rot = (points.astype(np.float32) @ R.T)
    vel_rot = (velocities.astype(np.float32) @ R.T)

    # 3. Scale uniformly so the longest extent of the rotated cloud equals
    #    target_height_m.  The "longest extent" choice tracks user intent —
    #    we asked them to set the *height* of the captured fluid.
    extents = pts_rot.max(axis=0) - pts_rot.min(axis=0)
    longest = float(extents.max())
    if longest < 1e-9:
        raise ValueError("Captured cloud has zero extent.")
    scale_factor = float(target_height_m / longest)

    pts_scl = pts_rot * scale_factor
    vel_scl = vel_rot * scale_factor   # velocities scale with positions (per second)

    # 4. Translate so the bottom of the cloud sits at z = 0 and the cloud
    #    is centred in x/y.
    bbox_min = pts_scl.min(axis=0)
    bbox_max = pts_scl.max(axis=0)
    centre_xy = 0.5 * (bbox_min + bbox_max)
    translation = np.array([
        -centre_xy[0],
        -centre_xy[1],
        -bbox_min[2],
    ], dtype=np.float32)
    pts_world = pts_scl + translation
    bounds_min = pts_world.min(axis=0)
    bounds_max = pts_world.max(axis=0)

    # 5. Build the full 4×4 transform for provenance.
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R * scale_factor   # combined rotation + uniform scale
    T[:3, 3] = translation
    # (Strictly: T applied to a homogeneous point is R·scale·p + translation.)

    return WorldFluid(
        points=pts_world.astype(np.float32),
        velocities=vel_scl.astype(np.float32),
        colors=colors.astype(np.uint8),
        transform=T,
        scale_factor=scale_factor,
        gravity_dir_capture=g_capture.astype(np.float32),
        bounds_min=bounds_min.astype(np.float32),
        bounds_max=bounds_max.astype(np.float32),
    )


def load_fluid_state_npz(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Convenience loader for the .npz that capture_fluid.py emits."""
    data = np.load(str(path))
    return (
        data["points"].astype(np.float32),
        data["velocities"].astype(np.float32),
        data["colors"].astype(np.uint8),
        {
            "intrinsics": data["intrinsics"].astype(np.float64),
            "dt": float(data["dt"]),
            "pixel_indices": data["pixel_indices"].astype(np.int32),
        },
    )
