#!/usr/bin/env python3
"""extract_fluid_state_v2.py — Phase C v2 extractor.

Differences from v1 (extract_fluid_state.py)
--------------------------------------------
1. Oil motion threshold default 3.0 → 0.05.
   Empirically (see diagnose_motion_classes.py output) the pool Gaussians
   sit at velocity 0.1–1.0 in canonical-units / normalised-time, while
   v1's threshold of 3.0 was killing them.  0.05 keeps the pool *and* the
   column and only drops genuinely static noise.

2. Bottle extraction (NEW).
   The grey-class Gaussians (sat<0.20, val>=0.30) form a recognisable
   bottle silhouette in the trained model.  We spatially filter them
   (keep only those near the oil cluster), compute their temporal-mean
   position across N sample times to suppress deformation-MLP noise,
   voxelise them into a static occupancy mask, dilate by 1 cell to seal
   wall gaps, and emit ``bottle_solid.bin`` alongside the per-frame fluid
   states.  The simulator picks this up and applies it via
   ``MACWater3D::setVoxelSolids`` so the captured fluid actually has a
   container to settle into.

3. Coordinate scale.
   v1 scaled the captured cloud's longest extent to ``--target-height``,
   which compressed the bottle + column system uniformly.  v2 scales by
   the bottle's z-extent instead (``--bottle-height-target``, default
   0.15 m), preserving the natural proportions between bottle and column.

4. Single global grid.
   v1 sized the simulator grid per-frame, so the origin drifted between
   frames and the SDF/particle render paths went out of sync.  v2
   computes ONE global grid containing the bottle bbox plus the union of
   all per-frame oil bboxes (with headroom above for the column), and
   emits every frame on that fixed grid.  Same fixed grid is used to
   voxelise the bottle.

5. Dilation default 1 → 0 for fluid.
   v1 fattened the (already thin) column by ~3 cells before particle
   seeding.  v2 keeps the captured surface mask and seeds particles
   directly inside it.  Bottle solid uses its own dilation knob
   (``--bottle-dilate-iters``, default 1) since wall gaps are a real
   problem there.

Usage
-----
    cd ~/Deformable-3D-Gaussians
    python dynamic_capture/extract_fluid_state_v2.py \
        --model-dir          output/oil_pour \
        --iteration          40000 \
        --output-dir         dynamic_capture/captured_states_v2 \
        --bottle-height-target 0.15 \
        --oil-motion-threshold 0.05 \
        --motion-samples     10
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


# Make the Deformable-3D-Gaussians repo importable when this script lives in
# .../Deformable-3D-Gaussians/dynamic_capture/.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# -----------------------------------------------------------------------------
# Binary formats — must match the C++ readers in
#   pipe_fluid_engine/include/pipe_fluid/fluid_state_loader.h
# -----------------------------------------------------------------------------
_FST_MAGIC   = 0x46535431   # 'F','S','T','1' — fluid state per-frame
_FST_VERSION = 1
_BSL_MAGIC   = 0x42534C31   # 'B','S','L','1' — bottle solid mask
_BSL_VERSION = 1


_SH_C0 = 0.28209479177387814  # SH(0,0) basis function constant


# =============================================================================
# CLI
# =============================================================================
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v2: extract per-frame fluid + static bottle solid from "
                    "a trained Deformable 3D-Gaussians model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Trained Deformable 3D-GS output directory.")
    p.add_argument("--iteration", type=int, default=-1,
                   help="Checkpoint iteration. -1 = latest.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write sim_state_NNNN.bin + bottle_solid.bin "
                        "+ manifest.json.  Recommended: captured_states_v2.")
    p.add_argument("--n-frames", type=int, default=0,
                   help="Number of time stamps to extract (0 = match training).")
    p.add_argument("--dx", type=float, default=0.015,
                   help="Simulator voxel size (metres).")
    p.add_argument("--bottle-height-target", type=float, default=0.15,
                   help="World-space z-extent the bottle should map to (metres). "
                        "Replaces v1's --target-height.  All other geometry "
                        "scales by the same factor.")
    p.add_argument("--headroom-above", type=float, default=0.10,
                   help="Extra grid headroom above the highest captured oil "
                        "Gaussian (m), so the simulator has room above the "
                        "bottle mouth for the falling column.")
    p.add_argument("--headroom-below", type=float, default=0.05,
                   help="Extra grid headroom below the bottle base (m).")
    p.add_argument("--headroom-lateral", type=float, default=0.05,
                   help="Extra grid headroom to the side of the bottle (m).")
    p.add_argument("--particles-per-cell", type=int, default=25,
                   help="FLIP particles seeded per oil-occupied cell.  v2 bumped "
                        "this from 13 to 25 because the captured oil cloud is "
                        "spatially sparse — denser per-cell seeding gives the "
                        "pressure solver enough particle support to project "
                        "stably instead of buzzing.")
    # HSV thresholds (oil)
    p.add_argument("--hue-min", type=float, default=15.0)
    p.add_argument("--hue-max", type=float, default=45.0)
    p.add_argument("--sat-min", type=float, default=0.30)
    p.add_argument("--val-min", type=float, default=0.20)
    # HSV thresholds (bottle / grey class)
    p.add_argument("--bottle-sat-max", type=float, default=0.20)
    p.add_argument("--bottle-val-min", type=float, default=0.30)
    # Motion filter (oil)
    p.add_argument("--oil-motion-threshold", type=float, default=0.05,
                   help="Drop yellow Gaussians whose max velocity over the "
                        "video is below this.  v1 default was 3.0 (too high; "
                        "killed the pool).")
    p.add_argument("--motion-samples", type=int, default=10,
                   help="Number of evenly-spaced time stamps for the motion "
                        "filter and for the bottle's temporal-mean position.")
    # Bottle spatial filter
    p.add_argument("--bottle-margin-lateral", type=float, default=0.05,
                   help="Lateral expansion (in scaled metres) of the oil bbox "
                        "when filtering grey Gaussians for the bottle.")
    p.add_argument("--bottle-margin-vertical", type=float, default=0.05,
                   help="Vertical expansion of the oil bbox when filtering "
                        "grey Gaussians for the bottle.")
    p.add_argument("--bottle-dilate-iters", type=int, default=1,
                   help="6-connectivity dilation passes applied to the bottle "
                        "solid mask to seal small wall gaps.  1 → +1 cell of "
                        "thickness in each direction.")
    # Bottle base seal — topological closure prior.  4DGS reconstructions of
    # refractive containers under-represent the base because the base is
    # mostly visible only through refraction of the contained fluid.  Without
    # this seal, the bottle is a hollow tube, fluid leaks out the bottom,
    # and pancakes on the simulation's grid floor.
    p.add_argument("--no-seal-bottle-base", dest="seal_bottle_base",
                   action="store_false", default=True,
                   help="Disable the closed-bottom topological prior.  By "
                        "default we infer the bottle's full xy footprint by "
                        "filling holes in the captured wall outline, then "
                        "stamp a layer of solid cells at the lowest captured "
                        "wall z to close the base.  Without this, fluid "
                        "leaks through the missing base.")
    p.add_argument("--seal-base-thickness", type=int, default=1,
                   help="Number of solid cell layers added at the bottle's "
                        "z-floor when --seal-bottle-base is enabled.  1 = a "
                        "single voxel-thick disk; 2 = thicker base.  Higher "
                        "is more conservative (no leaks) at the cost of a "
                        "shorter usable interior.")
    # Wall-only filter — interior-noise removal.  Grey-class HSV Gaussians
    # don't only live on the bottle's outer walls; 4DGS often captures
    # refraction artifacts and view-dependent visual features as grey
    # Gaussians INSIDE the bottle's volume.  When voxelised, those
    # become spurious solid cells throughout the interior, turning the
    # bottle into a "maze" the fluid has to thread through.  Per z layer,
    # we keep only solid cells with at least one non-solid neighbour in
    # xy; cells fully surrounded by other solid cells in the same layer
    # are dropped.  Walls are preserved because they always border the
    # bottle's interior or exterior.
    p.add_argument("--no-wall-only-filter", dest="wall_only_filter",
                   action="store_false", default=True,
                   help="Disable the per-layer wall-only filter.  Default "
                        "(enabled) removes interior-noise cells from the "
                        "bottle solid mask, keeping only cells on the xy "
                        "boundary of the solid region.")
    # 3D connected-components filter — drops disconnected interior blobs.
    # Sparse refraction-noise voxels inside the bottle's volume can have
    # air neighbours (so they survive the wall-only filter) but are still
    # not part of the bottle's wall shell.  This filter keeps only the
    # largest 26-connected 3D component, dropping all disconnected blobs.
    # Justified by the topological prior: a container is one connected
    # geometric shape.
    p.add_argument("--no-connected-components", dest="connected_components",
                   action="store_false", default=True,
                   help="Disable the 3D connected-components filter.  "
                        "Default (enabled) keeps only the largest 26-connected "
                        "3D component of the bottle solid mask, dropping "
                        "disconnected interior-noise blobs.")
    # Exterior-facing wall filter — strongest of the noise filters.  Per
    # z layer, flood-fill exterior from the grid boundary; keep only
    # solid cells with at least one exterior 8-neighbour.  Drops
    # interior noise even when it's diagonally connected to walls.
    p.add_argument("--no-exterior-facing-filter", dest="exterior_facing_filter",
                   action="store_false", default=True,
                   help="Disable the exterior-facing wall filter.  Default "
                        "(enabled) keeps only cells that border the bottle's "
                        "exterior region; drops interior-noise cells even "
                        "when they're 26-connected to walls via diagonals.")
    p.add_argument("--no-exterior-facing-pre-dilate",
                   dest="exterior_facing_pre_dilate",
                   action="store_false", default=True,
                   help="Disable the 1-cell xy pre-dilation that temporarily "
                        "seals small wall gaps before the exterior-facing "
                        "flood-fill.  Default (enabled) prevents the "
                        "flood-fill from leaking through 1-cell-wide gaps "
                        "and misclassifying the bottle's interior as exterior. "
                        "The filter's output still respects the ORIGINAL mask "
                        "thickness — pre-dilation only affects flood-fill "
                        "topology.")
    # v2.2: post-hoc alignment of captured oil to the bottle.  The deformation
    # MLP often places the captured oil column at a 3D position that renders
    # correctly from the (single) training viewpoint but is laterally offset
    # from where the actual bottle is in our extraction.  Without correction,
    # the oil falls past the bottle and pancakes on the grid floor.  This
    # shift moves the oil's xy center to the bottle's xy center and the oil's
    # z-min to the bottle's z-floor, preserving the oil's INTERNAL shape but
    # anchoring it to the bottle.
    p.add_argument("--no-align-oil-to-bottle", dest="align_oil_to_bottle",
                   action="store_false", default=True,
                   help="Disable v2.2 alignment that shifts captured oil's xy "
                        "center onto the bottle's xy center and oil's z-min "
                        "onto the bottle's z-floor.  Default (enabled) "
                        "compensates for deformation-MLP overfitting that "
                        "places oil at the wrong position relative to the "
                        "bottle.  Disable for ablation / diagnostics.")
    # Fluid mask reconstruction
    p.add_argument("--fluid-dilate-iters", type=int, default=1,
                   help="6-connectivity dilation passes applied to the fluid "
                        "occupancy mask before particle seeding.  Each pass "
                        "adds one cell of thickness in each of +-x/y/z, so "
                        "iters=1 turns a single occupied cell into a 7-cell "
                        "+ shape.  We default to 1 because the captured oil "
                        "column at dx=1.5cm is only ~1 cell wide and the pool "
                        "is shallow — without dilation the per-frame oil "
                        "occupancy comes out to 15-35 cells and FLIP physics "
                        "buzzes from too few particles.  Set 0 to keep the "
                        "raw thin column, 2 for an even chunkier representation.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rng-seed", type=int, default=0)
    return p.parse_args(argv)


# =============================================================================
# Model load (mirror of v1 _load_model)
# =============================================================================
def _load_model(model_dir: Path, iteration: int):
    import torch  # noqa: F401
    from scene import Scene, GaussianModel
    from scene.deform_model import DeformModel

    cfg_path = model_dir / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(f"cfg_args not found in {model_dir}")
    cfg_text = cfg_path.read_text(encoding="utf-8").strip()
    from argparse import Namespace
    ns = eval(cfg_text, {"Namespace": Namespace, "__builtins__": {}})
    cfg = vars(ns)

    sh_degree = int(cfg.get("sh_degree", 3))
    is_blender = bool(cfg.get("is_blender", True))
    gaussians = GaussianModel(sh_degree)

    class _Args:
        pass
    a = _Args()
    a.sh_degree = sh_degree
    a.source_path = cfg.get("source_path", "")
    a.model_path = str(model_dir)
    a.images = cfg.get("images", "images")
    a.resolution = int(cfg.get("resolution", -1))
    a.white_background = bool(cfg.get("white_background", False))
    a.data_device = "cuda"
    a.eval = True
    a.is_blender = is_blender
    a.is_6dof = bool(cfg.get("is_6dof", False))
    a.render_process = False
    a.load2gpu_on_the_fly = bool(cfg.get("load2gpu_on_the_fly", False))

    scene = Scene(a, gaussians,
                  load_iteration=iteration if iteration > 0 else -1,
                  shuffle=False)
    deform = DeformModel(is_blender=is_blender, is_6dof=a.is_6dof)
    deform.load_weights(str(model_dir), iteration=scene.loaded_iter)
    return gaussians, deform, scene.loaded_iter


# =============================================================================
# Colour + motion helpers (mirror of diagnostic / v1 helpers)
# =============================================================================
def _canonical_rgb(gaussians) -> np.ndarray:
    import torch
    with torch.no_grad():
        dc = gaussians._features_dc.squeeze(1)
        rgb = (dc * _SH_C0 + 0.5).clamp(0.0, 1.0).cpu().numpy()
    return rgb.astype(np.float32)


def _rgb_to_hsv_180(rgb: np.ndarray):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    delta = cmax - cmin
    hue = np.zeros_like(r)
    nz = delta > 1e-9
    rmax = nz & (cmax == r); gmax = nz & (cmax == g); bmax = nz & (cmax == b)
    hue[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6.0
    hue[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2.0
    hue[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4.0
    hue *= 60.0
    hue_180 = hue * 0.5
    sat = np.where(cmax > 1e-9, delta / np.maximum(cmax, 1e-9), 0.0)
    val = cmax
    return hue_180, sat, val


def _query_deform(deform, xyz_t, t):
    import torch
    N = xyz_t.shape[0]
    tin = torch.full((N, 1), float(t), dtype=xyz_t.dtype, device=xyz_t.device)
    with torch.no_grad():
        d_xyz, _, _ = deform.step(xyz_t.detach(), tin)
    return d_xyz


def _per_gaussian_max_vel(deform, xyz_t, n_samples: int) -> np.ndarray:
    import torch
    N = xyz_t.shape[0]
    if N == 0:
        return np.zeros((0,), np.float32)
    sample_ts = np.linspace(0.0, 1.0, max(2, n_samples + 1))[:-1]
    dt = 1.0 / max(2, n_samples + 1)
    max_v = torch.zeros(N, device=xyz_t.device)
    for st in sample_ts:
        d_a = _query_deform(deform, xyz_t, float(st))
        d_b = _query_deform(deform, xyz_t, float(st + dt))
        vmag = torch.linalg.norm((d_b - d_a) / dt, dim=1)
        max_v = torch.maximum(max_v, vmag)
    return max_v.cpu().numpy().astype(np.float32)


def _per_gaussian_mean_pos(deform, xyz_t, n_samples: int) -> np.ndarray:
    """Average position over N evenly-spaced time stamps.  More robust than
    canonical position for "static" Gaussians whose deformation MLP encodes
    view-dependent wobble (we average it out)."""
    import torch
    N = xyz_t.shape[0]
    if N == 0:
        return np.zeros((0, 3), np.float32)
    sample_ts = np.linspace(0.0, 1.0, max(2, n_samples + 1))[:-1]
    accum = torch.zeros((N, 3), device=xyz_t.device, dtype=xyz_t.dtype)
    for st in sample_ts:
        d = _query_deform(deform, xyz_t, float(st))
        accum += (xyz_t + d)
    accum /= max(1, len(sample_ts))
    return accum.cpu().numpy().astype(np.float32)


# =============================================================================
# Geometry helpers
# =============================================================================
def _estimate_gravity_dir(velocities: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    vmag = np.linalg.norm(velocities, axis=1)
    if vmag.size == 0 or vmag.max() < 1e-9:
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    threshold = float(np.percentile(vmag, percentile))
    fast_mask = vmag >= max(threshold, 1e-9)
    if fast_mask.sum() < 10:
        fast_mask = vmag > 0.0
    mean_dir = velocities[fast_mask].mean(axis=0)
    n = float(np.linalg.norm(mean_dir))
    if n < 1e-9:
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    return (mean_dir / n).astype(np.float32)


def _rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a /= np.linalg.norm(a) + 1e-12
    b /= np.linalg.norm(b) + 1e-12
    dot = float(np.dot(a, b))
    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float32)
    if dot < -1.0 + 1e-9:
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp); axis /= np.linalg.norm(axis) + 1e-12
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        return (np.eye(3) + 2.0 * (K @ K)).astype(np.float32)
    v = np.cross(a, b); s = float(np.linalg.norm(v))
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float64)
    return (np.eye(3) + K + K @ K * ((1.0 - dot) / (s * s))).astype(np.float32)


def _fill_holes_2d(mask_2d: np.ndarray) -> np.ndarray:
    """Fill enclosed holes in a 2D boolean mask via flood-fill from the
    boundary inward.  No scipy dependency.  Used to close the bottle's
    xy cross-section when only the cylinder walls were voxelised.

    The captured grey-class Gaussians give us a ring (or thicker outline)
    of cells; the cylinder's interior is hollow.  After this fill, the
    interior cells are also True, giving us the bottle's full footprint
    so we can stamp a closed base across it.
    """
    if not mask_2d.any():
        return mask_2d.copy()

    nx, ny = mask_2d.shape
    # Mark non-mask cells touching the boundary as "outside" — flood-fill
    # from there through other non-mask cells.  Whatever isn't reached is
    # an enclosed hole.
    outside = np.zeros_like(mask_2d)
    outside[0, :]  = ~mask_2d[0, :]
    outside[-1, :] = ~mask_2d[-1, :]
    outside[:, 0]  = ~mask_2d[:, 0]
    outside[:, -1] = ~mask_2d[:, -1]

    # Iteratively grow.  Convergence is bounded by the grid diameter.
    while True:
        prev = outside.copy()
        outside[1:, :]  |= prev[:-1, :] & ~mask_2d[1:, :]
        outside[:-1, :] |= prev[1:,  :] & ~mask_2d[:-1, :]
        outside[:, 1:]  |= prev[:, :-1] & ~mask_2d[:, 1:]
        outside[:, :-1] |= prev[:, 1:]  & ~mask_2d[:, :-1]
        if (outside == prev).all():
            break

    # Filled = original mask OR cells that are non-mask but unreachable
    # from outside (i.e. interior holes).
    return mask_2d | ~outside


def _keep_largest_3d_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 26-connected 3D component of solid cells.
    Used to drop interior-noise blobs that aren't part of the bottle's
    wall shell.

    Justification: the bottle is one connected geometric shape.  The
    outer wall + base form a single connected 3D component spanning all
    z layers.  Refraction-noise voxels inside the bottle's volume form
    separate 3D components — even if individually they have air
    neighbours (and so survive the wall-only filter), they are
    disconnected from the wall shell.  Dropping all but the largest
    component is the principled way to remove them: a container is, by
    definition, one connected shape.

    Uses scipy.ndimage.label with a 3x3x3 structuring element (26-conn).
    Falls back to a NumPy-only flood-fill labelling if scipy is missing.
    """
    if not mask.any():
        return mask.copy()
    try:
        from scipy.ndimage import label as ndi_label
        structure = np.ones((3, 3, 3), dtype=bool)
        labelled, n_components = ndi_label(mask, structure=structure)
        if n_components <= 1:
            return mask.copy()
        # Count cells per label, ignoring background (label 0).
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0
        largest_label = int(np.argmax(sizes))
        return labelled == largest_label
    except ImportError:
        # NumPy-only fallback — slower but no scipy needed.  BFS flood-fill
        # to label components, then return the largest.
        nx, ny, nz = mask.shape
        labels = np.zeros(mask.shape, dtype=np.int32)
        cur_label = 0
        sizes_list = [0]  # background size at index 0
        for i0 in range(nx):
            for j0 in range(ny):
                for k0 in range(nz):
                    if not mask[i0, j0, k0] or labels[i0, j0, k0] != 0:
                        continue
                    cur_label += 1
                    stack = [(i0, j0, k0)]
                    count = 0
                    while stack:
                        i, j, k = stack.pop()
                        if (i < 0 or i >= nx or j < 0 or j >= ny or
                                k < 0 or k >= nz):
                            continue
                        if labels[i, j, k] != 0 or not mask[i, j, k]:
                            continue
                        labels[i, j, k] = cur_label
                        count += 1
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                for dk in (-1, 0, 1):
                                    if di == 0 and dj == 0 and dk == 0:
                                        continue
                                    stack.append((i + di, j + dj, k + dk))
                    sizes_list.append(count)
        sizes = np.array(sizes_list, dtype=np.int64)
        largest_label = int(np.argmax(sizes))
        return labels == largest_label


def _keep_exterior_facing_cells(mask: np.ndarray,
                                pre_dilate: bool = True) -> np.ndarray:
    """Keep only solid cells that face the bottle's EXTERIOR.

    Per z layer, flood-fill from the grid boundary through non-solid
    cells; whatever gets marked is the exterior (the world outside the
    bottle, reachable from outside the grid without going through walls).
    The bottle's hollow interior is NOT marked because the walls block
    the flood-fill.  We then keep solid cells with at least one exterior
    8-neighbour and drop the rest.

    Net effect:
      - Wall cells border the exterior on their outside face → kept.
      - Interior-noise cells border only the bottle's interior or other
        noise → dropped, even if 26-connected to wall cells via diagonal
        contacts.

    Stronger than the wall-only filter: that one asked "any non-solid
    neighbour"; this one asks "any *exterior* non-solid neighbour".

    Known failure mode: if the wall has gaps, the flood-fill leaks
    through, marks the bottle's interior as exterior, and interior noise
    survives.  ``pre_dilate=True`` (default) temporarily dilates the
    mask by 1 cell in xy for the flood-fill ONLY (the output respects
    the original mask thickness — only originally-solid cells can be
    kept) so small gaps don't break the topology.

    Out-of-grid in xy is treated as exterior, so cells on the grid edge
    are kept by their out-of-grid 8-neighbour.
    """
    nx, ny, nz = mask.shape

    if pre_dilate:
        sealed = mask.copy()
        sealed[1:, :, :]  |= mask[:-1, :, :]
        sealed[:-1, :, :] |= mask[1:, :, :]
        sealed[:, 1:, :]  |= mask[:, :-1, :]
        sealed[:, :-1, :] |= mask[:, 1:, :]
    else:
        sealed = mask

    out = np.zeros_like(mask)
    for k in range(nz):
        original_layer = mask[:, :, k]
        if not original_layer.any():
            continue
        sealed_layer = sealed[:, :, k]

        # Flood-fill exterior from the grid boundary through non-solid
        # cells of the sealed layer.
        exterior = np.zeros_like(sealed_layer)
        exterior[0,  :] = ~sealed_layer[0,  :]
        exterior[-1, :] = ~sealed_layer[-1, :]
        exterior[:,  0] = ~sealed_layer[:,  0]
        exterior[:, -1] = ~sealed_layer[:, -1]
        while True:
            prev = exterior.copy()
            exterior[1:,  :] |= prev[:-1, :] & ~sealed_layer[1:,  :]
            exterior[:-1, :] |= prev[1:,  :] & ~sealed_layer[:-1, :]
            exterior[:,  1:] |= prev[:, :-1] & ~sealed_layer[:,  1:]
            exterior[:, :-1] |= prev[:, 1:]  & ~sealed_layer[:, :-1]
            if (exterior == prev).all():
                break

        # 8-neighbour check, with out-of-grid treated as exterior.
        padded = np.ones((nx + 2, ny + 2), dtype=bool)  # ring is True
        padded[1:-1, 1:-1] = exterior
        any_ext = (
            padded[:-2,  :-2] | padded[:-2,  1:-1] | padded[:-2,  2:] |
            padded[1:-1, :-2] |                      padded[1:-1, 2:] |
            padded[2:,   :-2] | padded[2:,   1:-1] | padded[2:,   2:]
        )
        out[:, :, k] = original_layer & any_ext
    return out


def _keep_xy_boundary_cells(mask: np.ndarray) -> np.ndarray:
    """Per z layer, keep solid cells that have at least one non-solid
    neighbour in xy (8-connected, including diagonals).  Drops cells
    fully surrounded by other solid cells in the same layer.

    Used to remove refraction-noise voxels from the bottle solid mask
    while preserving walls.  A wall always borders the bottle's interior
    or exterior in xy, so it has at least one non-solid 8-neighbour and
    is kept.  Refraction-noise voxels deep inside the bottle's volume,
    surrounded by other noise voxels, are dropped.

    8-connected adjacency (rather than 4-connected) is used because the
    4-connected version produces speckled walls — cells whose 4 cardinal
    neighbours are solid but whose diagonal neighbours are not get
    dropped, creating wall gaps that the fluid can leak through.  The
    8-connected version preserves clean wall outlines.

    Out-of-grid is treated as non-solid, so cells on the grid edge are
    always kept.

    Returns a new mask; the input is unchanged.
    """
    nx, ny, nz = mask.shape
    out = np.zeros_like(mask)
    for k in range(nz):
        layer = mask[:, :, k]
        if not layer.any():
            continue
        # Pad with False (out-of-grid = non-solid).
        padded = np.zeros((nx + 2, ny + 2), dtype=bool)
        padded[1:-1, 1:-1] = layer
        non_solid = ~padded
        # 8-connected non-solid neighbour presence.
        has_l  = non_solid[:-2,  1:-1]
        has_r  = non_solid[2:,   1:-1]
        has_d  = non_solid[1:-1, :-2]
        has_u  = non_solid[1:-1, 2:]
        has_dl = non_solid[:-2,  :-2]
        has_dr = non_solid[2:,   :-2]
        has_ul = non_solid[:-2,  2:]
        has_ur = non_solid[2:,   2:]
        any_nonsolid_neighbour = (has_l | has_r | has_d | has_u |
                                  has_dl | has_dr | has_ul | has_ur)
        out[:, :, k] = layer & any_nonsolid_neighbour
    return out


def _dilate_mask_6connect(mask: np.ndarray, iters: int) -> np.ndarray:
    if iters <= 0 or not mask.any():
        return mask
    out = mask.copy()
    for _ in range(int(iters)):
        nxt = out.copy()
        nxt[1:, :, :] |= out[:-1, :, :]
        nxt[:-1, :, :] |= out[1:, :, :]
        nxt[:, 1:, :] |= out[:, :-1, :]
        nxt[:, :-1, :] |= out[:, 1:, :]
        nxt[:, :, 1:] |= out[:, :, :-1]
        nxt[:, :, :-1] |= out[:, :, 1:]
        out = nxt
    return out


# =============================================================================
# Voxelisation utilities (single global grid)
# =============================================================================
def _voxelise_points(points: np.ndarray,
                     origin: np.ndarray,
                     dx: float,
                     nx: int, ny: int, nz: int) -> np.ndarray:
    """Voxelise (M, 3) points into an (nx, ny, nz) bool occupancy mask.
    Points outside the grid are silently dropped."""
    mask = np.zeros((nx, ny, nz), dtype=bool)
    if points.size == 0:
        return mask
    rel = (points - origin[None, :]) / dx
    idx = np.floor(rel).astype(np.int32)
    in_b = (
        (idx[:, 0] >= 0) & (idx[:, 0] < nx) &
        (idx[:, 1] >= 0) & (idx[:, 1] < ny) &
        (idx[:, 2] >= 0) & (idx[:, 2] < nz)
    )
    idx = idx[in_b]
    if idx.shape[0] > 0:
        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return mask


def _seed_flip_particles(fluid_mask: np.ndarray,
                         points_world: np.ndarray,
                         vels_world: np.ndarray,
                         origin: np.ndarray, dx: float,
                         particles_per_cell: int,
                         rng) -> Tuple[np.ndarray, np.ndarray]:
    """Seed FLIP particles inside the fluid cells; nearest-neighbour velocity
    from the captured Gaussians.

    The NN-velocity step is O(P*G) brute-force in v1 (chunked numpy).  At
    P=40k particles, G=15k Gaussians, 501 frames, that's ~300 billion
    distance comparisons — ~2 hours on CPU.  This implementation prefers
    scipy.spatial.cKDTree (O(P log G), ~150 ms per frame, ~1 minute total)
    and falls back to the brute-force path if scipy is unavailable.
    """
    fluid_idx = np.argwhere(fluid_mask)
    M = fluid_idx.shape[0]
    if M == 0 or points_world.shape[0] == 0:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
    P = M * particles_per_cell
    cell_offsets = rng.random((P, 3), dtype=np.float32)
    cell_idx = np.repeat(fluid_idx, particles_per_cell, axis=0).astype(np.float32)
    pos = (origin[None, :] + (cell_idx + cell_offsets) * dx).astype(np.float32)

    # Fast path — kd-tree.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points_world)
        _, nn_idx = tree.query(pos, k=1)
        vel = vels_world[nn_idx].astype(np.float32, copy=False)
    except ImportError:
        # Slow fallback — brute-force chunked NN if scipy is missing.
        vel = np.empty((P, 3), dtype=np.float32)
        chunk = 2048
        for i0 in range(0, P, chunk):
            i1 = min(i0 + chunk, P)
            d2 = np.sum((pos[i0:i1, None, :] - points_world[None, :, :]) ** 2,
                        axis=2)
            nn = np.argmin(d2, axis=1)
            vel[i0:i1] = vels_world[nn]
    return pos, vel


# =============================================================================
# Binary writers
# =============================================================================
def _write_sim_state_bin(path: Path, origin: np.ndarray,
                         nx: int, ny: int, nz: int, dx: float,
                         pos: np.ndarray, vel: np.ndarray) -> None:
    n = pos.shape[0]
    if vel.shape[0] != n:
        raise ValueError(f"pos/vel size mismatch: {pos.shape} vs {vel.shape}")
    interleaved = np.empty((n, 6), dtype=np.float32)
    interleaved[:, 0:3] = pos.astype(np.float32, copy=False)
    interleaved[:, 3:6] = vel.astype(np.float32, copy=False)
    flat = interleaved.reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack(
            "<IIIII fff fI",
            _FST_MAGIC, _FST_VERSION,
            int(nx), int(ny), int(nz),
            float(dx),
            float(origin[0]), float(origin[1]), float(origin[2]),
            n,
        ))
        f.write(flat.astype("<f4", copy=False).tobytes(order="C"))


def _write_bottle_solid_bin(path: Path, origin: np.ndarray,
                            nx: int, ny: int, nz: int, dx: float,
                            mask: np.ndarray) -> None:
    """Format:
       uint32 magic       = 'BSL1' (0x42534C31)
       uint32 version     = 1
       uint32 nx, ny, nz
       float32 dx
       float32 origin_x, origin_y, origin_z
       uint32 n_solid_cells   (informational; mask still has nx*ny*nz bytes)
       uint8 [nx*ny*nz]       0 = air, 1 = solid     (i + nx*(j + ny*k))
    """
    if mask.shape != (nx, ny, nz):
        raise ValueError(
            f"mask shape {mask.shape} != ({nx},{ny},{nz})"
        )
    n_solid = int(mask.sum())
    # Match the simulator's index convention: i + nx*(j + ny*k).
    # numpy stores [i,j,k] in C-order, which is k major; we need i major,
    # so we transpose to (k, j, i) then ravel C-order.
    flat = np.ascontiguousarray(mask.transpose(2, 1, 0)).astype(np.uint8)
    # After transpose, flat[k,j,i] is the original mask[i,j,k]; ravel C-order
    # gives index k*nj*ni + j*ni + i, but the simulator expects i + nx*(j + ny*k)
    # which is i + nx*j + nx*ny*k = same thing.  Verified.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack(
            "<IIIII fff fI",
            _BSL_MAGIC, _BSL_VERSION,
            int(nx), int(ny), int(nz),
            float(dx),
            float(origin[0]), float(origin[1]), float(origin[2]),
            n_solid,
        ))
        f.write(flat.tobytes(order="C"))


# =============================================================================
# Main
# =============================================================================
def main(argv=None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  extract_fluid_state_v2.py — oil + bottle from Deformable 3D-GS")
    print("=" * 72)
    print(f"  Model dir:           {args.model_dir}")
    print(f"  Iteration:           {'latest' if args.iteration < 0 else args.iteration}")
    print(f"  Output dir:          {args.output_dir}")
    print(f"  N frames:            {'auto' if args.n_frames == 0 else args.n_frames}")
    print(f"  dx (cell size, m):   {args.dx}")
    print(f"  Bottle target h (m): {args.bottle_height_target}")
    print(f"  Oil motion thresh:   {args.oil_motion_threshold}")
    print(f"  Motion samples:      {args.motion_samples}")
    print(f"  Bottle dilate iters: {args.bottle_dilate_iters}")
    print(f"  Fluid  dilate iters: {args.fluid_dilate_iters}")
    print("-" * 72)

    print("[1/7] Loading trained Deformable 3D-GS model ...")
    import torch
    gaussians, deform, loaded_iter = _load_model(args.model_dir, args.iteration)
    canonical_xyz = gaussians.get_xyz.detach()
    n_total = int(canonical_xyz.shape[0])
    print(f"      iteration={loaded_iter}  N_canonical={n_total}")

    print("[2/7] Classifying canonical Gaussians by HSV colour ...")
    rgb = _canonical_rgb(gaussians)
    h, s, v = _rgb_to_hsv_180(rgb)
    yellow_mask_np = (
        (h >= args.hue_min) & (h <= args.hue_max)
        & (s >= args.sat_min) & (v >= args.val_min)
    )
    grey_mask_np = (
        (s < args.bottle_sat_max) & (v >= args.bottle_val_min)
        & ~yellow_mask_np
    )
    n_yellow = int(yellow_mask_np.sum())
    n_grey   = int(grey_mask_np.sum())
    print(f"      yellow (oil candidates):    {n_yellow:7d}  "
          f"({100.0 * n_yellow / max(1, n_total):5.2f}%)")
    print(f"      grey   (bottle candidates): {n_grey:7d}  "
          f"({100.0 * n_grey   / max(1, n_total):5.2f}%)")
    if n_yellow < 100:
        print("ERROR: too few yellow Gaussians.", file=sys.stderr)
        return 4
    if n_grey < 100:
        print("ERROR: too few grey Gaussians.", file=sys.stderr)
        return 4

    yellow_xyz_canonical = canonical_xyz[
        torch.from_numpy(yellow_mask_np).to(canonical_xyz.device)
    ]
    grey_xyz_canonical = canonical_xyz[
        torch.from_numpy(grey_mask_np).to(canonical_xyz.device)
    ]

    print("[3/7] Computing per-Gaussian motion (oil filter) ...")
    yellow_max_vel = _per_gaussian_max_vel(
        deform, yellow_xyz_canonical, args.motion_samples,
    )
    oil_keep = yellow_max_vel > float(args.oil_motion_threshold)
    n_oil = int(oil_keep.sum())
    print(f"      yellow max-vel  min={yellow_max_vel.min():.4f}  "
          f"med={np.median(yellow_max_vel):.4f}  "
          f"max={yellow_max_vel.max():.4f}")
    print(f"      oil after motion threshold ({args.oil_motion_threshold}): "
          f"{n_oil} / {n_yellow}  "
          f"({100.0 * n_oil / max(1, n_yellow):5.2f}%)")
    if n_oil < 100:
        print("ERROR: too few oil Gaussians passed motion filter.", file=sys.stderr)
        return 4
    oil_xyz_canonical = yellow_xyz_canonical[
        torch.from_numpy(oil_keep).to(yellow_xyz_canonical.device)
    ]

    # ---- Compute world-frame transform: rotation, scale, translation -------
    print("[4/7] Computing world-frame transform (rotate/scale/translate) ...")
    # 4a) Gravity from sample velocities at t=0.5.
    n_frames = args.n_frames
    if n_frames <= 0:
        # Mirror v1's heuristic.
        n_frames = 200
        try:
            for spath_guess in [
                args.model_dir.parent.parent / "data" / args.model_dir.name,
                Path.cwd() / "data" / args.model_dir.name,
            ]:
                tj = spath_guess / "transforms_train.json"
                if tj.exists():
                    obj = json.loads(tj.read_text())
                    n_frames = len(obj.get("frames", [])) + 50 + 50
                    break
        except Exception:
            pass
    print(f"      n_frames = {n_frames}")
    dt_norm = 1.0 / max(1, n_frames - 1)

    sample_t = 0.5
    with torch.no_grad():
        d_a = _query_deform(deform, oil_xyz_canonical, sample_t)
        d_b = _query_deform(deform, oil_xyz_canonical,
                            min(1.0, sample_t + dt_norm))
        sample_vel = ((d_b - d_a) / dt_norm).cpu().numpy().astype(np.float32)
    g_capture = _estimate_gravity_dir(sample_vel)
    R = _rotation_aligning(g_capture, np.array([0.0, 0.0, -1.0]))
    print(f"      gravity_dir(capture) = "
          f"[{g_capture[0]:+.3f}, {g_capture[1]:+.3f}, {g_capture[2]:+.3f}]")

    # 4b) Bottle reference height — use grey Gaussians' temporal-mean position
    #     so we average out deformation MLP wobble.
    print("      computing bottle temporal-mean positions ...")
    grey_mean_pos = _per_gaussian_mean_pos(
        deform, grey_xyz_canonical, args.motion_samples,
    )
    grey_rot = grey_mean_pos @ R.T
    bottle_z_min_canon = float(np.percentile(grey_rot[:, 2],  1.0))
    bottle_z_max_canon = float(np.percentile(grey_rot[:, 2], 99.0))
    bottle_h_canon = bottle_z_max_canon - bottle_z_min_canon
    if bottle_h_canon < 1e-6:
        print(f"ERROR: degenerate bottle height ({bottle_h_canon:.6e}).", file=sys.stderr)
        return 5
    scale = float(args.bottle_height_target / bottle_h_canon)
    print(f"      bottle z-extent (canonical) = {bottle_h_canon:.6f}")
    print(f"      scale factor = {scale:.6f}")

    # 4c) Oil bbox over all sample times (rotated, scaled).  Use this to
    #     define the lateral and vertical-above grid extent.
    print("      sampling oil bboxes across time ...")
    sample_times = list(np.linspace(0.0, 1.0, max(5, args.motion_samples)))
    oil_min_world = np.array([+np.inf, +np.inf, +np.inf], dtype=np.float64)
    oil_max_world = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    for st in sample_times:
        with torch.no_grad():
            d_xyz = _query_deform(deform, oil_xyz_canonical, float(st))
            pt = (oil_xyz_canonical + d_xyz).cpu().numpy().astype(np.float32)
            pt_rot_scl = (pt @ R.T) * scale
            oil_min_world = np.minimum(oil_min_world, pt_rot_scl.min(axis=0))
            oil_max_world = np.maximum(oil_max_world, pt_rot_scl.max(axis=0))

    # 4d) Bottle bbox after rotation+scale (use temporal-mean positions).
    # IMPORTANT: use percentile-clipped min/max here, not raw .min()/.max().
    # The "grey" HSV class includes a long tail of low-saturation Gaussians
    # that lie OUTSIDE the bottle (table, background, refraction noise).  If
    # we use raw min/max, those outliers stretch the bottle bbox by tens of
    # cm and the bottle solid mask ends up containing the table and
    # background as solid — which then fights the falling fluid.
    # The same p1/p99 percentiles that defined bottle_h_canon for the scale
    # factor are used here so the bbox represents the actual bottle, not
    # the contaminated grey class.
    grey_world_unshifted = grey_rot * scale
    _bottle_clip_lo, _bottle_clip_hi = 1.0, 99.0
    bottle_min_world = np.array([
        np.percentile(grey_world_unshifted[:, 0], _bottle_clip_lo),
        np.percentile(grey_world_unshifted[:, 1], _bottle_clip_lo),
        np.percentile(grey_world_unshifted[:, 2], _bottle_clip_lo),
    ], dtype=np.float64)
    bottle_max_world = np.array([
        np.percentile(grey_world_unshifted[:, 0], _bottle_clip_hi),
        np.percentile(grey_world_unshifted[:, 1], _bottle_clip_hi),
        np.percentile(grey_world_unshifted[:, 2], _bottle_clip_hi),
    ], dtype=np.float64)

    # 4e) Translation: put the bottle base at z=0 and centre xy on the bottle.
    centre_xy = 0.5 * (bottle_min_world[:2] + bottle_max_world[:2])
    translation = np.array([
        -centre_xy[0], -centre_xy[1], -bottle_min_world[2],
    ], dtype=np.float32)
    bottle_min_world += translation
    bottle_max_world += translation
    oil_min_world    += translation
    oil_max_world    += translation
    print(f"      bottle world bbox: "
          f"[{bottle_min_world[0]:+.3f}, {bottle_min_world[1]:+.3f}, {bottle_min_world[2]:+.3f}] "
          f"-> "
          f"[{bottle_max_world[0]:+.3f}, {bottle_max_world[1]:+.3f}, {bottle_max_world[2]:+.3f}]")
    print(f"      oil    world bbox: "
          f"[{oil_min_world[0]:+.3f}, {oil_min_world[1]:+.3f}, {oil_min_world[2]:+.3f}] "
          f"-> "
          f"[{oil_max_world[0]:+.3f}, {oil_max_world[1]:+.3f}, {oil_max_world[2]:+.3f}]")

    # ---- 4ee) v2.2: align captured oil to the bottle ---------------------
    # Compute a per-axis shift so that:
    #   - oil's xy center coincides with bottle's xy center
    #   - oil's z-min coincides with bottle's z-floor (so the pool sits on
    #     the bottle base instead of below or above it)
    # The shift is applied PER FRAME inside the main loop (so velocities
    # are unaffected — this is a pure translation in world space).  Here
    # we just compute it and update the oil bbox so grid sizing accounts
    # for the shifted oil extent.
    if args.align_oil_to_bottle:
        oil_center_xy = 0.5 * (oil_min_world[:2] + oil_max_world[:2])
        bottle_center_xy = 0.5 * (bottle_min_world[:2] + bottle_max_world[:2])
        oil_align_offset = np.array([
            float(bottle_center_xy[0] - oil_center_xy[0]),
            float(bottle_center_xy[1] - oil_center_xy[1]),
            float(bottle_min_world[2] - oil_min_world[2]),
        ], dtype=np.float64)
        oil_min_world += oil_align_offset
        oil_max_world += oil_align_offset
        print(f"      align-oil-to-bottle offset: "
              f"[{oil_align_offset[0]:+.4f}, "
              f"{oil_align_offset[1]:+.4f}, "
              f"{oil_align_offset[2]:+.4f}] m")
        print(f"      oil bbox AFTER alignment: "
              f"[{oil_min_world[0]:+.3f}, {oil_min_world[1]:+.3f}, {oil_min_world[2]:+.3f}] "
              f"-> "
              f"[{oil_max_world[0]:+.3f}, {oil_max_world[1]:+.3f}, {oil_max_world[2]:+.3f}]")
    else:
        oil_align_offset = np.zeros(3, dtype=np.float64)
        print("      align-oil-to-bottle: disabled (--no-align-oil-to-bottle)")

    # ---- 4f) Build the global grid ---------------------------------------
    grid_min = np.minimum(bottle_min_world, oil_min_world).astype(np.float32)
    grid_max = np.maximum(bottle_max_world, oil_max_world).astype(np.float32)
    grid_min[0] -= args.headroom_lateral
    grid_min[1] -= args.headroom_lateral
    grid_min[2] -= args.headroom_below
    grid_max[0] += args.headroom_lateral
    grid_max[1] += args.headroom_lateral
    grid_max[2] += args.headroom_above
    extent = grid_max - grid_min
    nx = max(1, int(np.ceil(extent[0] / args.dx)))
    ny = max(1, int(np.ceil(extent[1] / args.dx)))
    nz = max(1, int(np.ceil(extent[2] / args.dx)))
    grid_origin = grid_min.astype(np.float32)
    print(f"      global grid: {nx}x{ny}x{nz} cells "
          f"({nx*ny*nz} total) at dx={args.dx}")
    print(f"      grid origin: "
          f"[{grid_origin[0]:+.4f}, {grid_origin[1]:+.4f}, {grid_origin[2]:+.4f}] "
          f"size={extent[0]:.3f}x{extent[1]:.3f}x{extent[2]:.3f} m")

    # ---- 5) Bottle solid mask -------------------------------------------
    print("[5/7] Voxelising bottle solid mask ...")
    grey_world = grey_world_unshifted + translation
    # Spatial filter: keep grey points within the (percentile-clipped) bottle
    # bbox plus a small margin.  Filtering by the BOTTLE bbox (not the oil
    # bbox) is what discards the table / background / refraction noise that
    # the HSV "grey" classifier picks up — those points lie outside the
    # bottle's clipped bbox by definition.
    keep = (
        (grey_world[:, 0] >= bottle_min_world[0] - args.bottle_margin_lateral) &
        (grey_world[:, 0] <= bottle_max_world[0] + args.bottle_margin_lateral) &
        (grey_world[:, 1] >= bottle_min_world[1] - args.bottle_margin_lateral) &
        (grey_world[:, 1] <= bottle_max_world[1] + args.bottle_margin_lateral) &
        (grey_world[:, 2] >= bottle_min_world[2] - args.bottle_margin_vertical) &
        (grey_world[:, 2] <= bottle_max_world[2] + args.bottle_margin_vertical)
    )
    bottle_pts = grey_world[keep].astype(np.float32)
    print(f"      grey before spatial filter: {grey_world.shape[0]}")
    print(f"      grey after  spatial filter: {bottle_pts.shape[0]}")

    bottle_surface_mask = _voxelise_points(bottle_pts, grid_origin, args.dx,
                                            nx, ny, nz)
    bottle_mask = _dilate_mask_6connect(bottle_surface_mask,
                                         args.bottle_dilate_iters)
    n_voxelised = int(bottle_mask.sum())
    print(f"      bottle voxelised cells (walls + interior noise): "
          f"{n_voxelised} ({100.0 * n_voxelised / max(1, nx*ny*nz):.2f}% of grid)")

    # Wall-only filter — applied BEFORE the base seal so the seal's
    # centre cells (which are interior to a closed disk) are not dropped.
    if args.wall_only_filter:
        bottle_mask = _keep_xy_boundary_cells(bottle_mask)
        n_after_filter = int(bottle_mask.sum())
        n_dropped = n_voxelised - n_after_filter
        print(f"      wall-only filter:   dropped {n_dropped} interior "
              f"noise cells, kept {n_after_filter} wall cells "
              f"({100.0 * n_after_filter / max(1, nx*ny*nz):.2f}% of grid)")

    # 3D connected-components filter — keep only the largest connected
    # component of solid cells.  Drops sparse interior-noise blobs that
    # have air neighbours (and so survive the wall-only filter) but are
    # not part of the bottle's wall shell.  Applied BEFORE the base seal
    # so the seal cells (which would form a separate disk component if
    # the wall doesn't reach z=0) are not lost.
    if args.connected_components:
        n_before_cc = int(bottle_mask.sum())
        bottle_mask = _keep_largest_3d_component(bottle_mask)
        n_after_cc = int(bottle_mask.sum())
        n_dropped_cc = n_before_cc - n_after_cc
        print(f"      connected-components: dropped {n_dropped_cc} cells "
              f"in disconnected blobs, kept {n_after_cc} cells in the "
              f"largest 3D component "
              f"({100.0 * n_after_cc / max(1, nx*ny*nz):.2f}% of grid)")

    # Exterior-facing wall filter — drops interior-noise cells that are
    # diagonally connected to walls (which the connected-components
    # filter doesn't catch).  Applied BEFORE the base seal because the
    # seal's centre cells border only solid (the seal disk is filled),
    # which would be classified as not-exterior-facing and dropped if
    # the filter ran on the seal.
    if args.exterior_facing_filter:
        n_before_ext = int(bottle_mask.sum())
        bottle_mask = _keep_exterior_facing_cells(
            bottle_mask, pre_dilate=bool(args.exterior_facing_pre_dilate))
        n_after_ext = int(bottle_mask.sum())
        n_dropped_ext = n_before_ext - n_after_ext
        print(f"      exterior-facing filter (pre_dilate="
              f"{bool(args.exterior_facing_pre_dilate)}): dropped "
              f"{n_dropped_ext} cells with no exterior 8-neighbour, "
              f"kept {n_after_ext} cells "
              f"({100.0 * n_after_ext / max(1, nx*ny*nz):.2f}% of grid)")

    # ---- Seal the bottle base ------------------------------------------
    # Refractive-container failure mode of monocular 4DGS: the bottle's
    # base is mostly visible through refraction of the contained fluid,
    # so the grey-class Gaussians under-represent it.  After voxelisation
    # we have a hollow tube (cylinder walls but missing bottom).  Without
    # closure, fluid leaks out the bottom and pancakes on the grid floor.
    #
    # The seal: stamp an elliptical layer of solid cells at the lowest
    # captured wall z, with the ellipse matching the bottle's xy bbox.
    # An ellipse matches the captured bottle's apparent xy aspect (the
    # 4DGS reconstruction often recovers an asymmetric cross-section due
    # to camera-angle bias) without assuming a specific shape.
    #
    # We avoid taking a "fill_holes" of the bottle_mask.any(axis=2)
    # projection, because that's the *union* of all z-layers and the
    # individual-layer holes get covered by walls at neighbouring z's,
    # leaving no enclosed void to fill.  The bbox-ellipse approach is
    # robust to that: it's defined directly from the bottle's xy
    # extent in cells.  Single topological prior: the container has a
    # closed base whose footprint matches its sides.
    if args.seal_bottle_base and bottle_mask.any():
        bottle_idx = np.argwhere(bottle_mask)
        # Bottle's xy bbox in cell coordinates.
        x_lo = int(bottle_idx[:, 0].min())
        x_hi = int(bottle_idx[:, 0].max())
        y_lo = int(bottle_idx[:, 1].min())
        y_hi = int(bottle_idx[:, 1].max())
        x_c  = 0.5 * (x_lo + x_hi)
        y_c  = 0.5 * (y_lo + y_hi)
        x_r  = max(1.0, 0.5 * (x_hi - x_lo))
        y_r  = max(1.0, 0.5 * (y_hi - y_lo))

        # Build an elliptical base footprint matching the bottle's bbox.
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        base_footprint = (((ii - x_c) / x_r) ** 2
                          + ((jj - y_c) / y_r) ** 2) <= 1.0

        k_floor = int(bottle_idx[:, 2].min())
        thickness = max(1, int(args.seal_base_thickness))
        k_lo = max(0, k_floor - thickness + 1)
        k_hi = k_floor

        n_footprint_cells = int(base_footprint.sum())
        n_added = 0
        for k in range(k_lo, k_hi + 1):
            new_cells = base_footprint & ~bottle_mask[:, :, k]
            n_added += int(new_cells.sum())
            bottle_mask[:, :, k] |= base_footprint

        seal_world_z = grid_origin[2] + (k_lo + 0.5) * args.dx
        print(f"      base seal: bottle xy bbox = "
              f"x[{x_lo}..{x_hi}] y[{y_lo}..{y_hi}] cells, "
              f"ellipse footprint = {n_footprint_cells} cells")
        print(f"      base seal: stamped k={k_lo}..{k_hi} "
              f"(world z ~{seal_world_z:.4f} m), "
              f"+{n_added} new solid cells.")

    n_bottle_cells = int(bottle_mask.sum())
    print(f"      bottle solid cells (final): {n_bottle_cells} "
          f"({100.0 * n_bottle_cells / max(1, nx*ny*nz):.2f}% of grid)")

    bottle_path = args.output_dir / "bottle_solid.bin"
    _write_bottle_solid_bin(bottle_path, grid_origin, nx, ny, nz, args.dx,
                            bottle_mask)
    print(f"      wrote {bottle_path}")

    # ---- 6) Per-frame fluid states --------------------------------------
    print("[6/7] Querying deformation MLP per frame, voxelising oil, "
          "writing sim_state_NNNN.bin ...")
    rng = np.random.default_rng(args.rng_seed)
    n_frames_out = 0
    for f_idx in range(n_frames):
        t = f_idx / max(1, n_frames - 1)
        t_next = min(1.0, t + dt_norm)
        with torch.no_grad():
            d_xyz_t  = _query_deform(deform, oil_xyz_canonical, t)
            d_xyz_tp = _query_deform(deform, oil_xyz_canonical, t_next)
            pos_t = (oil_xyz_canonical + d_xyz_t).cpu().numpy().astype(np.float32)
            vel_t = ((d_xyz_tp - d_xyz_t) / max(dt_norm, 1e-9)) \
                .cpu().numpy().astype(np.float32)

        # Apply rotation + scale + translation.
        pos_world = (pos_t @ R.T) * scale + translation
        # v2.2: post-hoc alignment of oil to bottle (xy center + z floor).
        # Pure translation — does NOT affect velocities.
        pos_world = pos_world + oil_align_offset.astype(np.float32)
        vel_world = (vel_t @ R.T) * scale  # velocity scales with positions

        # Build oil occupancy on the GLOBAL grid (NOT a per-frame grid).
        oil_surface_mask = _voxelise_points(pos_world, grid_origin, args.dx,
                                             nx, ny, nz)
        oil_fluid_mask = _dilate_mask_6connect(oil_surface_mask,
                                                args.fluid_dilate_iters)
        # NOTE: we deliberately do NOT subtract the bottle solid mask from
        # the oil mask here.  The earlier v2 did `oil_fluid_mask &= ~bottle_mask`,
        # which erased any oil cell that voxelization quantization had put in
        # the same dx-cube as a bottle wall — including the entire pool's
        # bottom row (sitting on the bottle base).  The simulator's
        # MACWater3D::removeParticlesInSolids handles overlaps correctly:
        # particles seeded inside solid cells get pushed to the nearest air
        # cell on the next step.  Letting that path do its job preserves the
        # pool.

        pos_p, vel_p = _seed_flip_particles(
            oil_fluid_mask, pos_world, vel_world,
            grid_origin, args.dx,
            args.particles_per_cell, rng,
        )
        out_path = args.output_dir / f"sim_state_{f_idx:04d}.bin"
        _write_sim_state_bin(out_path, grid_origin, nx, ny, nz, args.dx,
                             pos_p, vel_p)
        n_frames_out += 1
        if (f_idx + 1) % 25 == 0 or f_idx == n_frames - 1:
            print(f"      [{f_idx + 1:4d}/{n_frames}]  particles={len(pos_p)}  "
                  f"oil_cells={int(oil_fluid_mask.sum())}")

    # ---- 7) Manifest ----------------------------------------------------
    print("[7/7] Writing manifest ...")
    manifest = {
        "version": 2,
        "n_frames": n_frames_out,
        "dx": args.dx,
        "grid": {
            "nx": int(nx), "ny": int(ny), "nz": int(nz),
            "origin_x": float(grid_origin[0]),
            "origin_y": float(grid_origin[1]),
            "origin_z": float(grid_origin[2]),
        },
        "bottle_solid_path": "bottle_solid.bin",
        "bottle_height_target": args.bottle_height_target,
        "scale_factor": scale,
        "iteration": int(loaded_iter),
        "n_canonical_gaussians": n_total,
        "n_yellow_gaussians": int(n_yellow),
        "n_grey_gaussians": int(n_grey),
        "n_oil_gaussians": int(n_oil),
        "n_bottle_solid_cells": int(n_bottle_cells),
        "thresholds": {
            "hue_min": args.hue_min, "hue_max": args.hue_max,
            "sat_min": args.sat_min, "val_min": args.val_min,
            "bottle_sat_max": args.bottle_sat_max,
            "bottle_val_min": args.bottle_val_min,
            "oil_motion_threshold": args.oil_motion_threshold,
        },
        "dilation": {
            "bottle_iters": int(args.bottle_dilate_iters),
            "fluid_iters":  int(args.fluid_dilate_iters),
        },
        "seal_bottle_base":     bool(args.seal_bottle_base),
        "seal_base_thickness":  int(args.seal_base_thickness),
        "wall_only_filter":             bool(args.wall_only_filter),
        "connected_components_filter":  bool(args.connected_components),
        "exterior_facing_filter":       bool(args.exterior_facing_filter),
        "exterior_facing_pre_dilate":   bool(args.exterior_facing_pre_dilate),
        "align_oil_to_bottle": bool(args.align_oil_to_bottle),
        "oil_align_offset": [
            float(oil_align_offset[0]),
            float(oil_align_offset[1]),
            float(oil_align_offset[2]),
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("-" * 72)
    print(f"  Done.  Wrote {n_frames_out} sim_state files + bottle_solid.bin "
          f"+ manifest.json to {args.output_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
