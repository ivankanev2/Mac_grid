#!/usr/bin/env python3
"""Phase C: extract per-frame fluid state from a trained Deformable 3D-GS model.

Pipeline:

    1. Load the trained Deformable 3D-Gaussians model (canonical Gaussians +
       deformation MLP) from a Deformable 3D-GS output directory.
    2. HSV-threshold the canonical Gaussian colours to identify the "oil"
       Gaussians (yellow-ish hue, high saturation).
    3. For each video time stamp, query the deformation MLP to get the
       positions of those oil Gaussians at that moment, plus their
       velocities by central finite difference.
    4. Transform the cloud to the simulator's world frame (rotate gravity
       to -Z, scale so the longest extent matches --target-height,
       translate so the bottom sits at z=0 and centre x/y).
    5. Voxelise + vertical-fill into the simulator's MAC grid; seed FLIP
       particles inside fluid cells with velocities sampled from the
       nearest captured Gaussian.
    6. Emit one ``sim_state_NNNN.bin`` per video frame in the same binary
       format the simulator's loadFluidState already understands.

The script must be run from inside the Deformable-3D-Gaussians repo (or
with that repo on PYTHONPATH) so that ``scene``, ``gaussian_renderer``,
and ``arguments`` import correctly.

Usage
-----
    cd ~/Deformable-3D-Gaussians
    python dynamic_capture/extract_fluid_state.py \\
        --model-dir       output/oil_pour \\
        --iteration       40000 \\
        --output-dir      captured_states \\
        --n-frames        501 \\
        --target-height   0.15 \\
        --hue-min         15 --hue-max 45
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Make Deformable-3D-Gaussians importable when this script lives in
# .../Deformable-3D-Gaussians/dynamic_capture/.
# -----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# -----------------------------------------------------------------------------
# Binary fluid_state format — must match
# pipe_fluid_engine/include/pipe_fluid/fluid_state_loader.h
# -----------------------------------------------------------------------------
_FST_MAGIC = 0x46535431   # 'F','S','T','1'
_FST_VERSION = 1


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-frame fluid state from a trained "
                    "Deformable 3D-Gaussians model (Phase C).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Trained Deformable 3D-GS output directory "
                        "(e.g. output/oil_pour).")
    p.add_argument("--iteration", type=int, default=-1,
                   help="Checkpoint iteration to load. -1 = latest.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory for sim_state_NNNN.bin time series.")
    p.add_argument("--n-frames", type=int, default=0,
                   help="Number of time stamps to extract (0 = match the "
                        "training frame count).")
    p.add_argument("--target-height", type=float, default=0.15,
                   help="World-space height the tallest extent of the "
                        "captured cloud should map to (metres).")
    p.add_argument("--dx", type=float, default=0.015,
                   help="Simulator voxel size (metres).")
    p.add_argument("--headroom-below", type=float, default=0.20,
                   help="Domain headroom below the fluid (m).")
    p.add_argument("--headroom-above", type=float, default=0.05,
                   help="Domain headroom above the fluid (m).")
    p.add_argument("--headroom-lateral", type=float, default=0.05,
                   help="Domain lateral headroom (m).")
    p.add_argument("--particles-per-cell", type=int, default=13,
                   help="FLIP particles seeded per fluid cell.")
    # HSV thresholds — defaults tuned for olive oil.
    p.add_argument("--hue-min", type=float, default=15.0,
                   help="HSV hue lower bound for oil (0-180, OpenCV scale).")
    p.add_argument("--hue-max", type=float, default=45.0,
                   help="HSV hue upper bound for oil (0-180, OpenCV scale).")
    p.add_argument("--sat-min", type=float, default=0.30,
                   help="HSV saturation minimum (0-1).")
    p.add_argument("--val-min", type=float, default=0.20,
                   help="HSV value minimum (0-1).")
    # Motion-based filter (Phase C1.5).
    p.add_argument("--motion-threshold", type=float, default=0.1,
                   help="Drop Gaussians whose maximum velocity over the video "
                        "is below this value (in canonical units per "
                        "normalised-time).  HSV-yellow filtering alone tends "
                        "to grab static bottle/glass Gaussians that are tinted "
                        "yellow when oil is present; motion filtering keeps "
                        "only Gaussians that genuinely move during the pour.")
    p.add_argument("--motion-samples", type=int, default=10,
                   help="Number of evenly-spaced time stamps used to compute "
                        "per-Gaussian max velocity for the motion filter.")
    # Volume reconstruction (Phase C1.6).
    p.add_argument("--fill-mode",
                   choices=["dilate", "vertical", "none"],
                   default="dilate",
                   help="How to recover volume from the captured surface "
                        "Gaussians.  'dilate' (recommended): voxelise the "
                        "captured points and expand the resulting mask by "
                        "--dilate-iters cells in every direction, thickening "
                        "the captured shell without filling gaps between "
                        "disjoint fluid regions.  'vertical': for each (x,y) "
                        "column, fill all cells between lowest and highest "
                        "occupied k — works for puddle-like captures, "
                        "creates brick artefacts for column-into-pool "
                        "configurations.  'none': use the raw voxelised "
                        "surface mask, no fill.")
    p.add_argument("--dilate-iters", type=int, default=1,
                   help="Number of 6-connectivity dilation passes for "
                        "fill-mode=dilate.  1 → ~3-cell-thick shell; "
                        "2 → ~5-cell-thick; etc.  Higher values give more "
                        "fluid volume but smear column-vs-pool detail.")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device for model queries.")
    p.add_argument("--rng-seed", type=int, default=0,
                   help="Random seed for FLIP particle jitter.")
    return p.parse_args(argv)


# -----------------------------------------------------------------------------
# Deformable 3D-GS model loader (uses the upstream repo's modules)
# -----------------------------------------------------------------------------
def _load_model(model_dir: Path, iteration: int, device: str):
    """Return (gaussians, deform, search_iter)."""
    # Heavy imports deferred so --help works without torch loaded.
    import torch  # noqa: F401
    from scene import Scene, GaussianModel
    from scene.deform_model import DeformModel
    from arguments import ModelParams, PipelineParams

    # We can't use get_combined_args() because that wants a parser; instead
    # we manually replicate what render.py does: read cfg_args from the
    # model dir, instantiate ModelParams with sentinel, and load the scene.
    cfg_path = model_dir / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Could not find cfg_args in {model_dir} — is this a "
            "Deformable-3D-GS output directory?"
        )

    # cfg_args is the repr of an argparse.Namespace saved by train.py; eval
    # it back into a Namespace and turn it into a plain dict via vars().
    # WARNING: cfg_args files written by Deformable-3D-GS are trusted.
    cfg_text = cfg_path.read_text(encoding="utf-8").strip()
    from argparse import Namespace
    try:
        ns = eval(cfg_text, {"Namespace": Namespace, "__builtins__": {}})
        cfg_dict = vars(ns)
    except Exception as e:
        raise RuntimeError(f"Failed to parse {cfg_path}: {e}") from e

    sh_degree = int(cfg_dict.get("sh_degree", 3))
    is_blender = bool(cfg_dict.get("is_blender", True))

    gaussians = GaussianModel(sh_degree)
    # Construct a minimal namespace that Scene's loader is happy with.
    class _ModelArgs:
        pass
    margs = _ModelArgs()
    margs.sh_degree = sh_degree
    margs.source_path = cfg_dict.get("source_path", "")
    margs.model_path = str(model_dir)
    margs.images = cfg_dict.get("images", "images")
    margs.resolution = int(cfg_dict.get("resolution", -1))
    margs.white_background = bool(cfg_dict.get("white_background", False))
    margs.data_device = "cuda"
    margs.eval = True
    margs.is_blender = is_blender
    margs.is_6dof = bool(cfg_dict.get("is_6dof", False))
    margs.render_process = False
    margs.load2gpu_on_the_fly = bool(cfg_dict.get("load2gpu_on_the_fly", False))

    scene = Scene(margs, gaussians,
                  load_iteration=iteration if iteration > 0 else -1,
                  shuffle=False)
    search_iter = scene.loaded_iter

    deform = DeformModel(is_blender=is_blender, is_6dof=margs.is_6dof)
    deform.load_weights(str(model_dir), iteration=search_iter)

    return gaussians, deform, search_iter


# -----------------------------------------------------------------------------
# HSV segmentation on canonical Gaussian colours
# -----------------------------------------------------------------------------
_SH_C0 = 0.28209479177387814  # SH(0,0) basis function constant


def _canonical_rgb(gaussians) -> np.ndarray:
    """Compute approximate diffuse RGB per Gaussian from the SH degree-0 (DC)
    component.  Higher-order SH terms are ignored — they encode view-dependent
    appearance, which is small for diffuse oil and bottle.
    """
    import torch
    with torch.no_grad():
        dc = gaussians._features_dc.squeeze(1)  # (N, 3)
        rgb = (dc * _SH_C0 + 0.5).clamp(0.0, 1.0).cpu().numpy()
    return rgb.astype(np.float32)


def _hsv_yellow_mask(
    rgb_01: np.ndarray,
    hue_min_deg180: float, hue_max_deg180: float,
    sat_min: float, val_min: float,
) -> np.ndarray:
    """Return boolean (N,) mask of Gaussians whose canonical colour matches
    the oil-yellow HSV band.  Hue is on OpenCV's 0-180 scale to match the
    M1 segmenter's convention.
    """
    r, g, b = rgb_01[:, 0], rgb_01[:, 1], rgb_01[:, 2]
    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    delta = cmax - cmin

    # Hue in [0, 360)
    hue = np.zeros_like(r)
    nz = delta > 1e-9
    rmax = nz & (cmax == r)
    gmax = nz & (cmax == g)
    bmax = nz & (cmax == b)
    hue[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6.0
    hue[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2.0
    hue[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4.0
    hue *= 60.0  # → degrees
    # OpenCV scales hue to [0, 180] by halving.
    hue_180 = hue * 0.5

    sat = np.where(cmax > 1e-9, delta / np.maximum(cmax, 1e-9), 0.0)
    val = cmax

    return (
        (hue_180 >= hue_min_deg180) & (hue_180 <= hue_max_deg180)
        & (sat >= sat_min) & (val >= val_min)
    )


# -----------------------------------------------------------------------------
# Deformation queries
# -----------------------------------------------------------------------------
def _query_deformation(deform, canonical_xyz_t, time_t):
    """Return (Δxyz, Δrotation, Δscale) at scalar ``time_t`` for all canonical
    Gaussians.  Uses the same wiring render.py does."""
    import torch
    N = canonical_xyz_t.shape[0]
    time_input = torch.full((N, 1), float(time_t),
                            dtype=canonical_xyz_t.dtype,
                            device=canonical_xyz_t.device)
    with torch.no_grad():
        d_xyz, d_rot, d_scale = deform.step(canonical_xyz_t.detach(), time_input)
    return d_xyz, d_rot, d_scale


# -----------------------------------------------------------------------------
# Bridge: capture-frame Gaussian cloud → simulator world frame
# -----------------------------------------------------------------------------
def _estimate_gravity_dir(velocities: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """The fast-moving Gaussians are the falling oil; their mean direction is
    'down' in the capture frame."""
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
    """3x3 rotation matrix that maps unit vector ``a`` onto unit vector ``b``."""
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


# -----------------------------------------------------------------------------
# Voxelisation + FLIP seeding (mirrors fluid_capture/pipeline/voxelize.py)
# -----------------------------------------------------------------------------
def _dilate_mask_6connect(mask: np.ndarray, iters: int) -> np.ndarray:
    """Numpy-only 6-connectivity 3D dilation.  Each pass adds one cell of
    thickness in each of ±x, ±y, ±z.  No scipy dependency."""
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


def _voxelise_and_seed(
    points: np.ndarray, velocities: np.ndarray,
    bounds_min: np.ndarray, bounds_max: np.ndarray,
    dx: float, headroom_below: float, headroom_above: float,
    headroom_lateral: float, particles_per_cell: int, rng,
    fill_mode: str = "dilate", dilate_iters: int = 1,
):
    """Build the MAC grid, occupancy mask (with the chosen fill strategy),
    and seed FLIP particles.  Returns (grid_origin, (nx, ny, nz), pos, vel).

    ``fill_mode``:
      - ``"dilate"`` — voxelise surface points, then morphologically dilate
        the mask by ``dilate_iters`` 6-connectivity passes.  Adds local
        thickness without bridging disjoint fluid regions.  Best for
        column-into-pool scenes.
      - ``"vertical"`` — for each (i, j) column, fill cells between min and
        max occupied k.  Works for puddle-like single-region captures;
        creates brick artefacts on multi-region scenes (the column-into-pool
        bug).
      - ``"none"`` — raw surface mask, no fill.
    """
    domain_min = np.array([
        bounds_min[0] - headroom_lateral,
        bounds_min[1] - headroom_lateral,
        bounds_min[2] - headroom_below,
    ], dtype=np.float32)
    domain_max = np.array([
        bounds_max[0] + headroom_lateral,
        bounds_max[1] + headroom_lateral,
        bounds_max[2] + headroom_above,
    ], dtype=np.float32)
    extent = domain_max - domain_min
    nx = max(1, int(np.ceil(extent[0] / dx)))
    ny = max(1, int(np.ceil(extent[1] / dx)))
    nz = max(1, int(np.ceil(extent[2] / dx)))

    if points.size == 0:
        return domain_min, (nx, ny, nz), \
            np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    rel = (points - domain_min) / dx
    idx = np.floor(rel).astype(np.int32)
    in_b = (
        (idx[:, 0] >= 0) & (idx[:, 0] < nx) &
        (idx[:, 1] >= 0) & (idx[:, 1] < ny) &
        (idx[:, 2] >= 0) & (idx[:, 2] < nz)
    )
    idx = idx[in_b]
    surface_mask = np.zeros((nx, ny, nz), dtype=bool)
    if idx.shape[0] > 0:
        surface_mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # Volume reconstruction.  See docstring for trade-offs between modes.
    if fill_mode == "vertical" and surface_mask.any():
        any_occ = surface_mask.any(axis=2)
        k_idx_grid = np.broadcast_to(np.arange(nz, dtype=np.int32),
                                     (nx, ny, nz))
        masked_min = np.where(surface_mask, k_idx_grid, nz + 1).min(axis=2)
        masked_max = np.where(surface_mask, k_idx_grid, -1).max(axis=2)
        fluid_mask = surface_mask.copy()
        for k in range(nz):
            in_range = (k >= masked_min) & (k <= masked_max) & any_occ
            fluid_mask[..., k] = fluid_mask[..., k] | in_range
    elif fill_mode == "dilate":
        fluid_mask = _dilate_mask_6connect(surface_mask, dilate_iters)
    else:  # "none" or fall-through
        fluid_mask = surface_mask

    fluid_idx = np.argwhere(fluid_mask)
    M = fluid_idx.shape[0]
    if M == 0:
        return domain_min, (nx, ny, nz), \
            np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    P = M * particles_per_cell
    cell_offsets = rng.random((P, 3), dtype=np.float32)
    cell_idx = np.repeat(fluid_idx, particles_per_cell, axis=0).astype(np.float32)
    pos = (domain_min + (cell_idx + cell_offsets) * dx).astype(np.float32)

    # Velocity: nearest-neighbour brute force.  For the modest particle and
    # Gaussian counts in this project (~10⁴ Gaussians, ~10⁵ particles per
    # frame) this is fast enough — and avoids pulling open3d/scipy on the
    # workstation.
    if points.shape[0] == 0:
        vel = np.zeros((P, 3), dtype=np.float32)
    else:
        # Chunked to keep peak memory bounded.
        vel = np.empty((P, 3), dtype=np.float32)
        chunk = 2048
        for i0 in range(0, P, chunk):
            i1 = min(i0 + chunk, P)
            d2 = np.sum(
                (pos[i0:i1, None, :] - points[None, :, :]) ** 2, axis=2,
            )
            nn = np.argmin(d2, axis=1)
            vel[i0:i1] = velocities[nn]

    return domain_min, (nx, ny, nz), pos, vel


# -----------------------------------------------------------------------------
# Binary writer (matches the M2 sim_state.bin format byte-for-byte)
# -----------------------------------------------------------------------------
def _write_sim_state_bin(
    path: Path, grid_origin: np.ndarray,
    nx: int, ny: int, nz: int, dx: float,
    pos: np.ndarray, vel: np.ndarray,
) -> None:
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
            float(grid_origin[0]),
            float(grid_origin[1]),
            float(grid_origin[2]),
            n,
        ))
        f.write(flat.astype("<f4", copy=False).tobytes(order="C"))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Phase C — extract per-frame fluid state from Deformable 3D-GS")
    print("=" * 72)
    print(f"  Model dir:      {args.model_dir}")
    print(f"  Iteration:      {'latest' if args.iteration < 0 else args.iteration}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  N frames:       {'auto' if args.n_frames == 0 else args.n_frames}")
    print(f"  Target height:  {args.target_height}  m")
    print(f"  Voxel size:     {args.dx}  m")
    print(f"  HSV oil band:   hue [{args.hue_min}-{args.hue_max}]  "
          f"sat≥{args.sat_min}  val≥{args.val_min}")
    print(f"  Device:         {args.device}")
    print("-" * 72)

    print("[1/4] Loading trained Deformable 3D-GS model ...")
    import torch
    gaussians, deform, loaded_iter = _load_model(
        args.model_dir, args.iteration, args.device,
    )
    canonical_xyz = gaussians.get_xyz.detach()
    n_gauss = canonical_xyz.shape[0]
    print(f"      iteration={loaded_iter}  canonical Gaussians={n_gauss}")

    print("[2/4] HSV-segmenting oil Gaussians ...")
    rgb = _canonical_rgb(gaussians)
    hsv_mask = _hsv_yellow_mask(
        rgb, args.hue_min, args.hue_max, args.sat_min, args.val_min,
    )
    n_hsv = int(hsv_mask.sum())
    print(f"      after HSV filter:    {n_hsv} / {n_gauss}  "
          f"({100.0 * n_hsv / max(1, n_gauss):.2f}%)")
    if n_hsv < 100:
        print("WARNING: very few HSV-yellow Gaussians selected. Consider "
              "widening the HSV thresholds.", file=sys.stderr)

    # ---- Motion-based filter ------------------------------------------------
    # The HSV filter alone grabs both real oil and bottle/glass Gaussians that
    # appear yellow because oil is behind them.  The bottle is static — its
    # canonical Gaussians barely move under the deformation MLP.  Real oil
    # Gaussians move significantly during the pour.  Sample the deformation
    # field at several time stamps, compute per-Gaussian max velocity, and
    # require it to exceed a threshold.
    print("[2.5/4] Computing per-Gaussian motion magnitudes ...")
    hsv_xyz_canonical = canonical_xyz[
        torch.from_numpy(hsv_mask).to(canonical_xyz.device)
    ]
    sample_ts = np.linspace(0.0, 1.0, max(2, args.motion_samples + 1))[:-1]
    dt_motion = 1.0 / max(2, args.motion_samples + 1)
    n_hsv_g = hsv_xyz_canonical.shape[0]
    max_vel_per_g = torch.zeros(n_hsv_g, device=hsv_xyz_canonical.device)
    for st in sample_ts:
        with torch.no_grad():
            d_a, _, _ = _query_deformation(deform, hsv_xyz_canonical, float(st))
            d_b, _, _ = _query_deformation(
                deform, hsv_xyz_canonical, float(st + dt_motion),
            )
            vel = (d_b - d_a) / dt_motion
            vmag = torch.linalg.norm(vel, dim=1)
            max_vel_per_g = torch.maximum(max_vel_per_g, vmag)
    max_vel_np = max_vel_per_g.cpu().numpy()
    motion_mask_local = max_vel_np > float(args.motion_threshold)
    n_kept = int(motion_mask_local.sum())
    print(f"      max-velocity stats:   "
          f"min={max_vel_np.min():.4f}  median={np.median(max_vel_np):.4f}  "
          f"max={max_vel_np.max():.4f}  threshold={args.motion_threshold:.4f}")
    print(f"      after motion filter:  {n_kept} / {n_hsv}  "
          f"({100.0 * n_kept / max(1, n_hsv):.2f}% of HSV)")
    if n_kept < 50:
        print("WARNING: very few oil Gaussians survived the motion filter. "
              "Lower --motion-threshold if needed.", file=sys.stderr)
    oil_xyz_canonical = hsv_xyz_canonical[
        torch.from_numpy(motion_mask_local).to(hsv_xyz_canonical.device)
    ]
    n_oil = oil_xyz_canonical.shape[0]
    print(f"      final oil Gaussians:  {n_oil} "
          f"({100.0 * n_oil / max(1, n_gauss):.2f}% of all canonical Gaussians)")

    # Number of frames to extract: match the training transforms_train.json
    # frame count if not specified.
    n_frames = args.n_frames
    if n_frames <= 0:
        # Try to read transforms_train.json from the source path.
        from arguments import ModelParams
        # fallback: count the train-set images in the dataset dir.
        n_frames = 200
        # Slightly hacky but works if cfg_args points to data/oil_pour:
        try:
            cfg_text = (args.model_dir / "cfg_args").read_text(encoding="utf-8")
            for k in ("source_path",):
                if f"'{k}'" in cfg_text or f'"{k}"' in cfg_text:
                    pass
            import json
            for spath_guess in [
                args.model_dir.parent.parent / "data" / args.model_dir.name,
                Path.cwd() / "data" / args.model_dir.name,
            ]:
                tj = spath_guess / "transforms_train.json"
                if tj.exists():
                    obj = json.loads(tj.read_text())
                    n_frames = len(obj.get("frames", [])) + 50 + 50  # rough
                    break
        except Exception:
            pass
    print(f"      Extracting {n_frames} time stamps.")

    print("[3/4] Querying deformation MLP per frame, voxelising, writing .bin ...")

    # First pass: compute world-frame transform parameters from a sample
    # frame's velocity field (we use frame n_frames // 2, the middle of the
    # captured time window, since the splash there has well-defined motion).
    sample_t = 0.5
    dt_norm = 1.0 / max(1, n_frames - 1)
    with torch.no_grad():
        d_xyz_a, _, _ = _query_deformation(deform, oil_xyz_canonical, sample_t)
        d_xyz_b, _, _ = _query_deformation(
            deform, oil_xyz_canonical,
            min(1.0, sample_t + dt_norm),
        )
        sample_pos = (oil_xyz_canonical + d_xyz_a).cpu().numpy().astype(np.float32)
        sample_vel = ((d_xyz_b - d_xyz_a) / dt_norm).cpu().numpy().astype(np.float32)

    g_capture = _estimate_gravity_dir(sample_vel)
    R = _rotation_aligning(g_capture, np.array([0.0, 0.0, -1.0]))

    # We compute scale + translation from the bounds across the entire
    # captured time window so the scaling is stable across frames (otherwise
    # later frames with bigger pool would scale differently than early
    # frames).  Sample 5 frames evenly to estimate bounds.
    sample_times = [i / 4.0 for i in range(5)]
    all_bounds_min = []
    all_bounds_max = []
    for st in sample_times:
        with torch.no_grad():
            d_xyz, _, _ = _query_deformation(deform, oil_xyz_canonical, st)
            pos_t = (oil_xyz_canonical + d_xyz).cpu().numpy().astype(np.float32)
            pos_rot = pos_t @ R.T
            all_bounds_min.append(pos_rot.min(axis=0))
            all_bounds_max.append(pos_rot.max(axis=0))
    bounds_min_global = np.min(all_bounds_min, axis=0)
    bounds_max_global = np.max(all_bounds_max, axis=0)
    extents = bounds_max_global - bounds_min_global
    longest = float(extents.max())
    if longest < 1e-9:
        print("ERROR: captured cloud has zero extent.", file=sys.stderr)
        return 4
    scale_factor = float(args.target_height / longest)
    print(f"      gravity_dir(capture)=[{g_capture[0]:+.3f}, {g_capture[1]:+.3f}, "
          f"{g_capture[2]:+.3f}]")
    print(f"      scale_factor={scale_factor:.6f}")
    # Translate so the bottom sits at z=0 and centre x/y after scaling.
    bbox_min_scl = bounds_min_global * scale_factor
    bbox_max_scl = bounds_max_global * scale_factor
    centre_xy = 0.5 * (bbox_min_scl + bbox_max_scl)
    translation = np.array([
        -centre_xy[0], -centre_xy[1], -bbox_min_scl[2],
    ], dtype=np.float32)

    rng = np.random.default_rng(args.rng_seed)

    n_frames_out = 0
    for f_idx in range(n_frames):
        t = f_idx / max(1, n_frames - 1)
        t_next = min(1.0, t + dt_norm)

        with torch.no_grad():
            d_xyz_t, _, _ = _query_deformation(deform, oil_xyz_canonical, t)
            d_xyz_tp, _, _ = _query_deformation(deform, oil_xyz_canonical, t_next)
            pos_t = (oil_xyz_canonical + d_xyz_t).cpu().numpy().astype(np.float32)
            vel_t = ((d_xyz_tp - d_xyz_t) / max(dt_norm, 1e-9)) \
                .cpu().numpy().astype(np.float32)

        # Apply rotation + scale + translation.
        pos_world = (pos_t @ R.T) * scale_factor + translation
        # Velocities scale with positions (per second) — same factor.
        vel_world = (vel_t @ R.T) * scale_factor

        bounds_min = pos_world.min(axis=0)
        bounds_max = pos_world.max(axis=0)
        grid_origin, (nx, ny, nz), pos_p, vel_p = _voxelise_and_seed(
            pos_world, vel_world,
            bounds_min, bounds_max,
            dx=args.dx,
            headroom_below=args.headroom_below,
            headroom_above=args.headroom_above,
            headroom_lateral=args.headroom_lateral,
            particles_per_cell=args.particles_per_cell,
            rng=rng,
            fill_mode=args.fill_mode,
            dilate_iters=args.dilate_iters,
        )
        out_path = args.output_dir / f"sim_state_{f_idx:04d}.bin"
        _write_sim_state_bin(
            out_path, grid_origin, nx, ny, nz, args.dx, pos_p, vel_p,
        )
        n_frames_out += 1
        if (f_idx + 1) % 25 == 0 or f_idx == n_frames - 1:
            print(f"      [{f_idx + 1:4d}/{n_frames}]  "
                  f"grid={nx}x{ny}x{nz}  particles={len(pos_p)}")

    print("[4/4] Writing manifest ...")
    manifest = {
        "version": 1,
        "n_frames": n_frames_out,
        "dx": args.dx,
        "target_height": args.target_height,
        "iteration": loaded_iter,
        "n_oil_gaussians": n_oil,
        "n_canonical_gaussians": n_gauss,
        "hsv_band": {
            "hue_min": args.hue_min, "hue_max": args.hue_max,
            "sat_min": args.sat_min, "val_min": args.val_min,
        },
    }
    import json
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"      {args.output_dir / 'manifest.json'}")

    print("-" * 72)
    print(f"  Done.  Wrote {n_frames_out} sim_state_NNNN.bin files to "
          f"{args.output_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
