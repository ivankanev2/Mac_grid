#!/usr/bin/env python3
"""Diagnostic — dump canonical Gaussian point clouds from a trained Deformable
3D-GS model so we can visually inspect what the model actually contains.

The motivation: when our simulator-side replay shows a brick-like blob instead
of a recognisable oil pour, *several* things could be the culprit, and they
need different fixes.  Looking at the underlying Gaussians directly tells us
which one it actually is.

Three PLYs are written:

    1. canonical_all.ply
        Every Gaussian in the model, in its canonical-frame position,
        coloured by canonical RGB (SH degree-0 component).  This shows the
        underlying 3D geometry the model learned.

        - If this is a recognisable 3D bottle shape with internal structure
          ⇒ model has real 3D understanding.  Failure is downstream of this.
        - If this is a flat sheet (single plane of points perpendicular to
          the training camera) ⇒ model is "cardboard" — it overfit the
          static-camera input and has no real 3D structure.  No
          per-Gaussian extraction can work in that case.

    2. canonical_hsv_yellow.ply
        Gaussians passing the HSV-yellow colour filter only.  Same as the
        intermediate result inside extract_fluid_state.py before motion
        filtering.

        - If this is a column + pool shape ⇒ HSV is good, problem is
          downstream.
        - If this is a whole-bottle outline ⇒ HSV is grabbing oil-tinted
          glass Gaussians, motion filter is the right next step.

    3. canonical_filtered.ply
        Gaussians passing HSV + motion filter (the set extract_fluid_state.py
        currently feeds the simulator).

        - If this is column + pool ⇒ extraction is fine; the
          voxelisation / vertical-fill / replay step is the failure
          point.
        - If this is still a whole-bottle outline ⇒ motion filter isn't
          discriminating enough; we need a smarter filter (or the
          render-based approach).

Open each PLY in Open3D / MeshLab / any PLY viewer.  The combination of what
each shape looks like uniquely identifies which step is failing.

Run from inside the Deformable-3D-Gaussians repo, same way as
extract_fluid_state.py:

    python dynamic_capture/inspect_geometry.py \\
        --model-dir   output/oil_pour \\
        --output-dir  verify_geometry
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Make the Deformable-3D-Gaussians package importable when this script lives
# in .../Deformable-3D-Gaussians/dynamic_capture/.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_SH_C0 = 0.28209479177387814


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dump canonical Gaussian PLYs from a trained "
                    "Deformable 3D-GS model for visual diagnosis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Trained Deformable 3D-GS output directory.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write the diagnostic PLYs.")
    p.add_argument("--iteration", type=int, default=-1,
                   help="Checkpoint iteration to load. -1 = latest.")
    p.add_argument("--hue-min", type=float, default=15.0)
    p.add_argument("--hue-max", type=float, default=45.0)
    p.add_argument("--sat-min", type=float, default=0.30)
    p.add_argument("--val-min", type=float, default=0.20)
    p.add_argument("--motion-threshold", type=float, default=3.0)
    p.add_argument("--motion-samples", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def _load_model(model_dir: Path, iteration: int):
    import torch  # noqa: F401
    from scene import Scene, GaussianModel
    from scene.deform_model import DeformModel

    cfg_text = (model_dir / "cfg_args").read_text(encoding="utf-8").strip()
    from argparse import Namespace
    ns = eval(cfg_text, {"Namespace": Namespace, "__builtins__": {}})
    cfg_dict = vars(ns)

    sh_degree = int(cfg_dict.get("sh_degree", 3))
    is_blender = bool(cfg_dict.get("is_blender", True))
    is_6dof = bool(cfg_dict.get("is_6dof", False))

    gaussians = GaussianModel(sh_degree)

    class _MArgs:
        pass
    margs = _MArgs()
    margs.sh_degree = sh_degree
    margs.source_path = cfg_dict.get("source_path", "")
    margs.model_path = str(model_dir)
    margs.images = cfg_dict.get("images", "images")
    margs.resolution = int(cfg_dict.get("resolution", -1))
    margs.white_background = bool(cfg_dict.get("white_background", False))
    margs.data_device = "cuda"
    margs.eval = True
    margs.is_blender = is_blender
    margs.is_6dof = is_6dof
    margs.render_process = False
    margs.load2gpu_on_the_fly = bool(cfg_dict.get("load2gpu_on_the_fly", False))

    scene = Scene(margs, gaussians,
                  load_iteration=iteration if iteration > 0 else -1,
                  shuffle=False)

    deform = DeformModel(is_blender=is_blender, is_6dof=is_6dof)
    deform.load_weights(str(model_dir), iteration=scene.loaded_iter)
    return gaussians, deform, scene.loaded_iter


def _canonical_rgb(gaussians) -> np.ndarray:
    import torch
    with torch.no_grad():
        dc = gaussians._features_dc.squeeze(1)
        rgb = (dc * _SH_C0 + 0.5).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    return rgb


def _hsv_yellow_mask(rgb_01, hue_min, hue_max, sat_min, val_min) -> np.ndarray:
    r, g, b = rgb_01[:, 0], rgb_01[:, 1], rgb_01[:, 2]
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
    return ((hue_180 >= hue_min) & (hue_180 <= hue_max) &
            (sat >= sat_min) & (val >= val_min))


def _query_deformation(deform, canonical_xyz_t, time_t):
    import torch
    N = canonical_xyz_t.shape[0]
    time_input = torch.full((N, 1), float(time_t),
                            dtype=canonical_xyz_t.dtype,
                            device=canonical_xyz_t.device)
    with torch.no_grad():
        d_xyz, d_rot, d_scale = deform.step(canonical_xyz_t.detach(), time_input)
    return d_xyz, d_rot, d_scale


def _save_ply_xyz_rgb(path: Path, xyz: np.ndarray, rgb_01: np.ndarray) -> None:
    """Minimal binary-LE PLY writer with vertex xyz + uchar rgb.  No external
    dependency beyond numpy — keeps this script open3d-free so it runs even
    when only the Deformable 3D-GS env is available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = xyz.shape[0]
    rgb_uchar = (np.clip(rgb_01, 0.0, 1.0) * 255.0).astype(np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    # Build interleaved payload: each vertex is 3*float32 + 3*uint8 = 15 bytes
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    buf = np.empty(n, dtype=dtype)
    buf["x"] = xyz[:, 0].astype(np.float32)
    buf["y"] = xyz[:, 1].astype(np.float32)
    buf["z"] = xyz[:, 2].astype(np.float32)
    buf["red"]   = rgb_uchar[:, 0]
    buf["green"] = rgb_uchar[:, 1]
    buf["blue"]  = rgb_uchar[:, 2]

    with path.open("wb") as f:
        f.write(header)
        f.write(buf.tobytes())


def main(argv=None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Geometry diagnostic — dump Gaussian point clouds at filter levels")
    print("=" * 72)
    print(f"  Model dir:    {args.model_dir}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  HSV band:     hue [{args.hue_min}-{args.hue_max}]  "
          f"sat≥{args.sat_min}  val≥{args.val_min}")
    print(f"  Motion thr:   {args.motion_threshold}")
    print("-" * 72)

    print("[1/4] Loading model ...")
    import torch
    gaussians, deform, loaded_iter = _load_model(args.model_dir, args.iteration)
    canonical_xyz = gaussians.get_xyz.detach()
    n_total = canonical_xyz.shape[0]
    print(f"      iter={loaded_iter}  Gaussians={n_total}")

    print("[2/4] Writing PLY 1: all canonical Gaussians ...")
    rgb = _canonical_rgb(gaussians)
    out1 = args.output_dir / "canonical_all.ply"
    _save_ply_xyz_rgb(out1, canonical_xyz.cpu().numpy(), rgb)
    print(f"      {out1}  ({n_total} points)")

    print("[3/4] Writing PLY 2: HSV-yellow Gaussians ...")
    hsv_mask = _hsv_yellow_mask(
        rgb, args.hue_min, args.hue_max, args.sat_min, args.val_min,
    )
    n_hsv = int(hsv_mask.sum())
    xyz_np = canonical_xyz.cpu().numpy()
    out2 = args.output_dir / "canonical_hsv_yellow.ply"
    _save_ply_xyz_rgb(out2, xyz_np[hsv_mask], rgb[hsv_mask])
    print(f"      {out2}  ({n_hsv} points)")

    print("[4/4] Writing PLY 3: HSV + motion filtered Gaussians ...")
    hsv_xyz = canonical_xyz[torch.from_numpy(hsv_mask).to(canonical_xyz.device)]
    sample_ts = np.linspace(0.0, 1.0, max(2, args.motion_samples + 1))[:-1]
    dt_motion = 1.0 / max(2, args.motion_samples + 1)
    max_vel = torch.zeros(hsv_xyz.shape[0], device=hsv_xyz.device)
    for st in sample_ts:
        with torch.no_grad():
            d_a, _, _ = _query_deformation(deform, hsv_xyz, float(st))
            d_b, _, _ = _query_deformation(deform, hsv_xyz, float(st + dt_motion))
            vmag = torch.linalg.norm((d_b - d_a) / dt_motion, dim=1)
            max_vel = torch.maximum(max_vel, vmag)
    motion_mask_local = (max_vel > float(args.motion_threshold)).cpu().numpy()
    final_xyz = xyz_np[hsv_mask][motion_mask_local]
    final_rgb = rgb[hsv_mask][motion_mask_local]
    out3 = args.output_dir / "canonical_filtered.ply"
    _save_ply_xyz_rgb(out3, final_xyz, final_rgb)
    print(f"      {out3}  ({len(final_xyz)} points)")

    print("-" * 72)
    print("  Done.  Open the three PLYs in Open3D / MeshLab / any PLY viewer.")
    print()
    print("  What to check:")
    print("    canonical_all.ply        — does this look like a 3D bottle?")
    print("                               (orbit around it; if you can see it")
    print("                                from multiple angles ⇒ real 3D)")
    print("    canonical_hsv_yellow.ply — column + pool, or whole-bottle outline?")
    print("    canonical_filtered.ply   — same question, after motion filter.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
