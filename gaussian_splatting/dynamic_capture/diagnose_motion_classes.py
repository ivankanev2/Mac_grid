#!/usr/bin/env python3
"""diagnose_motion_classes.py — read-only diagnostic over the trained
Deformable 3D-GS model.

Purpose
-------
Before changing any extraction logic, we want to see what's actually inside
the model.  This script:

  1. Loads the same trained Deformable 3D-GS checkpoint that
     extract_fluid_state.py reads.
  2. Classifies every canonical Gaussian by its SH degree-0 colour into
     three buckets:
       - "yellow"  (hue 15-45, sat>=0.30, val>=0.20)  → oil candidates
       - "grey"    (sat<0.20, val>=0.30, not yellow)  → bottle/glass/table
       - "other"   (everything else)                   → background / noise
  3. Computes per-Gaussian max deformation velocity across N evenly
     sampled time stamps in [0, 1].
  4. Prints velocity histograms PER CLASS so we can see whether
     yellow Gaussians cleanly split into "fast column" vs "static pool",
     and whether grey Gaussians are overwhelmingly static (which would
     confirm they're the bottle).
  5. Dumps a binary-LE PLY per class with the canonical positions of
     that class's Gaussians, coloured by velocity magnitude (red=fast,
     green=static).  Open in MeshLab / CloudCompare to verify each
     class corresponds to the structure we expect.
  6. Writes a JSON summary with counts and velocity quantiles.

Does NOT modify the existing extraction pipeline or any of its outputs.

Usage
-----
    cd ~/Deformable-3D-Gaussians
    python dynamic_capture/diagnose_motion_classes.py \
        --model-dir   output/oil_pour \
        --iteration   40000 \
        --output-dir  dynamic_capture/diag_motion
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict

import numpy as np


# Make the Deformable-3D-Gaussians repo importable when this script lives in
# .../Deformable-3D-Gaussians/dynamic_capture/.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# SH(0,0) basis function constant — same value used in extract_fluid_state.py.
_SH_C0 = 0.28209479177387814


# -----------------------------------------------------------------------------
# Model loader (mirrors extract_fluid_state.py::_load_model)
# -----------------------------------------------------------------------------
def _load_model(model_dir: Path, iteration: int):
    """Return (gaussians, deform, loaded_iter)."""
    import torch  # noqa: F401
    from scene import Scene, GaussianModel
    from scene.deform_model import DeformModel

    cfg_path = model_dir / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"cfg_args not found in {model_dir} — is this a Deformable-3D-GS "
            "output directory?"
        )
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


# -----------------------------------------------------------------------------
# Colour classification
# -----------------------------------------------------------------------------
def _canonical_rgb(gaussians) -> np.ndarray:
    """Approximate diffuse RGB per Gaussian from the SH degree-0 component."""
    import torch
    with torch.no_grad():
        dc = gaussians._features_dc.squeeze(1)
        rgb = (dc * _SH_C0 + 0.5).clamp(0.0, 1.0).cpu().numpy()
    return rgb.astype(np.float32)


def _rgb_to_hsv_180(rgb: np.ndarray):
    """Convert (N,3) RGB in [0,1] to (hue_180, sat, val) — OpenCV's hue scale."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    delta = cmax - cmin

    hue = np.zeros_like(r)
    nz = delta > 1e-9
    rmax = nz & (cmax == r)
    gmax = nz & (cmax == g)
    bmax = nz & (cmax == b)
    hue[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6.0
    hue[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2.0
    hue[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4.0
    hue *= 60.0  # → degrees
    hue_180 = hue * 0.5  # OpenCV scale

    sat = np.where(cmax > 1e-9, delta / np.maximum(cmax, 1e-9), 0.0)
    val = cmax
    return hue_180, sat, val


def _classify(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Three-way classification — yellow, grey, other.  Disjoint."""
    h, s, v = _rgb_to_hsv_180(rgb)
    yellow = (h >= 15.0) & (h <= 45.0) & (s >= 0.30) & (v >= 0.20)
    grey   = (s < 0.20)  & (v >= 0.30) & ~yellow
    other  = ~(yellow | grey)
    return {"yellow": yellow, "grey": grey, "other": other}


# -----------------------------------------------------------------------------
# Motion (deformation MLP queries)
# -----------------------------------------------------------------------------
def _query_deform(deform, xyz_t, t):
    import torch
    N = xyz_t.shape[0]
    tin = torch.full((N, 1), float(t), dtype=xyz_t.dtype, device=xyz_t.device)
    with torch.no_grad():
        d_xyz, _, _ = deform.step(xyz_t.detach(), tin)
    return d_xyz


def _per_gaussian_max_vel(deform, xyz_t, n_samples: int) -> np.ndarray:
    """Per-Gaussian max ||Δxyz/Δt|| across n_samples evenly spaced time stamps.
    Units are canonical-units per normalised-time (same as the motion filter
    in extract_fluid_state.py)."""
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


# -----------------------------------------------------------------------------
# Histograms + PLY writer
# -----------------------------------------------------------------------------
_BIN_EDGES = np.array(
    [0.0, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00, 16.0, 1e9],
    dtype=np.float32,
)


def _ascii_hist(values: np.ndarray, title: str, bar_width: int = 50) -> dict:
    """Print a fixed-bin histogram and return a JSON-serialisable summary."""
    n = int(values.size)
    if n == 0:
        print(f"  {title:8s}  N=0  (empty)")
        return {"n": 0}
    counts, _ = np.histogram(values, bins=_BIN_EDGES)
    peak = max(int(counts.max()), 1)
    print(f"  {title:8s}  N={n:6d}  "
          f"min={values.min():.4f}  med={np.median(values):.4f}  "
          f"p90={np.percentile(values, 90):.4f}  "
          f"p99={np.percentile(values, 99):.4f}  "
          f"max={values.max():.4f}")
    for i in range(len(counts)):
        lo = _BIN_EDGES[i]
        hi = _BIN_EDGES[i + 1]
        hi_s = "+inf" if hi >= 1e8 else f"{hi:6.3f}"
        bar = "#" * int(bar_width * counts[i] / peak)
        print(f"      [{lo:6.3f}, {hi_s})  {int(counts[i]):6d}  {bar}")
    return {
        "n": n,
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "bins":  [float(x) for x in _BIN_EDGES],
        "counts": [int(c) for c in counts],
    }


def _write_ply_velcolored(path: Path, xyz: np.ndarray,
                          vel_mag: np.ndarray, cap: float) -> None:
    """Write a binary-LE PLY of (xyz, RGB-by-velocity).
    Red channel saturates at `cap`; green decays inversely; blue constant.
    No external deps — uses a packed numpy structured dtype."""
    n = xyz.shape[0]
    if n == 0:
        return
    t = np.clip(vel_mag.astype(np.float32) / max(float(cap), 1e-9), 0.0, 1.0)
    r = (t * 255.0).astype(np.uint8)
    g = ((1.0 - t) * 200.0).astype(np.uint8)
    b = np.full(n, 60, dtype=np.uint8)

    # Packed dtype — numpy default packs fields tightly when align is False.
    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    assert dt.itemsize == 15, f"expected packed 15-byte vertex, got {dt.itemsize}"
    rec = np.empty(n, dtype=dt)
    rec["x"] = xyz[:, 0].astype("<f4")
    rec["y"] = xyz[:, 1].astype("<f4")
    rec["z"] = xyz[:, 2].astype("<f4")
    rec["r"] = r
    rec["g"] = g
    rec["b"] = b

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
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only diagnostic — bin canonical Gaussians by "
                    "colour class and motion magnitude.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Trained Deformable 3D-GS output directory.")
    p.add_argument("--iteration", type=int, default=-1,
                   help="Checkpoint iteration to load. -1 = latest.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write per-class PLYs and the JSON summary.")
    p.add_argument("--motion-samples", type=int, default=10,
                   help="Number of evenly-spaced time stamps for the "
                        "per-Gaussian max-velocity computation.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  diagnose_motion_classes.py — read-only Gaussian classifier")
    print("=" * 72)
    print(f"  Model dir:       {args.model_dir}")
    print(f"  Iteration:       {'latest' if args.iteration < 0 else args.iteration}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Motion samples:  {args.motion_samples}")
    print("-" * 72)

    print("[1/4] Loading trained Deformable 3D-GS model ...")
    gaussians, deform, loaded_iter = _load_model(args.model_dir, args.iteration)
    import torch
    canonical = gaussians.get_xyz.detach()
    n_total = int(canonical.shape[0])
    print(f"      iteration={loaded_iter}  N_canonical={n_total}")

    print("[2/4] Classifying canonical Gaussians by HSV colour ...")
    rgb = _canonical_rgb(gaussians)
    classes = _classify(rgb)
    print(f"      {'yellow':8s}: {int(classes['yellow'].sum()):7d}  "
          f"({100.0 * classes['yellow'].sum() / max(1, n_total):5.2f}%)  "
          "(oil candidates)")
    print(f"      {'grey':8s}: {int(classes['grey'].sum()):7d}  "
          f"({100.0 * classes['grey'].sum() / max(1, n_total):5.2f}%)  "
          "(bottle/glass/table candidates)")
    print(f"      {'other':8s}: {int(classes['other'].sum()):7d}  "
          f"({100.0 * classes['other'].sum() / max(1, n_total):5.2f}%)  "
          "(background / noise)")

    print("[3/4] Sampling deformation MLP per Gaussian ...")
    all_vmag = _per_gaussian_max_vel(deform, canonical, args.motion_samples)
    print(f"      done — N_v_samples={int(all_vmag.size)}")

    print("\n=== Velocity histograms (canonical units / normalised-time) ===")
    hist_summary = {}
    hist_summary["all"] = _ascii_hist(all_vmag, title="ALL")
    print()
    for cls in ("yellow", "grey", "other"):
        hist_summary[cls] = _ascii_hist(all_vmag[classes[cls]], title=cls)
        print()

    print("[4/4] Writing per-class PLYs + JSON summary ...")
    cano_np = canonical.cpu().numpy()
    cap = float(np.percentile(all_vmag, 99)) if all_vmag.size else 1.0
    if cap < 1e-6:
        cap = 1.0
    for cls in ("yellow", "grey", "other"):
        out = args.output_dir / f"class_{cls}.ply"
        mask = classes[cls]
        _write_ply_velcolored(out, cano_np[mask], all_vmag[mask], cap)
        print(f"      {out}  (N={int(mask.sum())}, cap={cap:.4f})")

    # Combined PLY for context — colour by class, not by velocity, so you
    # can see the three classes overlaid in 3D space.
    combined_path = args.output_dir / "class_combined.ply"
    n = cano_np.shape[0]
    rgb_class = np.zeros((n, 3), dtype=np.uint8)
    rgb_class[classes["yellow"]] = (220, 200,  60)   # yellow
    rgb_class[classes["grey"]]   = (200, 200, 210)   # light grey
    rgb_class[classes["other"]]  = (120,  60, 160)   # purple
    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    rec = np.empty(n, dtype=dt)
    rec["x"] = cano_np[:, 0].astype("<f4")
    rec["y"] = cano_np[:, 1].astype("<f4")
    rec["z"] = cano_np[:, 2].astype("<f4")
    rec["r"] = rgb_class[:, 0]
    rec["g"] = rgb_class[:, 1]
    rec["b"] = rgb_class[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with combined_path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())
    print(f"      {combined_path}  (yellow=oil, grey=bottle, purple=other)")

    summary = {
        "iteration": int(loaded_iter),
        "n_canonical": n_total,
        "motion_samples": int(args.motion_samples),
        "counts": {k: int(m.sum()) for k, m in classes.items()},
        "histograms": hist_summary,
    }
    summary_path = args.output_dir / "diag_motion.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"      {summary_path}")

    print("-" * 72)
    print("  Done.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
