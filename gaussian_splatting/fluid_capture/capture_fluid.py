#!/usr/bin/env python3
"""Capture a 3D fluid state from a single video clip — Milestone 1.

Pipeline:

    extract frame pair → predict depth (×2) → predict optical flow →
    HSV-segment the fluid → lift to 3D points + per-point velocities →
    save PLYs and debug PNGs.

Example
-------
    python capture_fluid.py \\
        --input  input/videoplayback-3 \\
        --output-dir outputs/ \\
        --time   4.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# MPS sometimes hits unimplemented ops; let PyTorch fall back silently to CPU
# rather than crash.  Set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

from pipeline import (
    debug_viz,
    depth as depth_mod,
    flow as flow_mod,
    frames,
    lift,
    segmentation,
)


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture a 3D fluid state from monocular video (Milestone 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Input video path.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write PLYs, .npz state, and debug PNGs.")
    p.add_argument("--time", type=float, default=4.0,
                   help="Hand-off timestamp (seconds from video start).")
    p.add_argument("--dt", type=float, default=0.08,
                   help="Time gap (s) between the frame pair fed to optical flow.")
    p.add_argument("--fov", type=float, default=60.0,
                   help="Assumed horizontal FOV (degrees) for intrinsics.")
    p.add_argument("--hue-min", type=int, default=15,
                   help="HSV hue lower bound for fluid segmentation (0–180).")
    p.add_argument("--hue-max", type=int, default=45,
                   help="HSV hue upper bound for fluid segmentation (0–180).")
    p.add_argument("--sat-min", type=int, default=60,
                   help="HSV saturation lower bound (0–255).")
    p.add_argument("--val-min", type=int, default=60,
                   help="HSV value lower bound (0–255).")
    p.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Base-hf",
                   help="HuggingFace id for the Depth Anything V2 variant.")
    p.add_argument("--device", default=None,
                   help="Torch device: mps / cuda / cpu.  Auto if unset.")
    p.add_argument("--velocity-clip-pct", type=float, default=99.0,
                   help="Clip top (100 − this) percent of velocity magnitudes as outliers.")
    return p.parse_args(argv)


def _print_header(args: argparse.Namespace, meta: dict) -> None:
    print("=" * 72)
    print("  Fluid capture (Milestone 1: video → 3D fluid state)")
    print("=" * 72)
    print(f"  Input:        {args.input}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Resolution:   {meta['width']} × {meta['height']}")
    print(f"  FPS:          {meta['fps']:.2f}    Frames: {meta['n_frames']}    "
          f"Duration: {meta['duration_sec']:.2f} s")
    print(f"  Hand-off t:   {args.time:.3f} s")
    print(f"  Frame gap dt: {args.dt:.3f} s")
    print(f"  FOV (assumed):{args.fov:.1f}°")
    print(f"  HSV mask:     hue [{args.hue_min}-{args.hue_max}]  "
          f"sat≥{args.sat_min}  val≥{args.val_min}")
    print(f"  Depth model:  {args.depth_model}")
    print(f"  Device:       {args.device or '(auto)'}")
    print("-" * 72)


class _Stage:
    """Tiny context-manager that prints stage timing."""
    def __init__(self, label: str):
        self.label = label
        self.t0 = 0.0
    def __enter__(self):
        print(self.label, end="", flush=True)
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        print(f"   ({dt:.2f}s)")
        return False


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        print(f"ERROR: input video not found: {args.input}", file=sys.stderr)
        return 2

    meta = frames.video_metadata(args.input)
    if meta["fps"] <= 0:
        print(f"ERROR: could not determine FPS for {args.input}", file=sys.stderr)
        return 2
    _print_header(args, meta)

    if args.time < 0 or args.time + args.dt > meta["duration_sec"]:
        print(
            f"ERROR: --time {args.time} + --dt {args.dt} = "
            f"{args.time + args.dt:.3f}s exceeds duration {meta['duration_sec']:.3f}s",
            file=sys.stderr,
        )
        return 3

    # --- 1. Extract frame pair ----------------------------------------------
    with _Stage("[1/6] Extracting frame pair ..."):
        frame_a, frame_b = frames.extract_frame_pair(args.input, args.time, args.dt)
        K = frames.estimate_intrinsics_default(
            frame_a.image.shape[1], frame_a.image.shape[0], fov_horizontal_deg=args.fov
        )

    # --- 2. Depth on both frames --------------------------------------------
    with _Stage(f"[2/6] Loading depth model and predicting depth ..."):
        depth_estimator = depth_mod.DepthEstimator(
            model_name=args.depth_model, device=args.device,
        )
        depth_a = depth_estimator.predict(frame_a.image)
        depth_b = depth_estimator.predict(frame_b.image)

    # Save depth heatmap of the hand-off frame for sanity check.
    debug_viz.save_depth_heatmap(depth_a, debug_dir / "depth_a.png")
    debug_viz.save_depth_heatmap(depth_b, debug_dir / "depth_b.png")

    # --- 3. Optical flow ----------------------------------------------------
    with _Stage("[3/6] Loading flow model and predicting flow ..."):
        flow_estimator = flow_mod.FlowEstimator(device=args.device)
        flow_ab = flow_estimator.predict(frame_a.image, frame_b.image)

    debug_viz.save_flow_hsv(flow_ab, debug_dir / "flow_ab.png")

    # --- 4. Segmentation ----------------------------------------------------
    with _Stage("[4/6] Segmenting fluid (HSV threshold) ..."):
        mask = segmentation.mask_yellow_fluid(
            frame_a.image,
            hue_min=args.hue_min, hue_max=args.hue_max,
            sat_min=args.sat_min, val_min=args.val_min,
        )
        n_fluid = int(mask.sum())
    print(f"      fluid pixels: {n_fluid} / {mask.size}  "
          f"({100.0 * n_fluid / max(mask.size, 1):.1f}%)")
    if n_fluid < 200:
        print(
            "      WARNING: fewer than 200 fluid pixels selected — "
            "the HSV thresholds may be off for this clip.",
            file=sys.stderr,
        )
    segmentation.save_mask_overlay(frame_a.image, mask, debug_dir / "mask_overlay.png")

    # --- 5. Lift to 3D ------------------------------------------------------
    with _Stage("[5/6] Lifting to 3D fluid state ..."):
        state = lift.lift_to_fluid_state(
            image_a_rgb=frame_a.image,
            depth_a=depth_a, depth_b=depth_b,
            flow_ab=flow_ab,
            mask=mask, intrinsics=K,
            dt=args.dt,
            velocity_clip_pct=args.velocity_clip_pct,
        )
    print(f"      lifted points: {len(state.points)}")
    if len(state.points) == 0:
        print("ERROR: no points survived the lift — check segmentation and depth.",
              file=sys.stderr)
        return 4

    vmag = np.linalg.norm(state.velocities, axis=1)
    print(
        f"      velocity stats: mean={vmag.mean():.4f}  "
        f"median={np.median(vmag):.4f}  max={vmag.max():.4f}  "
        f"(units: depth-units / s)"
    )

    # --- 6. Save outputs ----------------------------------------------------
    with _Stage("[6/6] Writing PLYs and state .npz ..."):
        rgb_path = args.output_dir / "fluid_points_rgb.ply"
        vmag_path = args.output_dir / "fluid_points_vmag.ply"
        npz_path = args.output_dir / "fluid_state.npz"
        lift.save_rgb_point_cloud(rgb_path, state)
        lift.save_velocity_magnitude_point_cloud(vmag_path, state)
        lift.save_state_npz(npz_path, state)

    print(f"      {rgb_path}")
    print(f"      {vmag_path}")
    print(f"      {npz_path}")
    print(f"      debug PNGs in {debug_dir}")
    print("-" * 72)
    print("  Done.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
