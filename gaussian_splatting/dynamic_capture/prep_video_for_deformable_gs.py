#!/usr/bin/env python3
"""Convert a static-camera monocular video into a Deformable 3D-GS dataset.

Produces the D-NeRF / Blender directory layout that the upstream
``--is_blender`` data loader consumes:

    data/<scene>/
        transforms_train.json
        transforms_val.json
        transforms_test.json
        train/r_<i>.png
        val/r_<i>.png
        test/r_<i>.png

Each ``transforms_*.json`` looks like:

    {
        "camera_angle_x": <horizontal FOV in radians>,
        "frames": [
            {
                "file_path": "./train/r_0",
                "time": <float in [0, 1]>,
                "rotation": 0.0,
                "transform_matrix": [[...], [...], [...], [...]]   # 4x4 c2w
            },
            ...
        ]
    }

For our static-camera videos every frame uses the **same** ``transform_matrix``
(camera at ``(0, 0, --camera-distance)`` looking at the origin) and only the
``time`` field varies.  The deformation MLP that Deformable 3D-GS trains is
the thing that explains all temporal variation; the camera contributes no
extra signal.

Frames are split sequentially: first ``train_frac`` of them go to ``train/``,
next ``val_frac`` to ``val/``, the remainder to ``test/``.  The ``time``
field always reflects each frame's position in the **original** video,
normalised to ``[0, 1]`` — *not* its index within the split.

Usage
-----
    python prep_video_for_deformable_gs.py \\
        --input      /path/to/videoplayback-3 \\
        --output-dir data/oil_pour/ \\
        --fov        60.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare a static-camera video as a Deformable 3D-GS dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Input video path.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Output dataset directory (will be created).")
    p.add_argument("--fov", type=float, default=60.0,
                   help="Assumed horizontal field of view in degrees.")
    p.add_argument("--camera-distance", type=float, default=4.0,
                   help="Distance from camera to origin along +Z (for the static pose).")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Cap the number of frames extracted (0 = all).")
    p.add_argument("--train-frac", type=float, default=0.80,
                   help="Fraction of frames assigned to train split (sequential, from start).")
    p.add_argument("--val-frac", type=float, default=0.10,
                   help="Fraction of frames assigned to val split (next sequentially).")
    return p.parse_args(argv)


def static_camera_transform(distance: float) -> np.ndarray:
    """4x4 camera-to-world transform: camera at ``(0, 0, distance)`` looking at origin.

    Matches the standard D-NeRF / Blender convention of right-handed coords
    with +Y up and the camera looking down -Z.  With ``distance > 0`` the
    camera sits in front of the origin on the +Z axis and sees the scene
    centred at world origin.
    """
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = float(distance)
    return T


def extract_and_split(
    video_path: Path,
    output_dir: Path,
    max_frames: int,
    train_frac: float,
    val_frac: float,
) -> Dict:
    """Extract frames, write to ``train`` / ``val`` / ``test`` PNGs, return metadata.

    Returns a dict with:
        ``frame_paths`` — dict ``{"train": [...], "val": [...], "test": [...]}``
            where each entry is ``(split_local_index, original_frame_index)``.
        ``n_total`` — total frames extracted.
        ``fps``, ``width``, ``height`` — video metadata.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n_video <= 0 or fps <= 0:
        cap.release()
        raise IOError(
            f"Invalid video metadata for {video_path}: fps={fps}, frames={n_video}"
        )
    n_total = n_video if max_frames <= 0 else min(n_video, max_frames)

    n_train = max(1, int(round(n_total * train_frac)))
    n_val = max(1, int(round(n_total * val_frac)))
    n_test = max(1, n_total - n_train - n_val)
    # Guard against rounding edge cases.
    if n_train + n_val + n_test != n_total:
        n_test = max(1, n_total - n_train - n_val)

    print(f"      total={n_total}  split: train={n_train}  val={n_val}  test={n_test}")

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    for d in (train_dir, val_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    frame_paths: Dict[str, List[Tuple[int, int]]] = {
        "train": [], "val": [], "test": [],
    }
    width = height = 0

    for orig_idx in range(n_total):
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            print(
                f"WARNING: failed to read frame {orig_idx}; truncating dataset",
                file=sys.stderr,
            )
            break
        if width == 0:
            height, width = frame_bgr.shape[:2]

        if orig_idx < n_train:
            split = "train"
            split_idx = orig_idx
            out_dir = train_dir
        elif orig_idx < n_train + n_val:
            split = "val"
            split_idx = orig_idx - n_train
            out_dir = val_dir
        else:
            split = "test"
            split_idx = orig_idx - n_train - n_val
            out_dir = test_dir

        out_path = out_dir / f"r_{split_idx}.png"
        cv2.imwrite(str(out_path), frame_bgr)
        frame_paths[split].append((split_idx, orig_idx))

    cap.release()
    return {
        "frame_paths": frame_paths,
        "n_total": n_total,
        "fps": fps,
        "width": width,
        "height": height,
    }


def write_transforms_json(
    out_path: Path,
    split: str,
    frame_entries: List[Tuple[int, int]],
    n_total: int,
    camera_angle_x: float,
    transform_matrix: np.ndarray,
) -> None:
    """Write a single ``transforms_<split>.json`` in D-NeRF / Blender format."""
    frames = []
    denom = max(1, n_total - 1)
    for split_idx, orig_idx in frame_entries:
        time = orig_idx / denom
        frames.append({
            "file_path": f"./{split}/r_{split_idx}",
            "time": float(time),
            "rotation": 0.0,
            "transform_matrix": transform_matrix.tolist(),
        })

    payload = {
        "camera_angle_x": float(camera_angle_x),
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _print_header(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("  Prep video for Deformable 3D-GS (--is_blender format)")
    print("=" * 72)
    print(f"  Input:        {args.input}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  FOV:          {args.fov:.1f}°")
    print(f"  Camera dist:  {args.camera_distance:.2f}")
    print(f"  train/val:    {args.train_frac:.2f} / {args.val_frac:.2f} "
          f"(test = {1.0 - args.train_frac - args.val_frac:.2f})")
    if args.max_frames > 0:
        print(f"  Max frames:   {args.max_frames}")
    print("-" * 72)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input video not found: {args.input}", file=sys.stderr)
        return 2
    if not (0.0 < args.train_frac < 1.0):
        print(f"ERROR: --train-frac must be in (0, 1); got {args.train_frac}",
              file=sys.stderr)
        return 3
    if not (0.0 < args.val_frac < 1.0):
        print(f"ERROR: --val-frac must be in (0, 1); got {args.val_frac}",
              file=sys.stderr)
        return 3
    if args.train_frac + args.val_frac >= 1.0:
        print("ERROR: --train-frac + --val-frac must be < 1.0 "
              "(test set must be non-empty)", file=sys.stderr)
        return 3

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _print_header(args)

    print("[1/2] Extracting frames + splitting train/val/test ...")
    metadata = extract_and_split(
        args.input, args.output_dir, args.max_frames,
        args.train_frac, args.val_frac,
    )
    print(f"      fps={metadata['fps']:.2f}  "
          f"resolution={metadata['width']}x{metadata['height']}")

    print("[2/2] Writing transforms_*.json ...")
    camera_angle_x = float(np.deg2rad(args.fov))
    transform_matrix = static_camera_transform(args.camera_distance)

    for split in ("train", "val", "test"):
        out_json = args.output_dir / f"transforms_{split}.json"
        write_transforms_json(
            out_json, split,
            metadata["frame_paths"][split],
            metadata["n_total"],
            camera_angle_x, transform_matrix,
        )
        n = len(metadata["frame_paths"][split])
        print(f"      {out_json}  ({n} frames)")

    print("-" * 72)
    print("  Done.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
