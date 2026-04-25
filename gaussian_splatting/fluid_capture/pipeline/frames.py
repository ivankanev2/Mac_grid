"""Frame extraction from video + camera intrinsic estimation.

We don't have calibration metadata for arbitrary internet videos, so the
intrinsics are a heuristic: principal point at image centre, square
pixels, focal length derived from an assumed horizontal field of view
(default 60°).  This is good enough for a proof-of-concept — wrong
focal length scales the reconstructed point cloud uniformly, it does
not warp its topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class FrameSample:
    """One frame plus its timing metadata."""

    image: np.ndarray   # (H, W, 3) RGB uint8
    time: float         # seconds from video start
    frame_index: int    # 0-based frame number


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def extract_frame(video_path: str | Path, time_sec: float) -> FrameSample:
    """Read a single frame at ``time_sec`` from ``video_path``."""
    p = Path(video_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {p}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {p}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            raise IOError(f"Could not determine FPS for {p}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        target_idx = int(round(time_sec * fps))
        target_idx = max(0, min(total_frames - 1, target_idx)) if total_frames > 0 else max(0, target_idx)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise IOError(f"Failed to read frame at t={time_sec}s (index {target_idx}) from {p}")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return FrameSample(image=frame_rgb, time=target_idx / fps, frame_index=target_idx)
    finally:
        cap.release()


def extract_frame_pair(
    video_path: str | Path,
    time_sec: float,
    dt: float = 0.08,
) -> Tuple[FrameSample, FrameSample]:
    """Extract a pair of frames separated by ``dt`` seconds, suitable for optical flow.

    The pair is centred *before* ``time_sec`` rather than spanning it: the
    first frame is at ``time_sec`` (the hand-off moment), the second at
    ``time_sec + dt``.  Optical flow on this pair gives motion of fluid
    *as of* the hand-off moment, which is what we'll be feeding to the
    simulator in Milestone 2.
    """
    a = extract_frame(video_path, time_sec)
    b = extract_frame(video_path, time_sec + dt)
    return a, b


def estimate_intrinsics_default(
    width: int,
    height: int,
    fov_horizontal_deg: float = 60.0,
) -> CameraIntrinsics:
    """Heuristic pinhole intrinsics: principal point at centre, FOV-derived focal length.

    A 60° horizontal FOV is typical for "consumer camera in product video"
    framing.  If the video looks visually wider (super-wide-angle) or
    narrower (telephoto), pass a different ``fov_horizontal_deg`` — it
    only affects the absolute scale of the lifted point cloud.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")
    fov_rad = float(np.deg2rad(fov_horizontal_deg))
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # assume square pixels
    cx = width / 2.0
    cy = height / 2.0
    return CameraIntrinsics(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        width=int(width),
        height=int(height),
    )


def video_metadata(video_path: str | Path) -> dict:
    """Quick probe of video metadata — duration, fps, resolution, frame count."""
    p = Path(video_path).expanduser().resolve()
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {p}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return {
        "fps": fps,
        "n_frames": n_frames,
        "width": w,
        "height": h,
        "duration_sec": (n_frames / fps) if fps > 0 else 0.0,
    }
