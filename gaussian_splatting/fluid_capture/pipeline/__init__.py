"""Monocular video → 3D fluid state reconstruction (Milestone 1).

Submodules:
    frames        — extract video frames + estimate camera intrinsics
    depth         — Depth Anything V2 wrapper (MPS / CUDA / CPU)
    flow          — RAFT optical flow wrapper (MPS / CUDA / CPU)
    segmentation  — HSV-based fluid segmentation (yellow olive oil default)
    lift          — depth + flow + intrinsics → 3D points + per-point velocities
    debug_viz     — save per-stage PNGs (depth heatmap, flow visualisation, mask)
"""

__all__ = [
    "frames",
    "depth",
    "flow",
    "segmentation",
    "lift",
    "debug_viz",
]
