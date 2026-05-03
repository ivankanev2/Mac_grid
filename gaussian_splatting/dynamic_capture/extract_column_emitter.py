#!/usr/bin/env python3
"""extract_column_emitter.py — Tier B: per-frame column trajectory.

Reads the captured_states_v2/*.bin time series produced by
extract_fluid_state_v2.py and computes, per frame, the parameters of a
continuous-emission water source that approximates the falling oil column.
Writes a small binary file the simulator loads to drive its emitter over time.

For each captured frame:
  - filter to "column" particles: those whose z is above the bottle's top
  - if there are enough column particles to be a real pour, mark the frame
    ACTIVE and record:
      pos    = mean xy of column particles, z = bottle_top + small offset
      vel    = mean velocity of column particles
      radius = robust spread of column particles in xy (95th percentile of
               distance from mean)
  - otherwise mark INACTIVE (no emission this frame)

The simulator advances its internal time at the captured FPS rate.  Each sim
step it looks up the current captured frame, and if active, emits a sphere
of water with the recorded parameters.  Result: the simulated pour starts /
stops / shifts position over time matching the source video's actual pour.

Pure CPU.  Reads only existing .bin files — no GPU, no model query, no
torch import.  Runs in ~5 seconds for 501 frames.

Usage
-----
    python dynamic_capture/extract_column_emitter.py \\
        --series-folder dynamic_capture/captured_states_v2 \\
        --output        dynamic_capture/captured_states_v2/column_emitter.bin

Binary format
-------------
    uint32  magic        = 0x43454D31  // 'CEM1'
    uint32  version      = 1
    uint32  n_frames
    float32 captured_fps
    [n_frames *]:
        uint8   active       (0 = no emission, 1 = emit)
        uint8   _pad[3]
        float32 pos_x, pos_y, pos_z
        float32 vel_x, vel_y, vel_z
        float32 radius
        float32 amount
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


# Binary format constants — mirror the C++ reader in fluid_state_loader.cpp.
_FST_MAGIC   = 0x46535431   # 'FST1' — captured per-frame fluid state
_FST_VERSION = 1
_CEM_MAGIC   = 0x43454D31   # 'CEM1' — captured column emitter
_CEM_VERSION = 1


# -----------------------------------------------------------------------------
# Sim state reader (matches state_writer.py / extract_fluid_state_v2.py output)
# -----------------------------------------------------------------------------
def _load_sim_state_bin(path: Path):
    with path.open("rb") as f:
        data = f.read()
    if len(data) < 5*4 + 4*4 + 4:
        raise ValueError(f"file too small: {path}")
    magic, ver, nx, ny, nz, dx, ox, oy, oz, n_p = struct.unpack_from(
        "<IIIII fff fI", data, 0,
    )
    if magic != _FST_MAGIC:
        raise ValueError(f"bad magic 0x{magic:08X} in {path}")
    if ver != _FST_VERSION:
        raise ValueError(f"unsupported sim_state version {ver} in {path}")
    header_size = 5*4 + 4*4 + 4
    body = np.frombuffer(data, dtype=np.float32, offset=header_size)
    if body.size != 6 * n_p:
        raise ValueError(
            f"size mismatch in {path}: expected {6*n_p} floats, got {body.size}"
        )
    body = body.reshape(-1, 6)
    return {
        "nx": int(nx), "ny": int(ny), "nz": int(nz), "dx": float(dx),
        "origin": np.array([ox, oy, oz], dtype=np.float32),
        "pos":    body[:, 0:3].copy(),
        "vel":    body[:, 3:6].copy(),
    }


# -----------------------------------------------------------------------------
# Manifest helper — pulls bottle_height_target so we know the column z floor
# -----------------------------------------------------------------------------
def _load_manifest(folder: Path) -> dict:
    p = folder / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# Per-frame column statistics
# -----------------------------------------------------------------------------
def _column_stats_for_frame(pos: np.ndarray, vel: np.ndarray,
                             column_z_floor: float,
                             min_active: int):
    """Return dict {active, pos, vel, radius} for a single captured frame."""
    column_mask = pos[:, 2] > column_z_floor
    n_col = int(column_mask.sum())

    if n_col < min_active:
        return {
            "active": False,
            "pos":    np.array([0.0, 0.0, column_z_floor], dtype=np.float32),
            "vel":    np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "radius": 0.012,
            "n_column": n_col,
        }

    col_pos = pos[column_mask]
    col_vel = vel[column_mask]

    # Centre xy = mean of column points; z = floor + half-cell so the source
    # sphere sits just above the bottle's mouth (avoids spawning inside walls).
    mean_xy = col_pos[:, :2].mean(axis=0)
    out_pos = np.array([mean_xy[0], mean_xy[1], column_z_floor],
                       dtype=np.float32)

    # Velocity = mean column velocity.  Almost always dominated by the
    # downward component since the column is falling under gravity.
    out_vel = col_vel.mean(axis=0).astype(np.float32)

    # Radius = robust spread of xy around the mean.  95th-percentile distance
    # gives a stable estimate that ignores rare outliers.
    xy_dist = np.linalg.norm(col_pos[:, :2] - mean_xy[None, :], axis=1)
    radius = float(np.percentile(xy_dist, 95)) if n_col > 5 else 0.012
    radius = max(radius, 0.005)
    radius = min(radius, 0.04)   # cap at 4 cm so we don't spawn a wall-wide blob

    return {
        "active": True,
        "pos":    out_pos,
        "vel":    out_vel,
        "radius": radius,
        "n_column": n_col,
    }


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tier B: extract per-frame column emitter trajectory "
                    "from captured_states_v2/*.bin",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--series-folder", required=True, type=Path,
                   help="Folder containing sim_state_*.bin + manifest.json.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output binary path.  Default: <series>/column_emitter.bin")
    p.add_argument("--captured-fps", type=float, default=25.0,
                   help="Source video FPS — drives the simulator's playback rate.")
    p.add_argument("--column-z-floor", type=float, default=None,
                   help="z above which a particle counts as 'column' (m).  "
                        "Default: bottle_height_target from manifest.")
    p.add_argument("--column-z-margin", type=float, default=0.005,
                   help="Extra margin added to the z-floor so we don't pick "
                        "up the pool's top surface as 'column' (m).")
    p.add_argument("--min-active-particles", type=int, default=5,
                   help="Minimum column particles for a frame to be ACTIVE.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    folder: Path = args.series_folder
    if not folder.is_dir():
        print(f"ERROR: not a folder: {folder}", file=sys.stderr)
        return 2

    output: Path = args.output if args.output else folder / "column_emitter.bin"

    manifest = _load_manifest(folder)
    if args.column_z_floor is None:
        bh = manifest.get("bottle_height_target", 0.15)
        args.column_z_floor = float(bh)

    column_z_floor = float(args.column_z_floor) + float(args.column_z_margin)

    sim_files = sorted(folder.glob("sim_state_*.bin"))
    if not sim_files:
        print(f"ERROR: no sim_state_*.bin files in {folder}", file=sys.stderr)
        return 3
    n_frames = len(sim_files)

    print("=" * 72)
    print("  extract_column_emitter.py — Tier B column trajectory")
    print("=" * 72)
    print(f"  Series:           {folder}")
    print(f"  Output:           {output}")
    print(f"  Captured FPS:     {args.captured_fps}")
    print(f"  bottle top z:     {args.column_z_floor:.4f}  (margin {args.column_z_margin})")
    print(f"  column z floor:   {column_z_floor:.4f}")
    print(f"  min active count: {args.min_active_particles}")
    print(f"  N frames:         {n_frames}")
    print("-" * 72)

    frames = []
    for f_idx, p in enumerate(sim_files):
        st = _load_sim_state_bin(p)
        s = _column_stats_for_frame(
            st["pos"], st["vel"],
            column_z_floor=column_z_floor,
            min_active=args.min_active_particles,
        )
        frames.append(s)

    # Summary
    n_active = sum(1 for f in frames if f["active"])
    print(f"Active frames: {n_active} / {n_frames}  "
          f"({100.0 * n_active / max(1, n_frames):.1f}%)")
    if n_active > 0:
        active = [f for f in frames if f["active"]]
        avg_pos = np.mean([f["pos"] for f in active], axis=0)
        avg_vel = np.mean([f["vel"] for f in active], axis=0)
        avg_rad = float(np.mean([f["radius"] for f in active]))
        avg_n   = float(np.mean([f["n_column"] for f in active]))
        print(f"  avg active pos    = "
              f"[{avg_pos[0]:+.4f}, {avg_pos[1]:+.4f}, {avg_pos[2]:+.4f}]  m")
        print(f"  avg active vel    = "
              f"[{avg_vel[0]:+.4f}, {avg_vel[1]:+.4f}, {avg_vel[2]:+.4f}]  m/s")
        print(f"  avg active radius = {avg_rad:.4f}  m")
        print(f"  avg column count  = {avg_n:.1f} particles")
    else:
        print("WARNING: no active frames — column-z-floor may be too high "
              "or the captured states may not have a column above the bottle.")

    # Write binary
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(struct.pack("<II I f",
                            _CEM_MAGIC, _CEM_VERSION,
                            n_frames, float(args.captured_fps)))
        for fr in frames:
            f.write(struct.pack("<B", 1 if fr["active"] else 0))
            f.write(b"\x00\x00\x00")
            f.write(struct.pack("<fff",
                                float(fr["pos"][0]),
                                float(fr["pos"][1]),
                                float(fr["pos"][2])))
            f.write(struct.pack("<fff",
                                float(fr["vel"][0]),
                                float(fr["vel"][1]),
                                float(fr["vel"][2])))
            f.write(struct.pack("<f", float(fr["radius"])))
            f.write(struct.pack("<f", 1.0))   # amount

    print("-" * 72)
    print(f"Wrote {output} ({output.stat().st_size} bytes)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
