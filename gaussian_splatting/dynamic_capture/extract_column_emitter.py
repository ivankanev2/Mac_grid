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
# Per-frame bottle-fill volume (used by the data-derived flow throttle).
#
# The deformation MLP has been observed to smear the falling oil column
# laterally, producing a spurious 95th-percentile xy spread that's the
# diameter of the bottle itself.  Computing Q from r_real^2 * |v| then over-
# estimates the flow rate by 1000x.
#
# A robust alternative is to derive Q directly from how fast oil ENDS UP
# inside the bottle.  Count the cells (within the captured oil's voxelisation)
# that lie below the bottle's mouth z, multiply by dx^3 to get a volume
# V_below(t), and take its time derivative dV/dt = Q_real(t).
#
# This is invariant to MLP smearing of the column's geometry — the column's
# horizontal spread doesn't affect "how much oil is now inside the bottle",
# which is what the simulator needs to match.
# -----------------------------------------------------------------------------
def _volume_below_floor(pos: np.ndarray, dx: float, origin: np.ndarray,
                         z_floor: float) -> float:
    """Return the volume (m^3) of unique cells occupied by particles whose
    z position is strictly below z_floor.  Uses cell-occupancy (binary):
    one or many particles in a cell counts the same."""
    if pos.size == 0:
        return 0.0
    below = pos[pos[:, 2] < z_floor]
    if below.size == 0:
        return 0.0
    ix = np.floor((below[:, 0] - origin[0]) / dx).astype(np.int64)
    iy = np.floor((below[:, 1] - origin[1]) / dx).astype(np.int64)
    iz = np.floor((below[:, 2] - origin[2]) / dx).astype(np.int64)
    # Pack (ix, iy, iz) into a single int64 for unique-counting.  Add a large
    # offset so negatives (positions below grid origin) don't collide.
    OFFSET = 1 << 18
    keys = ((ix + OFFSET) << 40) | ((iy + OFFSET) << 20) | (iz + OFFSET)
    n_cells = int(np.unique(keys).size)
    return float(n_cells) * (float(dx) ** 3)


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
                             min_active: int,
                             velocity_mode: str = "gravity",
                             gravity: float = 9.8,
                             min_fall_height: float = 0.005,
                             radius_clamp_max: float = 0.04):
    """Return dict {active, pos, vel, radius, top_z, fall_height} for a
    single captured frame.

    velocity_mode controls how out_vel is derived:
      "captured": mean MLP-output velocity of column particles (unreliable
                  in monocular static-camera setting; for ablation only).
      "gravity":  out_vel.z = -sqrt(2 * g * fall_height), where
                  fall_height = top_z - column_z_floor (clamped to
                  min_fall_height so the velocity is never exactly zero).
                  xy components are zero — column falls straight down.
    """
    column_mask = pos[:, 2] > column_z_floor
    n_col = int(column_mask.sum())

    if n_col < min_active:
        # Inactive frame — write a sane default velocity for the binary
        # but mark active=False so the simulator skips the emission.
        return {
            "active": False,
            "pos":     np.array([0.0, 0.0, column_z_floor], dtype=np.float32),
            "vel":     np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "radius":  0.012,
            "radius_real": 0.0,         # used for flow-rate calc; 0 = no flow
            "n_column": n_col,
            "top_z":     float("nan"),
            "fall_height": float("nan"),
        }

    col_pos = pos[column_mask]
    col_vel = vel[column_mask]

    # Centre xy = mean of column points; z = column_z_floor so the source
    # sphere sits at the bottle mouth (avoids spawning inside walls).
    mean_xy = col_pos[:, :2].mean(axis=0)
    out_pos = np.array([mean_xy[0], mean_xy[1], column_z_floor],
                       dtype=np.float32)

    # The top of the captured column — used for the gravity velocity.
    top_z = float(col_pos[:, 2].max())
    fall_height = max(top_z - column_z_floor, float(min_fall_height))

    if velocity_mode == "gravity":
        # Free-fall from top_z to emitter z.  Equivalent to: a particle
        # released at rest at top_z reaches emitter z with this speed.
        v_z = -float(np.sqrt(2.0 * float(gravity) * fall_height))
        out_vel = np.array([0.0, 0.0, v_z], dtype=np.float32)
    else:  # "captured" — keep the legacy MLP-mean path for ablations.
        out_vel = col_vel.mean(axis=0).astype(np.float32)

    # Radius = robust spread of xy around the mean.  95th-percentile distance
    # gives a stable estimate that ignores rare outliers.
    #
    # We track TWO radii here:
    #   radius_real -- the captured value (loosely, the column's true cross-
    #     section as the deformation MLP saw it).  Used downstream to compute
    #     the physical flow rate Q = pi * r_real^2 * |v|.
    #   radius      -- the value used as the simulator's emit-sphere geometry.
    #     This is clamped UP to a minimum of one cell (otherwise the sphere
    #     misses cell centres entirely and emission is stochastic), and DOWN
    #     to a cell-resolution-aware ceiling (otherwise the MLP's lateral
    #     smearing produces a column wider than physically real).
    #
    # The throttle factor "amount" (computed in the main loop) decouples flow
    # rate from emit-sphere geometry: amount = Q_real / V_sim_per_emit, in
    # units of emits-per-second.  The viewer accumulates amount*dt each
    # substep and only fires addWaterSourceSphere when it crosses 1.
    xy_dist = np.linalg.norm(col_pos[:, :2] - mean_xy[None, :], axis=1)
    radius_raw = float(np.percentile(xy_dist, 95)) if n_col > 5 else 0.012
    radius_real = max(radius_raw, 1e-4)         # keep nearly-true value
    # Sphere geometry for the simulator: clamp to [dx, radius_clamp_max].
    radius = min(max(radius_raw, 0.005), float(radius_clamp_max))

    return {
        "active": True,
        "pos":    out_pos,
        "vel":    out_vel,
        "radius": radius,
        "radius_real": radius_real,
        "n_column": n_col,
        "top_z": top_z,
        "fall_height": fall_height,
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
    # v2: velocity computation mode.  The deformation MLP's velocity output
    # is unreliable in the static-camera monocular setting (the MLP fits the
    # visual pattern of a continuous stream rather than tracking Lagrangian
    # particle motion).  We default to a gravity-derived velocity computed
    # from the captured column's top z and the emitter z, which uses the
    # video's spatial structure but bypasses the broken velocity output.
    p.add_argument("--velocity-mode",
                   choices=["captured", "gravity"], default="gravity",
                   help="How the per-frame column velocity is computed. "
                        "'captured' = mean of MLP-output velocities "
                        "(unreliable, kept for ablation). "
                        "'gravity' (default) = sqrt(2*g*fall) from the "
                        "captured column's top-z minus the emitter z; "
                        "uses the video's spatial structure but bypasses "
                        "the MLP velocity output.")
    p.add_argument("--gravity", type=float, default=9.8,
                   help="Gravity magnitude (m/s^2) used by --velocity-mode "
                        "gravity.  Standard 9.8 unless you're simulating "
                        "elsewhere on the solar system.")
    p.add_argument("--min-fall-height", type=float, default=0.005,
                   help="Floor for fall_height in --velocity-mode gravity "
                        "(m).  Avoids zero velocities when the column's "
                        "top is right at the emitter z.")
    # Radius clamp: the deformation MLP smears the column laterally to fit
    # visual continuity, producing an unrealistically wide cross-section.
    # We cap the per-frame radius to a multiple of the simulator cell size
    # so the captured column is at least cell-coherent (no fatter than the
    # finest feature the grid can represent).
    p.add_argument("--radius-clamp-cells", type=float, default=1.5,
                   help="Maximum column radius in units of grid cells dx. "
                        "1.5 means radius is capped at 1.5*dx.  The captured "
                        "MLP-spread radius is unreliable because the "
                        "deformation MLP smears thin features laterally; "
                        "this clamp produces a thin column that's at least "
                        "consistent with the grid resolution.")
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

    # Cell size from manifest — drives the radius clamp.
    dx = float(manifest.get("dx", 0.015))
    radius_clamp_max = float(args.radius_clamp_cells) * dx

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
    print(f"  velocity mode:    {args.velocity_mode}")
    if args.velocity_mode == "gravity":
        print(f"  gravity:          {args.gravity} m/s^2")
        print(f"  min fall height:  {args.min_fall_height} m")
    print(f"  dx (from mf):     {dx} m")
    print(f"  radius clamp:     {args.radius_clamp_cells} cells "
          f"(= {radius_clamp_max:.4f} m max radius)")
    print(f"  N frames:         {n_frames}")
    print("-" * 72)

    # First pass: per-frame column stats AND below-floor occupied volume.
    frames = []
    v_below = np.zeros(n_frames, dtype=np.float64)   # m^3 oil-occupied below mouth
    for f_idx, p in enumerate(sim_files):
        st = _load_sim_state_bin(p)
        s = _column_stats_for_frame(
            st["pos"], st["vel"],
            column_z_floor=column_z_floor,
            min_active=args.min_active_particles,
            velocity_mode=args.velocity_mode,
            gravity=args.gravity,
            min_fall_height=args.min_fall_height,
            radius_clamp_max=radius_clamp_max,
        )
        frames.append(s)
        v_below[f_idx] = _volume_below_floor(
            st["pos"], st["dx"], st["origin"], column_z_floor,
        )

    # ------------------------------------------------------------------
    # Per-frame data-derived flow rate ("amount") — bottle-fill version.
    #
    # Earlier this used Q = pi * r_real^2 * |v|, but the deformation MLP
    # smears the falling oil column laterally so r_real (95th percentile xy
    # spread of "column" particles) reads as ~11cm — basically the bottle's
    # radius.  That gives a Q that's 1000x physical reality.
    #
    # Replacement: compute Q from the rate at which oil ENDS UP inside the
    # bottle.  V_below(t) = volume of cells occupied by oil below the bottle
    # mouth at frame t (computed above as v_below).  Q_real(t) = dV/dt via
    # centred finite difference, then smoothed with a short boxcar to
    # suppress per-frame voxelisation noise.
    #
    # This estimator is invariant to MLP smearing of the column's geometry
    # because we don't trust any column geometry — we just measure the
    # in-bottle volume directly.
    #
    # Sphere geometry V_sim(f) = (4/3) * pi * r(f)^3 stays the same (the
    # simulator's emit sphere needs to be at least one cell wide for cell-
    # centre tests to hit anything).
    #
    # amount(f) = Q_real(f) / V_sim(f)         [emits per second]
    # ------------------------------------------------------------------
    pi = float(np.pi)
    dt_capture = 1.0 / float(args.captured_fps)

    # Centred finite difference: Q[i] = (V[i+1] - V[i-1]) / (2 dt).
    Q_inst = np.zeros(n_frames, dtype=np.float64)
    if n_frames >= 3:
        Q_inst[1:-1] = (v_below[2:] - v_below[:-2]) / (2.0 * dt_capture)
        Q_inst[0]  = (v_below[1] - v_below[0]) / dt_capture
        Q_inst[-1] = (v_below[-1] - v_below[-2]) / dt_capture
    elif n_frames == 2:
        d = (v_below[1] - v_below[0]) / dt_capture
        Q_inst[:] = d

    # Clamp to non-negative — bottles only fill, they don't drain.
    Q_inst = np.maximum(Q_inst, 0.0)

    # Smooth with a short boxcar to absorb single-frame voxelisation jitter.
    # Window 7 frames at 25 fps = 0.28 s — short enough to track real flow
    # variations, long enough to filter cell-quantisation noise.
    smooth_window = 7
    if n_frames >= smooth_window:
        kernel = np.ones(smooth_window, dtype=np.float64) / float(smooth_window)
        Q_smooth = np.convolve(Q_inst, kernel, mode="same")
    else:
        Q_smooth = Q_inst.copy()

    for fr_idx, fr in enumerate(frames):
        if not fr["active"]:
            fr["amount"] = 0.0
            continue
        V_sim = (4.0 / 3.0) * pi * (float(fr["radius"]) ** 3)
        if V_sim > 1e-12:
            fr["amount"] = float(Q_smooth[fr_idx] / V_sim)
        else:
            fr["amount"] = 0.0
        # Stash for diagnostics
        fr["Q_real"]  = float(Q_smooth[fr_idx])
        fr["v_below"] = float(v_below[fr_idx])

    # Summary
    n_active = sum(1 for f in frames if f["active"])
    print(f"Active frames: {n_active} / {n_frames}  "
          f"({100.0 * n_active / max(1, n_frames):.1f}%)")
    if n_active > 0:
        active = [f for f in frames if f["active"]]
        avg_pos = np.mean([f["pos"] for f in active], axis=0)
        avg_vel = np.mean([f["vel"] for f in active], axis=0)
        avg_rad = float(np.mean([f["radius"] for f in active]))
        avg_rad_real = float(np.mean([f["radius_real"] for f in active]))
        avg_n   = float(np.mean([f["n_column"] for f in active]))
        avg_top_z       = float(np.mean([f["top_z"]       for f in active]))
        avg_fall_height = float(np.mean([f["fall_height"] for f in active]))
        avg_amount = float(np.mean([f["amount"] for f in active]))
        avg_Q_real = float(np.mean([f["Q_real"] for f in active]))
        active_duration = n_active / float(args.captured_fps)
        total_volume = float(np.sum(Q_smooth)) * dt_capture
        # Bottle-fill diagnostics
        v_below_max = float(v_below.max())
        v_below_final = float(v_below[-1])
        print(f"  avg active pos        = "
              f"[{avg_pos[0]:+.4f}, {avg_pos[1]:+.4f}, {avg_pos[2]:+.4f}]  m")
        print(f"  avg active vel        = "
              f"[{avg_vel[0]:+.4f}, {avg_vel[1]:+.4f}, {avg_vel[2]:+.4f}]  m/s")
        print(f"  avg sim-sphere radius = {avg_rad:.4f}  m  (clamped, used for emit geometry)")
        print(f"  avg real radius       = {avg_rad_real:.4f}  m  (captured, NOT used for Q anymore)")
        print(f"  avg column count      = {avg_n:.1f} particles")
        print(f"  avg top z             = {avg_top_z:.4f}  m  (max z of column)")
        print(f"  avg fall height       = {avg_fall_height:.4f}  m  "
              f"(top z - emitter z)")
        if args.velocity_mode == "gravity":
            v_implied = -float(np.sqrt(2.0 * args.gravity * max(
                avg_fall_height, args.min_fall_height)))
            print(f"  -> implied gravity vz at avg fall = {v_implied:+.3f} m/s")
        print(f"  --- bottle-fill flow rate (data-derived) ---")
        print(f"  V_below_floor max     = {v_below_max * 1e6:.1f}  mL  "
              f"(peak occupied volume below bottle mouth)")
        print(f"  V_below_floor final   = {v_below_final * 1e6:.1f}  mL  "
              f"(at last frame)")
        print(f"  avg Q_real            = {avg_Q_real * 1e6:.2f}  mL/s  "
              f"(smoothed dV/dt)")
        print(f"  avg emit rate         = {avg_amount:.3f}  emits/s")
        print(f"  estimated total pour  = {total_volume * 1e6:.1f}  mL "
              f"over {active_duration:.2f} s")
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
            # amount = data-derived emit rate in emits-per-second; the viewer
            # accumulates amount*dt per substep and emits when it crosses 1.
            f.write(struct.pack("<f", float(fr["amount"])))

    print("-" * 72)
    print(f"Wrote {output} ({output.stat().st_size} bytes)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
