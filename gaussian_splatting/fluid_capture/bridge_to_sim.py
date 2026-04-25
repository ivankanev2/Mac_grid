#!/usr/bin/env python3
"""Bridge captured fluid state to the simulator's coordinate / particle representation.

Pipeline:

    load fluid_state.npz → orient (gravity → -Z) → scale to target height →
    translate (bottom of fluid at z=0, centred in x/y) →
    voxelise + vertical-fill → seed FLIP particles →
    write sim_state.bin + preview PLYs.

Example
-------
    python bridge_to_sim.py \\
        --input  outputs/fluid_state.npz \\
        --output-dir outputs/

Output artefacts (alongside the inputs):

    outputs/sim_state.bin                — binary file the C++ side will load
    outputs/sim_state_particles.ply      — preview point cloud, vmag-coloured
    outputs/sim_domain.ply               — wireframe of the chosen sim domain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from pipeline import bridge, voxelize, state_writer, preview


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bridge captured fluid state into simulator units.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="fluid_state.npz emitted by capture_fluid.py")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write sim_state.bin and preview PLYs.")
    p.add_argument("--target-height", type=float, default=0.15,
                   help="World-space height (m) the tallest extent of the cloud should map to.")
    p.add_argument("--dx", type=float, default=0.015,
                   help="Simulator voxel size in metres.")
    p.add_argument("--headroom-below", type=float, default=0.20,
                   help="Domain headroom below the fluid (m) — room for it to fall.")
    p.add_argument("--headroom-above", type=float, default=0.05,
                   help="Domain headroom above the fluid (m).")
    p.add_argument("--headroom-lateral", type=float, default=0.05,
                   help="Domain lateral headroom (m, applied to both x and y).")
    p.add_argument("--particles-per-cell", type=int, default=13,
                   help="FLIP particles seeded per fluid cell.")
    p.add_argument("--rng-seed", type=int, default=0,
                   help="Random seed for particle jitter inside cells.")
    return p.parse_args(argv)


def _print_header(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("  Bridge captured fluid → simulator coordinates (M2 Phase 1)")
    print("=" * 72)
    print(f"  Input npz:    {args.input}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Target ht:    {args.target_height:.4f} m")
    print(f"  Voxel size:   {args.dx:.4f} m")
    print(f"  Headroom:     below={args.headroom_below}  above={args.headroom_above}  lateral={args.headroom_lateral}")
    print(f"  Particles/cell: {args.particles_per_cell}")
    print("-" * 72)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _print_header(args)

    if not args.input.exists():
        print(f"ERROR: input npz not found: {args.input}", file=sys.stderr)
        return 2

    # 1. Load.
    print("[1/5] Loading captured fluid state ...")
    points, velocities, colors, meta = bridge.load_fluid_state_npz(args.input)
    print(f"      {len(points)} points, dt={meta['dt']:.4f}s")

    # 2. Bridge to world frame.
    print("[2/5] Orienting + scaling + translating to sim world frame ...")
    world = bridge.transform_to_world(
        points=points,
        velocities=velocities,
        colors=colors,
        target_height_m=args.target_height,
    )
    g = world.gravity_dir_capture
    print(f"      gravity (capture frame) = [{g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f}]")
    print(f"      scale factor            = {world.scale_factor:.6f}")
    print(f"      world bounds            = "
          f"[{world.bounds_min[0]:.3f}, {world.bounds_min[1]:.3f}, {world.bounds_min[2]:.3f}] – "
          f"[{world.bounds_max[0]:.3f}, {world.bounds_max[1]:.3f}, {world.bounds_max[2]:.3f}]  m")

    # 3. Voxelise + fill + seed.
    print("[3/5] Voxelising + filling + seeding FLIP particles ...")
    state = voxelize.voxelise_and_seed(
        world,
        dx=args.dx,
        headroom_below=args.headroom_below,
        headroom_above=args.headroom_above,
        headroom_lateral=args.headroom_lateral,
        particles_per_cell=args.particles_per_cell,
        rng_seed=args.rng_seed,
    )
    g = state.grid
    print(f"      grid          = {g.nx} × {g.ny} × {g.nz}  ({g.nx * g.ny * g.nz} cells)")
    print(f"      grid origin   = [{g.origin[0]:.3f}, {g.origin[1]:.3f}, {g.origin[2]:.3f}]  m")
    print(f"      fluid cells   = {state.n_fluid_cells}  ({100.0 * state.n_fluid_cells / max(1, g.nx*g.ny*g.nz):.2f}% of domain)")
    print(f"      particles     = {len(state.particles_pos)}")

    if len(state.particles_pos) == 0:
        print("ERROR: 0 particles seeded — voxelisation produced no fluid cells.",
              file=sys.stderr)
        return 3

    vmag = np.linalg.norm(state.particles_vel, axis=1)
    print(f"      vel stats     = mean={vmag.mean():.3f}  median={np.median(vmag):.3f}  "
          f"max={vmag.max():.3f}  (m/s)")

    # 4. Write sim_state.bin.
    print("[4/5] Writing sim_state.bin ...")
    bin_path = state_writer.write_fluid_state(args.output_dir / "sim_state.bin", state)
    print(f"      {bin_path}")

    # 5. Write preview PLYs.
    print("[5/5] Writing preview PLYs ...")
    pre_pts = preview.save_particles_preview(
        args.output_dir / "sim_state_particles.ply", state
    )
    pre_dom = preview.save_domain_wireframe(
        args.output_dir / "sim_domain.ply", state
    )
    print(f"      {pre_pts}")
    print(f"      {pre_dom}")

    print("-" * 72)
    print("  Done.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
