#!/usr/bin/env python3
"""transpose_zup_to_yup.py — convert pipeline data files from Z-up to Y-up.

Background
----------
The extraction scripts (extract_fluid_state_v2.py, extract_column_emitter.py,
bottle_from_mesh.py) write data with the bottle's vertical axis along **Z**
and the column's fall direction along **-Z** (see the explicit alignment
``_rotation_aligning(g_capture, [0, 0, -1])`` in extract_fluid_state_v2.py).

The MAC water solver, however, applies gravity to the **V (Y) velocity
component** (``v[idxV(i, j, k)] += dt * params.gravity`` in water3d_particles.h),
so the simulator is Y-up.

Result: every previous bottle has been laying on its side in the simulator,
and gravity has been pulling fluid sideways relative to the bottle.  This
script fixes existing data files in place by swapping the Y and Z axes.

What gets swapped
-----------------
For every coordinate triple (x, y, z) in either world space or grid space:
  new_x = old_x                 (X is unchanged)
  new_y = old_z                 (the bottle's vertical moves into Y)
  new_z = old_y                 (the lateral becomes Z)

For 3D voxel masks stored in (z, y, x) byte order:
  new_array = old_array.transpose(1, 0, 2)
  new shape: (old_ny, old_nz, old_nx) — i.e., (new_nz, new_ny, new_nx)

For grid header fields:
  new_nx = old_nx
  new_ny = old_nz       (the vertical-axis cell count is now in ny)
  new_nz = old_ny

For origin coords:
  new_origin_x = old_origin_x
  new_origin_y = old_origin_z   (the floor of the world is now at y_min)
  new_origin_z = old_origin_y

For per-frame trajectories (positions and velocities):
  new_pos_or_vel = (x, z, y)

Usage
-----
    # Convert the mesh-derived bottle and its emitter:
    python3 transpose_zup_to_yup.py \\
        --bottle    captured_states_v2/bottle_solid_mesh.bin \\
        --emitter   captured_states_v2/column_emitter.bin \\
        --manifest  captured_states_v2/manifest.json \\
        --suffix    _yup

The script writes new files alongside the inputs with ``_yup`` appended to
each stem (e.g. ``bottle_solid_mesh_yup.bin``).  Originals are NOT modified.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Bottle solid (BSL1)
# -----------------------------------------------------------------------------
_BSL_MAGIC = 0x42534C31
_BSL_HEADER_FMT = "<IIIII fff fI"   # magic, ver, nx, ny, nz, dx, ox, oy, oz, n_solid
_BSL_HEADER_SIZE = struct.calcsize(_BSL_HEADER_FMT)


def transpose_bottle_solid(in_path: Path, out_path: Path) -> dict:
    data = in_path.read_bytes()
    magic, ver, nx, ny, nz, dx, ox, oy, oz, n_solid = struct.unpack_from(
        _BSL_HEADER_FMT, data, 0)
    if magic != _BSL_MAGIC:
        raise ValueError(f"Bad BSL magic in {in_path}: 0x{magic:08X}")

    # Old layout: byte stream is mask[old_z][old_y][old_x].
    arr = np.frombuffer(data, dtype=np.uint8,
                         offset=_BSL_HEADER_SIZE,
                         count=int(nx) * int(ny) * int(nz))
    mask_old = arr.reshape(nz, ny, nx)

    # Transpose axes 0 and 1 → mask_new[new_z][new_y][new_x] where
    # new_y=old_z (vertical) and new_z=old_y.
    mask_new = mask_old.transpose(1, 0, 2)
    new_nz, new_ny, new_nx = mask_new.shape   # (old_ny, old_nz, old_nx)

    # Sanity: cell count is preserved.
    assert int(mask_new.sum()) == int(n_solid), (
        f"cell count mismatch: old {n_solid} vs new {int(mask_new.sum())}")
    # Sanity: nx unchanged.
    assert new_nx == nx

    new_ox = ox
    new_oy = oz   # old origin_z (floor of the world) becomes new origin_y
    new_oz = oy

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack(_BSL_HEADER_FMT,
                            magic, ver,
                            int(new_nx), int(new_ny), int(new_nz),
                            float(dx),
                            float(new_ox), float(new_oy), float(new_oz),
                            int(n_solid)))
        f.write(np.ascontiguousarray(mask_new).tobytes(order="C"))

    return {
        "old_grid": (int(nx), int(ny), int(nz)),
        "new_grid": (int(new_nx), int(new_ny), int(new_nz)),
        "old_origin": (float(ox), float(oy), float(oz)),
        "new_origin": (float(new_ox), float(new_oy), float(new_oz)),
        "n_solid":   int(n_solid),
        "out_size":  out_path.stat().st_size,
    }


# -----------------------------------------------------------------------------
# Column emitter (CEM1)
# -----------------------------------------------------------------------------
_CEM_MAGIC = 0x43454D31
_CEM_HEADER_FMT = "<IIIf"           # magic, ver, n_frames, fps
_CEM_HEADER_SIZE = struct.calcsize(_CEM_HEADER_FMT)
_CEM_RECORD_SIZE = 36                # 1B active + 3B pad + 6f vec + 1f r + 1f amount


def transpose_column_emitter(in_path: Path, out_path: Path) -> dict:
    data = in_path.read_bytes()
    magic, ver, n_frames, fps = struct.unpack_from(_CEM_HEADER_FMT, data, 0)
    if magic != _CEM_MAGIC:
        raise ValueError(f"Bad CEM magic in {in_path}: 0x{magic:08X}")

    expected = _CEM_HEADER_SIZE + n_frames * _CEM_RECORD_SIZE
    if len(data) != expected:
        raise ValueError(
            f"CEM file size {len(data)} != expected {expected}")

    out = bytearray()
    out += struct.pack(_CEM_HEADER_FMT, magic, ver, n_frames, fps)

    off = _CEM_HEADER_SIZE
    n_active = 0
    pos_extents = [[+1e30, -1e30] for _ in range(3)]
    for _ in range(int(n_frames)):
        active = struct.unpack_from("<B", data, off)[0]
        # 3 pad bytes
        px, py, pz = struct.unpack_from("<fff", data, off + 4)
        vx, vy, vz = struct.unpack_from("<fff", data, off + 16)
        radius     = struct.unpack_from("<f",   data, off + 28)[0]
        amount     = struct.unpack_from("<f",   data, off + 32)[0]

        # Swap Y and Z components.
        new_p = (px, pz, py)
        new_v = (vx, vz, vy)

        out += struct.pack("<B", active)
        out += b"\x00\x00\x00"
        out += struct.pack("<fff", *new_p)
        out += struct.pack("<fff", *new_v)
        out += struct.pack("<f",   radius)
        out += struct.pack("<f",   amount)

        off += _CEM_RECORD_SIZE
        if active:
            n_active += 1
            for i, c in enumerate(new_p):
                pos_extents[i][0] = min(pos_extents[i][0], c)
                pos_extents[i][1] = max(pos_extents[i][1], c)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(out))

    return {
        "n_frames":       int(n_frames),
        "n_active":       int(n_active),
        "fps":            float(fps),
        "new_pos_extents_xyz": [[float(a), float(b)] for a, b in pos_extents],
        "out_size":       out_path.stat().st_size,
    }


# -----------------------------------------------------------------------------
# Manifest (JSON)
# -----------------------------------------------------------------------------
def transpose_manifest(in_path: Path, out_path: Path) -> dict:
    m = json.loads(in_path.read_text(encoding="utf-8"))
    grid = dict(m.get("grid", {}))
    nx = int(grid.get("nx", 0))
    ny = int(grid.get("ny", 0))
    nz = int(grid.get("nz", 0))
    ox = float(grid.get("origin_x", 0.0))
    oy = float(grid.get("origin_y", 0.0))
    oz = float(grid.get("origin_z", 0.0))

    new_grid = {
        "nx": nx,
        "ny": nz,           # was old nz (the vertical)
        "nz": ny,
        "origin_x": ox,
        "origin_y": oz,     # was old origin_z (the floor)
        "origin_z": oy,
    }
    new_m = dict(m)
    new_m["grid"] = new_grid

    # If oil_align_offset exists, swap Y and Z too.
    off = m.get("oil_align_offset")
    if isinstance(off, list) and len(off) == 3:
        new_m["oil_align_offset"] = [off[0], off[2], off[1]]

    new_m["coordinate_system"] = "y-up"
    new_m["transposed_from"] = str(in_path.name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(new_m, indent=2), encoding="utf-8")

    return {
        "old_grid": (nx, ny, nz),
        "new_grid": (new_grid["nx"], new_grid["ny"], new_grid["nz"]),
        "old_origin": (ox, oy, oz),
        "new_origin": (new_grid["origin_x"],
                       new_grid["origin_y"],
                       new_grid["origin_z"]),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _suffix_path(p: Path, suffix: str) -> Path:
    return p.with_name(p.stem + suffix + p.suffix)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Transpose Z-up extraction files into Y-up to match the "
                    "simulator's gravity convention.")
    ap.add_argument("--bottle",   type=Path, default=None,
                    help="Path to bottle_solid.bin (or _mesh.bin) to transpose.")
    ap.add_argument("--emitter",  type=Path, default=None,
                    help="Path to column_emitter.bin to transpose.")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Path to manifest.json to transpose.")
    ap.add_argument("--suffix",   type=str, default="_yup",
                    help="Suffix appended to each output filename "
                         "(default: _yup).")
    args = ap.parse_args(argv)

    if not (args.bottle or args.emitter or args.manifest):
        ap.error("must pass at least one of --bottle, --emitter, --manifest")

    print("=" * 72)
    print("  transpose_zup_to_yup.py — Z-up → Y-up data conversion")
    print("=" * 72)

    if args.bottle:
        out = _suffix_path(args.bottle, args.suffix)
        print(f"\n[bottle] {args.bottle} -> {out}")
        info = transpose_bottle_solid(args.bottle, out)
        print(f"  grid:    {info['old_grid']} -> {info['new_grid']}")
        print(f"  origin:  {info['old_origin']} -> {info['new_origin']}")
        print(f"  cells:   {info['n_solid']} (preserved)")
        print(f"  size:    {info['out_size']} bytes")

    if args.emitter:
        out = _suffix_path(args.emitter, args.suffix)
        print(f"\n[emitter] {args.emitter} -> {out}")
        info = transpose_column_emitter(args.emitter, out)
        print(f"  n_frames={info['n_frames']}  active={info['n_active']}  "
              f"fps={info['fps']}")
        for axis, (lo, hi) in zip("xyz", info["new_pos_extents_xyz"]):
            print(f"  pos_{axis}: [{lo:+.4f}, {hi:+.4f}]")
        print(f"  size:    {info['out_size']} bytes")

    if args.manifest:
        out = _suffix_path(args.manifest, args.suffix)
        print(f"\n[manifest] {args.manifest} -> {out}")
        info = transpose_manifest(args.manifest, out)
        print(f"  grid:    {info['old_grid']} -> {info['new_grid']}")
        print(f"  origin:  {info['old_origin']} -> {info['new_origin']}")

    print("\n" + "=" * 72)
    print("  Done.  Load the *_yup files into the viewer.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
