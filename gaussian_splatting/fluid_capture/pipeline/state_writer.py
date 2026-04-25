"""Binary fluid_state writer for the C++ simulator side.

Format (little-endian):

    [uint32  magic       = 0x46535431  // 'F','S','T','1' as 4 bytes]
    [uint32  version     = 1]
    [uint32  nx]
    [uint32  ny]
    [uint32  nz]
    [float32 dx]
    [float32 origin_x]
    [float32 origin_y]
    [float32 origin_z]
    [uint32  n_particles]
    [for i in 0..n_particles:
        float32 px, py, pz, vx, vy, vz   // 24 bytes per particle
    ]

That's the whole file.  We deliberately keep it minimal — no fluid-cell
mask in the binary because the simulator can recover it from
particle locations (any cell containing particles = fluid).  Future
versions can append additional sections after the particle block;
parsers must check ``version`` before reading anything beyond the
particles.

The matching C++ reader will mirror this layout byte-for-byte.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .voxelize import SeededState


MAGIC = 0x46535431  # 'F', 'S', 'T', '1'
VERSION = 1


def write_fluid_state(path: str | Path, state: SeededState) -> Path:
    """Write a SeededState to a fluid_state.bin file.  Returns the resolved path."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    pos = state.particles_pos.astype(np.float32, copy=False)
    vel = state.particles_vel.astype(np.float32, copy=False)
    n = int(pos.shape[0])
    if vel.shape[0] != n:
        raise ValueError(f"pos/vel size mismatch: {pos.shape} vs {vel.shape}")

    # Interleave into a single (N, 6) array, then flatten to (N*6,).
    interleaved = np.empty((n, 6), dtype=np.float32)
    interleaved[:, 0:3] = pos
    interleaved[:, 3:6] = vel
    flat = interleaved.reshape(-1)

    with p.open("wb") as f:
        # Header
        f.write(struct.pack(
            "<IIIII fff fI",   # see format spec at module top
            MAGIC, VERSION,
            int(state.grid.nx), int(state.grid.ny), int(state.grid.nz),
            float(state.grid.dx),
            float(state.grid.origin[0]),
            float(state.grid.origin[1]),
            float(state.grid.origin[2]),
            n,
        ))
        # Particles: (N, 6) float32 little-endian.
        f.write(flat.astype("<f4", copy=False).tobytes(order="C"))

    return p


def read_fluid_state(path: str | Path):
    """Round-trip companion: parse a fluid_state.bin and return its contents.

    Returned as a plain dict to keep the API minimal for tests / debugging.
    """
    p = Path(path).expanduser().resolve()
    with p.open("rb") as f:
        header_struct = "<IIIII fff fI"
        header_size = struct.calcsize(header_struct)
        head = f.read(header_size)
        if len(head) != header_size:
            raise IOError(f"Truncated header in {p}")
        (magic, version, nx, ny, nz, dx, ox, oy, oz, n) = struct.unpack(header_struct, head)
        if magic != MAGIC:
            raise IOError(f"Bad magic 0x{magic:08x} in {p}")
        if version != VERSION:
            raise IOError(f"Unsupported fluid_state version {version} in {p}")

        body_size = n * 6 * 4
        body = f.read(body_size)
        if len(body) != body_size:
            raise IOError(f"Truncated particle body in {p} (expected {body_size}, got {len(body)})")
        flat = np.frombuffer(body, dtype="<f4")
        flat = flat.reshape(n, 6)

    return {
        "version": version,
        "nx": nx, "ny": ny, "nz": nz,
        "dx": dx,
        "origin": np.array([ox, oy, oz], dtype=np.float32),
        "n_particles": n,
        "positions": flat[:, 0:3].copy(),
        "velocities": flat[:, 3:6].copy(),
    }
