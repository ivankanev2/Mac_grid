"""Voxelise the world-frame surface cloud into a MAC grid + FLIP particles.

Two challenges:

1. **The captured cloud is a *shell* — only the camera-facing surface is
   real.**  We can't just voxelise it directly: that gives us a thin
   surface of fluid cells with hollow interior, which the simulator
   would render as a paper-thin sheet.  We need to *fill* the interior.

2. **The simulator wants ~13 FLIP particles per fluid cell.**  We seed
   particles within each fluid cell at random offsets and assign each a
   velocity by nearest-neighbour lookup against the captured surface
   velocities.

The fill heuristic is a vertical column fill: at each (i, j) position in
the grid, mark every cell from the lowest occupied k to the highest
occupied k as fluid.  Works well for a pour-and-pool topology because
the fluid is column-shaped along the gravity axis.  For arbitrary
splash topologies we'd want a real volumetric reconstruction (alpha
shape / Poisson reconstruction), which is a Phase-2 upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .bridge import WorldFluid

# open3d is imported lazily inside seed_flip_particles so callers that
# only need SeededState / the grid maths don't pay the full import cost.


@dataclass
class SimGrid:
    """MAC-grid parameters for the simulator domain."""
    nx: int
    ny: int
    nz: int
    dx: float
    origin: np.ndarray   # (3,) world-frame coordinates of voxel (0,0,0) corner

    def cell_centre(self, i: int, j: int, k: int) -> np.ndarray:
        return self.origin + (np.array([i, j, k], dtype=np.float32) + 0.5) * self.dx

    def domain_min(self) -> np.ndarray:
        return self.origin

    def domain_max(self) -> np.ndarray:
        return self.origin + np.array([self.nx, self.ny, self.nz], dtype=np.float32) * self.dx


@dataclass
class SeededState:
    grid: SimGrid
    fluid_mask: np.ndarray         # (nx, ny, nz) bool
    particles_pos: np.ndarray      # (P, 3) float32, world-space positions
    particles_vel: np.ndarray      # (P, 3) float32, world-space velocities (m/s)
    n_fluid_cells: int


def make_grid_for_fluid(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    dx: float = 0.015,
    headroom_below: float = 0.20,
    headroom_above: float = 0.05,
    headroom_lateral: float = 0.05,
) -> SimGrid:
    """Pick a grid that contains the fluid plus enough room to fall.

    Defaults give 20 cm of headroom under the fluid (so it has room to
    fall under gravity), 5 cm above (so the top of the column has a
    little buffer), and 5 cm laterally (so lateral splashing has
    somewhere to go).
    """
    if dx <= 0:
        raise ValueError(f"dx must be > 0; got {dx}")

    domain_min = np.array([
        bounds_min[0] - headroom_lateral,
        bounds_min[1] - headroom_lateral,
        bounds_min[2] - headroom_below,
    ], dtype=np.float32)
    domain_max = np.array([
        bounds_max[0] + headroom_lateral,
        bounds_max[1] + headroom_lateral,
        bounds_max[2] + headroom_above,
    ], dtype=np.float32)

    extent = domain_max - domain_min
    nx = int(np.ceil(extent[0] / dx))
    ny = int(np.ceil(extent[1] / dx))
    nz = int(np.ceil(extent[2] / dx))
    nx = max(nx, 1)
    ny = max(ny, 1)
    nz = max(nz, 1)

    return SimGrid(nx=nx, ny=ny, nz=nz, dx=float(dx), origin=domain_min)


def voxelise_cloud(
    points: np.ndarray,
    grid: SimGrid,
) -> np.ndarray:
    """Boolean (nx, ny, nz) occupancy mask: True where any captured point falls."""
    if points.size == 0:
        return np.zeros((grid.nx, grid.ny, grid.nz), dtype=bool)

    rel = (points - grid.origin) / grid.dx
    idx = np.floor(rel).astype(np.int32)
    in_bounds = (
        (idx[:, 0] >= 0) & (idx[:, 0] < grid.nx) &
        (idx[:, 1] >= 0) & (idx[:, 1] < grid.ny) &
        (idx[:, 2] >= 0) & (idx[:, 2] < grid.nz)
    )
    idx = idx[in_bounds]

    mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=bool)
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return mask


def fill_vertical_columns(mask: np.ndarray) -> np.ndarray:
    """Vertical column fill: for each (i, j), fill all k between min and max occupied k.

    Converts a thin captured-surface shell into a filled volume that the
    FLIP solver can seed properly.  Z is assumed to be the gravity axis
    (set up that way in :mod:`bridge`).
    """
    nx, ny, nz = mask.shape
    filled = mask.copy()
    # Cumulative max from below and above lets us mark "between min and max k".
    any_occ = mask.any(axis=2)        # (nx, ny)
    if not any_occ.any():
        return filled

    # For each (i,j), find the range [k_min, k_max] of occupied cells.
    k_indices = np.arange(nz, dtype=np.int32)
    # Set unoccupied cells' k to a sentinel so min/max ignores them.
    k_idx_grid = np.broadcast_to(k_indices, (nx, ny, nz))
    masked_for_min = np.where(mask, k_idx_grid, nz + 1)
    masked_for_max = np.where(mask, k_idx_grid, -1)
    k_min = masked_for_min.min(axis=2)
    k_max = masked_for_max.max(axis=2)

    # Build the filled mask.
    for k in range(nz):
        in_range = (k >= k_min) & (k <= k_max) & any_occ
        filled[..., k] = filled[..., k] | in_range
    return filled


def seed_flip_particles(
    fluid_mask: np.ndarray,
    grid: SimGrid,
    captured_points: np.ndarray,
    captured_velocities: np.ndarray,
    particles_per_cell: int = 13,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each fluid cell, place ``particles_per_cell`` random offsets within
    the cell and assign each particle the nearest captured point's
    velocity.

    Returns
    -------
    pos : (P, 3) float32 — world-space particle positions
    vel : (P, 3) float32 — world-space particle velocities (m/s)
    """
    fluid_idx = np.argwhere(fluid_mask)   # (M, 3) int32 — (i,j,k) of fluid cells
    M = len(fluid_idx)
    if M == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    rng = np.random.default_rng(rng_seed)
    P = M * particles_per_cell
    cell_offsets = rng.random((P, 3), dtype=np.float32)
    cell_idx = np.repeat(fluid_idx, particles_per_cell, axis=0).astype(np.float32)
    pos = grid.origin + (cell_idx + cell_offsets) * grid.dx

    # Velocity assignment: nearest-neighbour against captured surface points.
    # Lazy-import open3d so the rest of this module stays import-light.
    if len(captured_points) == 0:
        vel = np.zeros((P, 3), dtype=np.float32)
    else:
        import open3d as o3d  # noqa: WPS433
        captured_pcd = o3d.geometry.PointCloud()
        captured_pcd.points = o3d.utility.Vector3dVector(
            captured_points.astype(np.float64)
        )
        tree = o3d.geometry.KDTreeFlann(captured_pcd)
        nn_idx = np.empty(P, dtype=np.int32)
        for i, p in enumerate(pos):
            _k, idx_list, _ = tree.search_knn_vector_3d(p.astype(np.float64), 1)
            nn_idx[i] = int(idx_list[0])
        vel = captured_velocities[nn_idx].astype(np.float32)

    return pos.astype(np.float32), vel


def voxelise_and_seed(
    world: WorldFluid,
    dx: float = 0.015,
    headroom_below: float = 0.20,
    headroom_above: float = 0.05,
    headroom_lateral: float = 0.05,
    particles_per_cell: int = 13,
    rng_seed: int = 0,
) -> SeededState:
    """Full pipeline: pick grid, voxelise + fill, seed FLIP particles."""
    grid = make_grid_for_fluid(
        world.bounds_min,
        world.bounds_max,
        dx=dx,
        headroom_below=headroom_below,
        headroom_above=headroom_above,
        headroom_lateral=headroom_lateral,
    )

    surface_mask = voxelise_cloud(world.points, grid)
    fluid_mask = fill_vertical_columns(surface_mask)
    n_fluid = int(fluid_mask.sum())

    pos, vel = seed_flip_particles(
        fluid_mask=fluid_mask,
        grid=grid,
        captured_points=world.points,
        captured_velocities=world.velocities,
        particles_per_cell=particles_per_cell,
        rng_seed=rng_seed,
    )

    return SeededState(
        grid=grid,
        fluid_mask=fluid_mask,
        particles_pos=pos,
        particles_vel=vel,
        n_fluid_cells=n_fluid,
    )
