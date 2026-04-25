"""Preview PLY writers — visualise the bridge output before any C++ is touched.

Two outputs:

* ``sim_state_particles.ply``: the seeded FLIP particles in world-frame
  coordinates, coloured by velocity magnitude (red = fast, blue = slow).
  Lets you see at a glance whether the fluid mass + velocity field
  ended up where you expect *after* orientation, scaling and voxel
  fill.

* ``sim_domain.ply``: a thin wireframe of the simulation bounding box,
  written as a triangle mesh of the 12 box edges.  Open it together
  with the particle PLY (``draw_geometries`` accepts a list) and
  confirm the fluid sits inside the domain with the headroom you
  asked for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from .voxelize import SeededState


def save_particles_preview(path: str | Path, state: SeededState) -> Path:
    pos = state.particles_pos
    vel = state.particles_vel

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos.astype(np.float64))

    if len(pos) > 0:
        vmag = np.linalg.norm(vel, axis=1)
        vmax = float(vmag.max()) if vmag.max() > 0 else 1.0
        norm = vmag / vmax
        colors = np.zeros((len(pos), 3), dtype=np.float64)
        colors[:, 0] = norm
        colors[:, 2] = 1.0 - norm
        pcd.colors = o3d.utility.Vector3dVector(colors)

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(p), pcd, write_ascii=False)
    if not ok:
        raise IOError(f"Failed to write {p}")
    return p


def save_domain_wireframe(path: str | Path, state: SeededState, edge_width: float = 0.0015) -> Path:
    """Write the simulation domain bounding box as 12 thin rectangular cuboids.

    Open3D's PLY writer doesn't preserve LineSet topology, so we extrude
    each edge into a thin tube (8 vertices, 12 triangles) instead.  The
    result renders identically to a wireframe in any PLY viewer.
    """
    g = state.grid
    lo = g.domain_min().astype(np.float64)
    hi = g.domain_max().astype(np.float64)

    # 8 corners of the box.
    corners = np.array([
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]],
        [lo[0], hi[1], hi[2]],
    ], dtype=np.float64)

    # 12 edges as pairs of corner indices.
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),   # top face
        (0, 4), (1, 5), (2, 6), (3, 7),   # vertical pillars
    ]

    all_verts = []
    all_tris = []
    base = 0

    for a, b in edges:
        pa = corners[a]
        pb = corners[b]
        d = pb - pa
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        # Build a local frame around the edge.
        axis = d / L
        up = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
        side = np.cross(axis, up); side /= (np.linalg.norm(side) + 1e-12)
        up = np.cross(side, axis); up /= (np.linalg.norm(up) + 1e-12)
        w = edge_width
        # 4 ring vertices at each end.
        ring = np.array([
            +w * side + +w * up,
            +w * side + -w * up,
            -w * side + -w * up,
            -w * side + +w * up,
        ], dtype=np.float64)
        v_a = pa + ring          # 4 verts
        v_b = pb + ring          # 4 verts
        verts = np.vstack([v_a, v_b])
        # 8 side triangles + 2 caps × 2 = 12 triangles.
        idx = np.array([
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
            [0, 1, 2], [0, 2, 3],   # cap A
            [4, 6, 5], [4, 7, 6],   # cap B
        ], dtype=np.int32)
        all_verts.append(verts)
        all_tris.append(idx + base)
        base += 8

    verts_all = np.vstack(all_verts) if all_verts else np.zeros((0, 3))
    tris_all = np.vstack(all_tris) if all_tris else np.zeros((0, 3), dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_all)
    mesh.triangles = o3d.utility.Vector3iVector(tris_all)
    mesh.compute_vertex_normals()

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(p), mesh, write_ascii=False)
    if not ok:
        raise IOError(f"Failed to write {p}")
    return p
