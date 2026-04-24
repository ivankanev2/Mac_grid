"""PLY I/O.

Handles the fused mesh produced by the Gaussian-splatting pipeline
(binary-little-endian PLY with per-vertex RGB + triangle faces, e.g.
``fuse_post.ply``) as well as plain point-cloud PLYs that downstream
modules may want to round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


@dataclass
class LoadedGeometry:
    """Bundle returned by :func:`load_mesh_or_points`.

    We always keep both a mesh (when available) and a point cloud view
    because the pipeline wants to *fit* on points but *export* a cleaned
    mesh.  Exactly one of `mesh` / `point_cloud_only` is truthy.
    """

    mesh: Optional[o3d.geometry.TriangleMesh]
    points: o3d.geometry.PointCloud  # always present; derived from mesh vertices if needed
    source_path: Path


def load_mesh_or_points(path: str | Path) -> LoadedGeometry:
    """Load a PLY file.  Prefers triangle mesh; falls back to point cloud.

    Parameters
    ----------
    path : str | Path
        Path to a `.ply` file (binary-LE or ASCII).

    Returns
    -------
    LoadedGeometry
        With ``mesh`` populated when the file contains faces, and
        ``points`` always populated (from vertices when it's a mesh).
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"PLY not found: {p}")

    # Try mesh first — this works for vertex+face PLYs like fuse_post.ply.
    mesh = o3d.io.read_triangle_mesh(str(p))
    has_faces = len(mesh.triangles) > 0
    has_verts = len(mesh.vertices) > 0

    if has_verts and has_faces:
        pts = o3d.geometry.PointCloud()
        pts.points = mesh.vertices
        if mesh.has_vertex_colors():
            pts.colors = mesh.vertex_colors
        return LoadedGeometry(mesh=mesh, points=pts, source_path=p)

    # Fallback: pure point cloud.
    pcd = o3d.io.read_point_cloud(str(p))
    if len(pcd.points) == 0:
        raise ValueError(f"PLY contained neither triangles nor points: {p}")
    return LoadedGeometry(mesh=None, points=pcd, source_path=p)


def save_mesh(path: str | Path, mesh: o3d.geometry.TriangleMesh) -> None:
    """Write a triangle mesh to a PLY file (binary-LE)."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(str(p), mesh, write_ascii=False)
    if not ok:
        raise IOError(f"Failed to write mesh: {p}")


def save_point_cloud(path: str | Path, pcd: o3d.geometry.PointCloud) -> None:
    """Write a point cloud to a PLY file (binary-LE)."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(str(p), pcd, write_ascii=False)
    if not ok:
        raise IOError(f"Failed to write point cloud: {p}")


def mesh_bbox(mesh_or_pcd) -> np.ndarray:
    """Return a ``(2, 3)`` ``[min; max]`` axis-aligned bounding box array."""
    aabb = mesh_or_pcd.get_axis_aligned_bounding_box()
    return np.stack([aabb.get_min_bound(), aabb.get_max_bound()], axis=0)
