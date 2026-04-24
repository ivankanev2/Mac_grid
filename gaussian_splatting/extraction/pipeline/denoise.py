"""Point-cloud denoising helpers.

The frayed splash field around the pipe in ``fuse_post.ply`` is dense,
irregular noise that dominates a naive bounding-box crop.  We attack it
with two orthogonal tools:

* :func:`voxel_downsample` — enforce a minimum spacing between points
  (collapses dense clumps, reduces RANSAC/PCA work).
* :func:`statistical_outlier_removal` — drop points whose neighbourhood
  mean distance is far above the cloud's own average.

The defaults are tuned for the scale of ``fuse_post.ply`` (metres, pipe
radius on the order of centimetres).  Override via the CLI for scenes at
different scales.
"""

from __future__ import annotations

import open3d as o3d


def voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.005,
) -> o3d.geometry.PointCloud:
    """Downsample the cloud to at most one point per ``voxel_size``³ cube."""
    if voxel_size <= 0.0:
        return pcd
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def statistical_outlier_removal(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """Remove points whose mean k-NN distance is > ``std_ratio`` σ above the
    cloud average.  Good at killing isolated splashes and thin fringes.
    """
    if len(pcd.points) == 0:
        return pcd
    filtered, _ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return filtered


def radius_outlier_removal(
    pcd: o3d.geometry.PointCloud,
    nb_points: int = 8,
    radius: float = 0.02,
) -> o3d.geometry.PointCloud:
    """Remove points with fewer than ``nb_points`` neighbours within
    ``radius`` metres.  Complement to statistical removal for very sparse
    stragglers.  Off by default — use when needed.
    """
    if len(pcd.points) == 0:
        return pcd
    filtered, _ind = pcd.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius,
    )
    return filtered
