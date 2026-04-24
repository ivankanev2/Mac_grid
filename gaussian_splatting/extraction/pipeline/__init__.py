"""Pipe extraction pipeline.

Submodules:
    io_ply            — load / save PLY meshes and point clouds.
    denoise           — voxel downsample + statistical outlier removal.
    crop_interactive  — Open3D viewer for picking a bounding box.
    cylinder_fit      — PCA axis + robust radius + iterative refinement.
    centerline        — endpoints, length and direction from axis projection.
    mesh_refine       — crop the original high-res mesh by the fit cylinder.
    blueprint_writer  — emit a .pipe blueprint consumable by pipe_fluid_engine.
"""

__all__ = [
    "io_ply",
    "denoise",
    "crop_interactive",
    "cylinder_fit",
    "centerline",
    "mesh_refine",
    "blueprint_writer",
]
