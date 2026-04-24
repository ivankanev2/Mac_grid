"""Emit a ``.pipe`` blueprint consumable by pipe_fluid_engine.

Format reference (see pipe_engine/Blueprint/blueprint_parser.h):

    name <string>
    inner_radius <float>      # metres
    outer_radius <float>      # metres
    start <x> <y> <z>         # metres
    direction <dx> <dy> <dz>  # unit-length (parser normalises, but we
                              # emit a normalised vector anyway)
    straight <length>         # metres

Bend commands (``bend``, ``bend90``) are supported by the parser but not
emitted here — v1 of the extraction pipeline handles straight pipes only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .centerline import Centerline


def write_straight_pipe(
    path: str | Path,
    centerline: Centerline,
    outer_radius: float,
    wall_thickness: float = 0.01,
    name: str = "extracted_pipe",
    header_comment: Optional[str] = None,
) -> Path:
    """Write a ``.pipe`` blueprint with a single straight segment.

    Parameters
    ----------
    path : str | Path
        Destination `.pipe` file.
    centerline : Centerline
        From :func:`derive_centerline`.  Provides start / direction / length.
    outer_radius : float
        Fitted cylinder radius, in metres.
    wall_thickness : float
        Subtracted from ``outer_radius`` to yield the inner radius.  Must
        be strictly less than ``outer_radius``.
    name : str
        Name field for the blueprint (``net.name``).
    header_comment : str, optional
        Free-form comment block to include at the top of the file — e.g.
        the input PLY path and the fit diagnostics.
    """
    if wall_thickness <= 0:
        raise ValueError(f"wall_thickness must be > 0, got {wall_thickness}")
    if wall_thickness >= outer_radius:
        raise ValueError(
            f"wall_thickness ({wall_thickness}) must be strictly less than "
            f"outer_radius ({outer_radius}); otherwise inner_radius ≤ 0"
        )

    inner_radius = float(outer_radius - wall_thickness)
    direction = centerline.direction / (
        np.linalg.norm(centerline.direction) + 1e-12
    )

    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        if header_comment:
            for line in header_comment.splitlines():
                f.write(f"# {line}\n")
            f.write("\n")

        f.write(f"name {name}\n")
        f.write(f"inner_radius {inner_radius:.6f}\n")
        f.write(f"outer_radius {float(outer_radius):.6f}\n\n")

        f.write(
            f"start {centerline.start[0]:.6f} "
            f"{centerline.start[1]:.6f} "
            f"{centerline.start[2]:.6f}\n"
        )
        f.write(
            f"direction {direction[0]:.6f} "
            f"{direction[1]:.6f} "
            f"{direction[2]:.6f}\n\n"
        )

        f.write(f"straight {centerline.length:.6f}\n")

    return out_path
