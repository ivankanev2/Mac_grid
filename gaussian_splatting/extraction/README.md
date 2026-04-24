# Pipe Extraction

Turn a noisy 2D-Gaussian-Splatting reconstruction of a pipe (e.g. `fuse_post.ply`) into a clean, parametric pipe that plugs directly into `pipe_fluid_engine`.

## What it does

1. **Load** the fused mesh produced by the GS pipeline (supports binary-little-endian `.ply` with per-vertex colour and faces).
2. **Crop** — interactive 3D viewer; you pick a bounding box around the pipe so we only work with the region of interest.
3. **Denoise** — voxel down-sample + statistical outlier removal to clean the frayed splash field around the pipe.
4. **Fit** — PCA to estimate the pipe axis, median orthogonal distance to estimate the radius, iterative refinement to drop radial outliers.
5. **Parameterise** — project inliers onto the axis to derive endpoints, length and a normalised direction.
6. **Emit** — a `.pipe` blueprint file (directly consumable by `pipe_fluid_engine::loadBlueprintFile`) **and** a cleaned triangle mesh (cropped to the fit cylinder, for visual reference).

## Install

Requires **Python 3.9 – 3.11**.  Open3D's wheels lag Python releases by a
few months; 3.12 and 3.13 aren't fully supported on every platform yet.
If `python --version` reports 3.12 or newer, make a dedicated env first:

```bash
conda create -n pipe-extract python=3.11 -y
conda activate pipe-extract
```

(or equivalent with plain `venv` on a system Python 3.11.)

Then:

```bash
cd gaussian_splatting/extraction
pip install -r requirements.txt
```

### Troubleshooting

**`ERROR: Could not find a version that satisfies the requirement open3d`** —
your Python is too new (3.12+) and pip can't find a matching Open3D wheel.
Follow the conda / venv recipe above to get a 3.11 interpreter, then
re-run the install.

**Open3D import crashes on macOS with "symbol not found"** — you likely
have mixed Python architectures (x86_64 vs arm64).  Make sure the Python
in your env matches your machine (on Apple Silicon, use an arm64 Python
via conda-forge or the Miniforge installer).

## Run

Interactive (default):

```bash
python extract_pipe.py \
    --input  /path/to/fuse_post.ply \
    --output-dir outputs/
```

The viewer will open; **Shift + Left-Click** at least two points that together bracket the pipe (opposite corners of a loose box around it is enough), then close the window. The rest runs automatically and prints a summary.

Non-interactive (for scripted runs — bbox is in world units):

```bash
python extract_pipe.py \
    --input  /path/to/fuse_post.ply \
    --output-dir outputs/ \
    --bbox   "-0.5,-0.5,0,0.5,0.5,2.0" \
    --no-interactive
```

## Outputs

In `--output-dir` (default `outputs/`):

- `pipe.pipe` — parametric blueprint (name / inner_radius / outer_radius / start / direction / straight L)
- `pipe_clean.ply` — cleaned triangle mesh (original high-res mesh cropped to the fit cylinder)
- `pipe_inliers.ply` — the point-cloud inliers used for the fit (useful for debugging the fit quality)

## Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--wall-thickness` | `0.01` m | Inner radius = outer_radius − wall_thickness |
| `--voxel-size` | `0.005` m | Downsample resolution for fitting (does not affect final mesh export) |
| `--outlier-nb` | `20` | Neighbours for statistical outlier removal |
| `--outlier-std` | `2.0` | Std-ratio threshold for statistical outlier removal |
| `--radial-tol` | `2.0` | Iterative refinement drops points beyond this × median radius |
| `--trim-frac` | `0.02` | Trim this fraction of axial extent from each end cap (noise rejection) |

## Scope (v1)

This first version handles **a single straight pipe**. Bent pipes and networks will land on top of the same skeleton: the `cylinder_fit` and `centerline` modules will gain a piecewise / graph variant, and `blueprint_writer` will grow `bend` / `bend90` emission.

## Importing into the simulation

Once `pipe.pipe` is produced, load it from the fluid scene:

```cpp
pipe_fluid::PipeFluidScene scene(cfg);
auto r = pipe_fluid::loadBlueprintFile(
    "gaussian_splatting/extraction/outputs/pipe.pipe",
    scene.network());
scene.rebuild();
```

or from the viewer's **Load blueprint** UI.
