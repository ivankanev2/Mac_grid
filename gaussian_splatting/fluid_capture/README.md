# Fluid Capture (Milestone 1)

Take a video of a moving fluid and reconstruct a 3D point cloud of the
fluid's surface — with per-point velocity vectors — at one chosen
moment in time.  No simulator hand-off yet (that's Milestone 2); this
folder is purely about validating that monocular video → 3D fluid state
reconstruction *works at all* on our chosen video.

The pipeline is **monocular and feed-forward**:

1. **Extract frames** at the hand-off moment and one frame later.
2. **Predict depth** on both frames via Depth Anything V2.
3. **Predict optical flow** between the two frames via RAFT.
4. **Segment the fluid** out of the background by colour (HSV thresholding
   for yellow olive oil; tuneable for other fluids).
5. **Lift to 3D**: pinhole back-projection of fluid pixels using their
   depth, plus per-pixel 3D velocities derived from optical flow + the
   depth on the second frame.
6. **Export** a PLY of the fluid points coloured by image RGB, a second
   PLY coloured by velocity magnitude, and debug PNGs for each stage.

## Install

Requires **Python 3.9 – 3.11**.  Reuse the `pipe-extract` conda env from
the pipe extraction work and add the new dependencies on top:

```bash
conda activate pipe-extract
cd gaussian_splatting/fluid_capture
pip install -r requirements.txt
```

First run will download Depth Anything V2 weights (~400 MB) and RAFT
weights (~25 MB) into your local caches.

## Run

```bash
python capture_fluid.py \
  --input  input/videoplayback-3 \
  --output-dir outputs/ \
  --time   4.0
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--time` | `4.0` | Hand-off timestamp in seconds. |
| `--dt`   | `0.08` | Time gap between the frame pair fed to optical flow (seconds). Higher = more motion to track but breaks if too high. |
| `--hue-min`, `--hue-max` | `15`, `45` | OpenCV HSV hue range for fluid segmentation.  Defaults are olive-oil yellow.  Wider for full-spectrum fluids. |
| `--depth-model` | `depth-anything/Depth-Anything-V2-Base-hf` | HuggingFace model id.  Use `Small-hf` for faster but rougher depth, `Large-hf` for higher quality at ~3× the runtime. |
| `--device` | (auto) | `mps` on Apple Silicon, `cuda` on NVIDIA, otherwise `cpu`. |
| `--fov`    | `60.0` | Assumed horizontal FOV in degrees for intrinsics estimate.  Affects scale only, not topology. |

## Outputs

Saved into `--output-dir`:

- `fluid_points_rgb.ply` — fluid surface points coloured by image RGB.
- `fluid_points_vmag.ply` — fluid surface points coloured by velocity
  magnitude (red = fast, blue = slow).  Use this to verify the velocity
  field looks plausible.
- `fluid_state.npz` — raw arrays (positions, velocities, colours) for
  Milestone 2 to ingest without re-running the heavy models.
- `debug/` — PNGs of depth heatmap, optical-flow HSV visualisation, and
  segmentation overlay.  First place to look if the point cloud looks wrong.

## Open the PLYs

```bash
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('outputs/fluid_points_rgb.ply')])"
```

(Same trick as the pipe extraction.  Drag-rotate to inspect; if the oil
column + pool show up recognisably, the reconstruction is working.)

## Known limits (v1)

- **Monocular depth is up-to-scale** — the reconstruction is shape-correct
  in a relative sense but absolute distances are arbitrary.  Milestone 2
  resolves the scale when we voxelise into the simulator's grid.
- **The back side of the fluid is hallucinated.**  Depth is only
  measured for the camera-facing surface; what lies behind is whatever
  the network's prior decided.  Fine for the demo (we'll only render
  the front), worth knowing for honesty.
- **Yellow-on-white segmentation only** — the segmentation step is
  HSV-threshold-based.  Works for olive oil / honey on a clean
  background; fails for transparent water on transparent backgrounds.
  We can swap in SAM or a learned segmenter later if needed.
