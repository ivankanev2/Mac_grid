# Dynamic Capture (4DGS pipeline)

Tooling for option 3 of the video-to-simulation roadmap: reconstruct a
*time-varying* 3D fluid field from a monocular video using Deformable
3D Gaussians, then hand the per-frame fluid state to the simulator as a
replay or blend constraint.

This folder is the sibling of `extraction/` (static pipe extraction) and
`fluid_capture/` (M1 monocular depth + flow → single-snapshot bridge).
The static / single-snapshot pipelines are *not* superseded — they're
faster and they validate the end-to-end concept — but for research-grade
demos that visually reproduce the source video, this is the path.

## Status

- **`prep_video_for_deformable_gs.py`** — convert a static-camera
  monocular video into the D-NeRF / Blender JSON format that
  Deformable 3D-GS's `--is_blender` loader consumes. Done.
- 4DGS training is then run via the upstream
  [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)
  repo on a CUDA workstation. Not part of this folder; see the
  workstation install notes in the project root.
- Per-frame fluid state extraction (query the trained 4DGS at each
  video timestep, voxelise into the simulator's MAC grid). Pending.
- Static / dynamic decomposition (extract the receiving bottle as a
  solid mesh, oil as the dynamic field). Pending.
- Simulator replay-mode + blend-mode hooks. Pending.

## Install

This script has a tiny dependency surface — just `opencv-python` and
`numpy`. You can install it into any conda env or just into the system
Python 3.8+. On the workstation:

```bash
pip install -r requirements.txt
```

(The Deformable 3D-GS conda env you set up earlier already includes
both, so the cleanest path is just to use that env.)

## Run: prep a video for 4DGS training

```bash
python prep_video_for_deformable_gs.py \
    --input      /path/to/videoplayback-3 \
    --output-dir data/oil_pour/ \
    --fov        60.0
```

Default behaviour:

- Every frame extracted at the video's native frame rate (≈ 200 frames
  for an 8-second 25 fps clip).
- Sequential 80 / 10 / 10 split: first 80 % of frames go to `train`,
  next 10 % to `val`, last 10 % to `test`. The held-out test set is
  the *end* of the clip, so reported PSNR genuinely measures the
  deformation MLP's ability to extrapolate to moments it never saw.
- Static camera at `(0, 0, 4)` looking at the origin, identical
  `transform_matrix` for every frame. Only the `time` field varies in
  the per-frame JSON entries.
- Camera intrinsics derived from `--fov` (default 60° horizontal FOV).
  Wrong FOV scales the reconstruction uniformly but does not warp it.

After running, the directory layout matches what the upstream
`train.py -s <output-dir> --eval --is_blender` expects:

```
data/oil_pour/
  transforms_train.json
  transforms_val.json
  transforms_test.json
  train/
    r_0.png
    r_1.png
    ...
  val/
    r_0.png
    ...
  test/
    r_0.png
    ...
```

## Why static camera + varying time?

Deformable 3D-GS treats the scene as a canonical Gaussian point cloud
plus a time-conditioned deformation MLP. The deformation MLP is fed
`(canonical_position, time)` and outputs a per-Gaussian offset. So all
the time-varying content (the falling oil, the splashing pool) is
captured in the deformation field, not in the camera trajectory.

A static-camera video is actually a *cleaner* input than a moving-camera
video for this model: the camera contributes no signal we have to
disentangle from the scene's own motion. The only adaptation we make
is feeding the same camera pose at every time, which the model handles
without any code changes — the deformation field just learns to explain
all per-pixel temporal variation.

## Limitations

- Camera intrinsics are heuristic (single FOV value). For higher
  fidelity, run a brief calibration on a known-size object in frame
  and update `--fov` accordingly.
- The 80/10/10 split on a sequential clip means the test set is on
  later (probably more interesting / harder) moments. PSNR on test
  will be lower than on train; that's intentional — it measures real
  generalisation.
- Video must be from a fixed camera. For hand-held / moving camera
  videos, you'd need COLMAP camera-pose recovery as a separate
  pre-processing step (out of scope for this script).
