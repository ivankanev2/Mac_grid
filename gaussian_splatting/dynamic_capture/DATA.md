# Binary artifacts — not in git

Large files for this project live outside the repo. To bootstrap:

1. **Trained Deformable 3D-GS model** → `gaussian_splatting/checkpoints/oil_pour/`
   Source: <Google Drive (https://drive.google.com/drive/folders/1gukh4B0GIc17jNGO9AfZGUUmwjM8NHLM?usp=sharing)>
   Approx 300 MB. Required to run `extract_fluid_state_v2.py`.

2. **Captured fluid states v2** → `gaussian_splatting/dynamic_capture/captured_states_v2/`
   Source: <link>
   Approx 1 GB (501 frames). Required for simulator replay.
   Can be regenerated from the trained model in ~5 min via:
       python extract_fluid_state_v2.py --model-dir output/oil_pour --iteration 40000 \
         --output-dir captured_states_v2

3. **Diagnostic PLYs** → `gaussian_splatting/dynamic_capture/diag_motion/`
   Source: <link>
   Approx 30 MB. Used for the histogram + class PLY figures in the paper.
   Regenerate with:
       python diagnose_motion_classes.py --model-dir output/oil_pour --iteration 40000 \
         --output-dir diag_motion

4. **Training data** (optional, only if re-training) → `gaussian_splatting/data/oil_pour/`
   Source: <link>
   Approx 1 GB. Pre-processed transforms_train/val/test.json + frame PNGs.

## Conda env
See `environment.yaml` in this folder. Recreate with:
   conda env create -f environment.yaml -n deformable_gaussian_env
