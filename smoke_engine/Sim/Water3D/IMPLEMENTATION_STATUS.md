# 3D Water Status

This folder is the separated 3D water implementation that lives beside the 2D solver in `Sim/Water/`.

## Implemented in the 3D path

- Separate 3D simulation facade: `MACWater3D`
- 3D staggered MAC face velocities: `u`, `v`, `w`
- 3D particle storage and RK2 particle advection
- Particle-to-grid and grid-to-particle PIC/FLIP transfers
- 3D APIC affine transfers
- 3D liquid occupancy mask with optional dilation
- 3D free-surface pressure projection on the liquid region
- Reusable 3D multigrid-capable Poisson pressure solver (`PressureSolver3D`)
- 3D velocity extrapolation into nearby air
- 3D implicit velocity diffusion
- 3D solid voxel mask, border walls, and face boundary handling
- 3D spherical source insertion
- 3D particle reseeding and dissipation
- 3D debug fields: water, pressure, divergence, speed
- Arbitrary `XY`, `XZ`, and `YZ` debug slice extraction
- Runtime UI/renderer integration for selecting and viewing the 3D solver inside the app
- Build integration without disturbing the current 2D app path
- Optional CUDA backend source file and CMake wiring behind `SMOKE_ENABLE_CUDA`
- CUDA runtime backend selection with device detection and CPU fallback when no CUDA device is available

## Still missing compared with the long-term target

- Shared 3D smoke/liquid coupled simulation path
- Signed-distance / cut-cell boundary treatment for pipe geometry
- Quantitative validation against benchmark 3D free-surface tests
- Verified NVIDIA-side compilation and runtime validation for the CUDA path
- Exact feature parity with every 2D-only tuning and diagnostics path

## Practical meaning

Today:
- The 2D app path still remains intact.
- The separated 3D module is a real CPU MAC water solver, not just a particle prototype.
- The 3D pressure path now uses a reusable multigrid-capable solver instead of only per-step RBGS/Jacobi relaxation.
- There is still a source-only CUDA backend for the 3D solver, but it has not been compiled or run in this workspace.
- Enabling `SMOKE_ENABLE_CUDA=ON` is intended for a Linux or Windows machine with NVIDIA drivers, CUDA, and `nvcc` installed.
