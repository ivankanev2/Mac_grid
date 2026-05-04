# Paper Journal — Video to Fluid Simulation

**Author:** Ivan Kanev (+ teammates: 3 collaborators on writing; ViewCrafter integration coworker)
**Target venue:** SIGGRAPH Asia Technical Papers
**Submission deadline:** May 12, 2026
**Last updated:** May 4, 2026

### Machines and paths
| Where | Account / host | Project root |
|---|---|---|
| Mac (this machine) | `ivan.kanev@LM007492` | `~/Desktop/jorge_project/` |
| Workstation (CUDA, runs 4DGS extraction) | `jorge.herrera@mbzuaiadmin-AS-5014A-TT` | `~/Deformable-3D-Gaussians/` |
| Workstation conda env | `deformable_gaussian_env` | (activates via `conda activate`) |

Workstation file layout differs from the Mac:
- **Mac:** `~/Desktop/jorge_project/gaussian_splatting/dynamic_capture/...`
- **Workstation:** `~/Deformable-3D-Gaussians/dynamic_capture/...` (no `gaussian_splatting/` parent — workstation puts the dynamic_capture folder directly under the Deformable-3D-Gaussians repo it was forked from)

So the **same file** lives at:
- Mac: `~/Desktop/jorge_project/gaussian_splatting/dynamic_capture/extract_column_emitter.py`
- Workstation: `~/Deformable-3D-Gaussians/dynamic_capture/extract_column_emitter.py`

### Transferring data between machines

Single file (e.g. just `column_emitter.bin` after re-running the emitter
extractor): straight scp is fastest.

Bulk transfer (after re-running `extract_fluid_state_v2.py`, which writes 501
sim_state files): tar first.

```bash
# Workstation: bundle
ssh jorge.herrera@mbzuaiadmin-AS-5014A-TT
cd ~/Deformable-3D-Gaussians/dynamic_capture
tar czf captured_states_v2.tar.gz captured_states_v2/
exit

# Mac: pull and unpack carefully (preserve the _yup files)
scp jorge.herrera@mbzuaiadmin-AS-5014A-TT:~/Deformable-3D-Gaussians/dynamic_capture/captured_states_v2.tar.gz \
    ~/Desktop/jorge_project/gaussian_splatting/dynamic_capture/

cd ~/Desktop/jorge_project/gaussian_splatting/dynamic_capture
mv captured_states_v2 captured_states_v2.bak.$(date +%s)
tar xzf captured_states_v2.tar.gz
# Re-run transposes on the new files (transpose_zup_to_yup.py)
```

**Watch out**: the workstation never has the `_yup` files (they're generated
on the Mac). Re-tarring from the workstation and unpacking on the Mac
overwrites the entire folder, so the transposes need to be re-run after each
bulk transfer. The .bak folder is the safety net.

This document is a living chronology of the technical journey, the dead ends, the
breakthroughs, and the open questions, intended to be the source for the
Methods, Limitations, and "Challenges Encountered" sections of the paper. It
also serves as a stable artefact so that, if my chat history with Claude is
truncated again, the project context can be recovered from disk.

---

## 1. Project overview

### One-line description
Take a single static-camera monocular video of olive oil pouring into a glass
bottle, and produce a physically-plausible re-simulation of the same pour
inside a MAC FLIP/APIC fluid simulator.

### Source data
- A 20-second, 25 fps, 1920x1080 video (`videoplayback-3`) of olive oil being
  poured into a wide glass bottle. The camera is fixed, the bottle is fixed,
  only the falling oil column and the oil pool inside the bottle are dynamic.
- Camera framing crops the bottle's neck and mouth out of frame; only the bottom
  ~60-70% of the bottle is visible.

### Pipeline (current state, May 4)
```
                                      Image-to-3D mesh (TripoSG)
                                              |
                                              v
   Monocular video  -->  Frame 0  -->  bottle_from_mesh.py  -->  bottle_solid.bin
        |
        v
   Deformable 3DGS training (workstation)
        |
        v
   Trained 4DGS model
        |
        +-->  extract_fluid_state_v2.py  -->  sim_state_*.bin (per-frame)
        |
        +-->  extract_column_emitter.py  -->  column_emitter.bin
                                                    |
                                                    v
                                       transpose_zup_to_yup.py
                                                    |
                                                    v
                                       MAC FLIP simulator (pipe_fluid_engine)
                                                    |
                                                    v
                                       Re-simulated pour
```

### Three contribution rows for the ablation table
1. **Bottle from captured 4DGS Gaussians** (baseline) — shows the limitation of
   monocular Deformable 3DGS on refractive containers.
2. **Bottle from image-to-3D (TripoSG) + voxelizer** — the cleaner static
   reconstruction; pipeline-side contribution.
3. **Bottle from coworker's ViewCrafter-augmented 4DGS** — when delivered.

---

## 2. Chronological journey

### Phase A — Set up Deformable 3DGS on the workstation
*Tasks #65–68.*

Installed Miniconda, created `deformable_gaussian_env`, ran the Deformable 3DGS
baseline on D-NeRF synthetic (Lego). Visually verified the trained Lego
reconstruction looked like Lego. No surprises here.

### Phase B — Reconstruct the oil_pour scene
*Tasks #69–70 (B1 done, B2 still in progress).*

Wrote `prep_video_for_deformable_gs.py` to convert the source video into the
COLMAP-conventioned dataset Deformable 3DGS expects. Trained on the oil-pour
sequence. The model converged but the static container reconstruction was
chunky, blocky, and looked nothing like a bottle — visible in the dumped
canonical Gaussian point cloud.

> **Why this happened (paper insight):** Monocular static-camera Deformable 3DGS
> has no parallax, so depth along the camera ray is poorly constrained.
> The deformation MLP fits the visual pattern but the geometry is
> under-determined, especially for transparent / refractive surfaces where
> there's almost no Lambertian signal to triangulate against.

### Phase C — Extracting per-frame fluid state
*Tasks #71–78 (extract_fluid_state.py v1) and #79–82 (v2).*

Built `extract_fluid_state.py` to:
- Query the trained Deformable 3DGS at each captured time
- Segment Gaussians by HSV (yellow = oil, grey = bottle)
- Voxelise into a MAC grid at dx=1.5cm
- Emit per-frame `sim_state_*.bin` files

Several iteration cycles:
- **C1.5:** Added motion-based segmentation when HSV alone misclassified.
- **C1.6:** Replaced naive vertical-fill with morphological dilation —
  vertical-fill produced a column of cells from the top of every Gaussian
  cluster down to the floor of the grid, which is wrong for a free-falling
  oil stream.
- **v2:** Added bottle extraction (`bottle_solid.bin`) alongside the oil state,
  with low motion threshold to discriminate static container from moving fluid.
- **v2.1:** Percentile-clipped bottle bbox to drop outlier Gaussians; switched
  from oil-AND-NOT-bottle to direct oil-class voxelisation.
- **v2.2:** Added explicit oil-to-bottle xy alignment because the captured
  oil's xy centroid drifted relative to the bottle's xy centroid frame to frame.

### Phase D — Simulator integration
*Tasks #58–64, #80.*

Wrote `fluid_state_loader.h/.cpp` (binary reader for `sim_state_*.bin`,
`bottle_solid.bin`, `column_emitter.bin`). Added
`PipeFluidScene::loadFluidState`, `loadFluidStateSeries`, and
`loadBottlePourDemo` paths. Wired up the viewer's CLI flags
(`--fluid-state-series`, `--bottle-pour-demo`, `--column-emitter`).

> **Architectural choice:** Two replay modes coexist.
> Replay mode (`loadFluidStateSeries`) teleports particles to each captured
> frame and suppresses physics between frames — produces a "captured shape
> that wiggles". Demo mode (`loadBottlePourDemo`) loads only the static bottle
> and runs a continuous emitter, producing organic falling/splashing/pooling
> motion. The paper's headline result lives in demo mode.

### Phase E — Bottle solid filters
*Tasks #87, #91–93.*

Iterated five filters on the captured-Gaussian bottle to suppress noise:
1. Wall-only filter (8-connected boundary cells)
2. Connected-components filter (largest component, 26-connected)
3. Exterior-facing wall filter (flood-fill from grid boundary)
4. Topological closed-base prior (elliptical disk seal at k_floor)
5. xy/xz centering of base seal

Each filter improved one failure mode but exposed another:
- Wall-only filter missed sparse interior noise
- Connected-components filter missed noise diagonally connected to walls
- Exterior-facing filter was too aggressive on 1-cell walls (642 of 835 cells dropped)

> **Lesson (paper insight):** Each filter is a topological prior, but the
> underlying 4DGS reconstruction itself is bad — none of the filters can
> recover what the model never reconstructed. This realisation drove the
> pivot to the factored approach in Phase G.

### Phase F — Two emitter strategies
*Tasks #83–86, #88–89.*

**Tier A** — bottle pour demo via continuous emitter from a fixed position
above the bottle. Worked as a proof of concept but the pour was hardcoded.

**Tier B** — extract per-frame column emitter trajectory from the captured oil
column's mean xy and gravity-derived velocity. `extract_column_emitter.py`
wrote a binary trajectory file the viewer drove `addWaterSourceSphere` from.

**Tier B v2** — switched from MLP-output velocity (unreliable in monocular
static-camera setting) to a gravity-derived velocity computed from the
captured top-z minus the emitter z. Fall height comes from data;
gravity is 9.8.

### Phase G — The image-to-3D pivot
*Tasks #95–101.*

User: "look the thing is we dont even have a stable simulation i dont know
what you are talking about but the particles are flying everywhere and its
leaking eveyrwherre this is nothing, tell me is there a way to get a better
model than this"

Reframing: separate the static reconstruction problem (bottle) from the
dynamic capture problem (oil column). Use image-to-3D for the static bottle,
keep 4DGS for the dynamic column. Each method plays to its strength.

> **Why this works (paper insight):** Image-to-3D models (TripoSR / TripoSG /
> Hunyuan3D / TRELLIS) are trained on Objaverse and are excellent at static
> object reconstruction from a single view. They fail on motion, but we
> don't need them to handle motion — that's the 4DGS's job.

Wrote `bottle_from_mesh.py`: voxelise an image-to-3D mesh into the simulator's
`bottle_solid.bin` format. Loads via trimesh, optionally rotates y-up to z-up
(more on that below), scales by manifest's `bottle_height_target`, voxelises
at dx, optionally fills holes via scipy.

#### Image-to-3D model survey (negative + positive results)

Tried several HuggingFace Spaces with `frame_0000.png` (the cropped bottle
view, no oil column):

| Model | Result | Notes |
|---|---|---|
| TripoSR (HF Space) | broken (`onnxruntime` import error in container) | infrastructure issue, not model |
| Hunyuan3D-2 | very bad — produced a melted-cup blob | model collapsed on cropped, transparent input |
| TRELLIS / InstantMesh | broken HF Spaces (ZeroGPU worker error) | infrastructure |
| **TripoSG** | **good silhouette, hollow interior, single back-side gap** | **Used in pipeline.** Triangular gap on back where camera never saw |

> **Negative result worth a paper figure:** Hunyuan3D-2's chunky-cup output is
> a useful illustration that "newer/bigger isn't always better" for non-canonical
> inputs. TripoSG's success on the same input shows that model architecture
> (rectified flow + better spatial prior) matters more than parameter count.

#### Cleaner input image preparation
The original 1920x1080 frame had the bottle off-centre with ~40% pinkish wall
in shot. Image-to-3D models internally crop to square and resize to ~512px,
so they were seeing the bottle reduced to a tiny portion. Variants tried:
v1 centered crop (missed bottle, wrong centre estimate), v4 wider crop, v5
full-bottle replicate-padded square (this worked), v6 white-padded, v7
GrabCut-segmented (chewed up the top edges due to transparency).

V5 (full bottle, replicate-padded) was the winning input format.

### Phase H — The coordinate system bug
*Tasks #102–106.*

This is the most embarrassing-but-paper-worthy lesson of the whole project.

#### The misdiagnosis
After voxelising the TripoSG mesh, the bottle in the viewer looked rotated.
First diagnosis: TripoSG had hallucinated a closed cap on top of the bottle
(known failure mode — image-to-3D models trained on complete objects
hallucinate completion when given a partial object). Added a `--carve-top-cells`
flag. The bottle still looked rotated.

User correctly pushed back: "but may i ask when i generated the 3d models i
didnt see no cap, there wasnt a cap, are you sure its not something that we
did is the problem".

#### The actual diagnosis
Three pieces of evidence aligned:
1. `extract_fluid_state_v2.py` explicitly aligns gravity to `[0, 0, -1]` —
   data is **Z-up**.
2. `MACWater3D::applyExternalForces` applies gravity to V (Y) velocity
   component — simulator is **Y-up**.
3. The C++ loader (`PipeFluidScene::loadBottleSolid`,
   `loadBottlePourDemo`) reads the bytes verbatim with no axis swap.

**The whole pipeline has been writing data with the bottle vertical along Z,
but the simulator treats Y as vertical.** Every bottle has been laying on its
side from day one. Gravity has been pulling fluid sideways relative to every
container in every simulation we've ever run. Many of the prior frustrations
("fluid leaks everywhere", "particles fly out the sides") had this as the
root cause. The original captured-Gaussian bottle didn't *look* obviously
rotated only because its Y extent (one of its lateral dimensions) happened
to be its largest dimension, so the rendered silhouette looked tall-ish.

#### The fix
**Option A (chosen):** convert all data files from Z-up to Y-up by transposing
Y and Z axes. Wrote `transpose_zup_to_yup.py` that handles BSL1 (bottle solid),
CEM1 (column emitter), and the manifest JSON. Originals preserved with `_yup`
suffix for the new files.

**Option B (rejected):** modify the simulator engine to apply gravity to W
(Z velocity) instead of V. Cleaner architecturally but much riskier — the
engine has many other consumers.

After applying option A, the bottle stood visibly upright in the viewer for
the first time. The dense-base-at-low-Y, walls-going-up, open-mouth-at-high-Y
profile was finally correct.

> **Lesson (paper-worthy in Limitations / Lessons):** Coordinate-system
> conventions silently broken between data-extraction and simulation can
> waste weeks of debugging by manifesting as plausible-looking-but-wrong
> physics. A defensive measure: write a runtime "gravity check" in the
> simulator that on first step applies a tiny test impulse and verifies the
> direction matches the loaded data's stored gravity vector.

### Phase J — Particle-direct emission (replaces sphere)
*Tasks #111–114.*

User asked the right pointed question after seeing the fat-sphere bursts:
"why do we have this clamp, it seems all of our issues stem from this, also
why are we spawning the fluid as a sphere, isnt that like a cheat or something
simulation sided rather than pulling from the video".

Both of those are sim-side abstractions — `addWaterSourceSphere` was the
existing simulator API and the radius clamp was a discretisation safety:
sub-cell spheres miss cell-centres and emit nothing.

Replaced with particle-direct injection: `MACWater3D::addParticlesDirect`
pushes individual particles into the simulator at exact (sub-cell) positions
with per-particle velocity. The viewer's column-emitter loop now
(a) computes a particles-per-second rate as
`amount * V_sphere / V_per_particle` (so the time-averaged volume rate is
unchanged from the sphere version), and (b) injects particles at the captured
xz centre with sub-cell jitter and gravity-derived velocity.

Result: a continuous thin stream that visually matches the source video,
not a fat-sphere strobe. Sphere mode kept behind a UI checkbox for the
ablation table.

> **Paper-worthy framing:** "the captured column emits particles directly at
> its data-derived xz centre and gravity-derived velocity, at a per-frame
> rate from the bottle-fill flow estimator". No sphere caveat needed.

### Phase I — The flow rate problem
*Tasks #107–109.*

After the coordinate fix, the bottle stood upright but the fluid filled it in
under 2 seconds, dramatically over-filling and overflowing — meanwhile the
real video shows a slow gradual fill over ~10-15 seconds.

#### Diagnosis
Two compounding issues:
1. The captured column's true xy spread (measured 95th percentile in
   `extract_column_emitter.py`) is ~5mm radius, but it gets clamped UP to
   1.5*dx = 22.5mm because at dx=1.5cm a sphere smaller than ~1 cell can't
   reliably hit any cell centre. The simulator's emit sphere is therefore
   ~4.5x wider than reality, producing ~91x more volume per emit.
2. The simulator's `addWaterSourceSphere` is called once per substep — at
   60 Hz with multiple substeps that's a high effective emission frequency.
3. The `amount` field in the existing column_emitter binary was hardcoded
   to 1.0 and the viewer didn't read it anyway.

#### The fix (data-derived flow throttle)
- `extract_column_emitter.py` now tracks the **unclamped** captured radius
  alongside the clamped sphere radius. From these and the gravity-derived
  velocity, computes per-frame
  `amount = (pi * r_real^2 * |v|) / ((4/3) * pi * r_clamp^3)`
  in units of emits-per-second. Writes into the existing `amount` slot.
- Viewer maintains `g_columnEmitterAccum` across substeps:
  `accumulator += amount * dt`; while `accumulator >= 1`: emit, decrement.
  Capped at 4 emits per substep to avoid runaway loops on pathological values.

> **Why this is mathematically clean:** Time-averaged emitted volume equals
> `amount * V_sim_per_emit * duration = (Q_real / V_sim) * V_sim * duration =
> Q_real * duration`. The clamp on the sphere geometry cancels — it only
> affects per-emit chunkiness, not the total volume integrated over time.

(Status as of writing: viewer code edited and Python script updated. Not yet
re-extracted on the workstation. User asked for thorough audit before running.)

---

## 3. Hardcoded constants — the honest list

For the paper's transparency:

| Constant | Value | Where | Justification |
|---|---|---|---|
| Voxel size dx | 0.015 m | manifest.json | discretisation choice |
| Bottle target height | 0.15 m | manifest.json | physical scale anchor for the visible bottle portion |
| Gravity | 9.8 m/s² | extract_column_emitter.py | physical constant |
| column_z_margin | 0.005 m | extract_column_emitter.py | offset above mouth so emit isn't inside walls |
| min_fall_height | 0.005 m | extract_column_emitter.py | floor for sqrt(2gh); kicks in only if column degenerate |
| min_active_particles | 5 | extract_column_emitter.py | filter out single-particle noise frames |
| radius_clamp range | [5mm, 1.5*dx] | extract_column_emitter.py | sphere stability bounds |
| particlesPerCell | 8 | simulator default | sampling density per fluid cell — doesn't affect volume |
| accumulator cap | 4 emits/substep | viewer | safety against pathological amount values |

Everything else in the pipeline is data-derived from the source video.

---

## 4. Negative results worth their own paper figure

1. **Vanilla 4DGS bottle reconstruction.** Chunky pyramid shape, motivates the
   factored approach. Already have figures of the captured Gaussian point
   cloud.
2. **Hunyuan3D-2 on cropped bottle frame.** Melted-cup output. Useful as a
   negative-result figure — model size doesn't substitute for input fit.
3. **The coordinate-system bug.** Side-by-side rendering of "Z-up bottle in
   Y-up simulator" (laying on side) vs "Y-up bottle in Y-up simulator"
   (standing). Useful for the Limitations / Implementation Notes section.
4. **Pre-throttle vs post-throttle pour rate.** The "gallons per second"
   simulation vs the data-derived rate. Demonstrates why physical-quantity
   matching matters.

---

## 5. Open questions / coworker integration

- **Coworker's ViewCrafter-derived 360 video.** Not yet delivered. Plan: take
  his synthetic 360 of the original pour event, then either (a) run static
  3DGS on the pre-pour frames for a clean bottle, or (b) run multi-view 4DGS
  on the full sequence for both bottle and column. Watch for: diffusion-induced
  geometric inconsistency between views; whether his method outputs camera
  poses or whether we need COLMAP.
- **Will the throttle's data-derived amount actually produce a physically
  reasonable pour rate?** Sanity check: extraction will print
  `implied volume rate = X mL/s` and `estimated total pour = X mL`. Olive
  oil pour from a small bottle should be O(100-300) mL/s.
- **What if `r_real` from the deformation MLP is overestimated?** Fallbacks:
  use a different percentile (e.g., 50th); estimate Q from oil-mass-flux
  by counting yellow Gaussians entering the bottle per frame.

---

## 6. Files / artefacts

```
gaussian_splatting/
  fluid_capture/
    input/videoplayback-3                         # source video
  dynamic_capture/
    bottle_frames/                                # extracted + prepped frames
      frame_0000.png
      v5_full_bottle_padded.png                   # input to TripoSG
      mesh_A_views.png                            # rendered ortho views
    bottle_mesh.glb                               # TripoSG output (mesh A)
    bottle_from_mesh.py                           # mesh -> bottle_solid.bin
    extract_column_emitter.py                     # 4DGS -> column_emitter.bin (now with throttle)
    extract_fluid_state_v2.py                     # 4DGS -> sim_state_*.bin + bottle_solid.bin
    transpose_zup_to_yup.py                       # one-shot Z-up -> Y-up converter
    captured_states_v2/
      bottle_solid.bin                            # original 4DGS bottle (z-up)
      bottle_solid_mesh.bin                       # TripoSG mesh bottle (z-up)
      bottle_solid_mesh_yup.bin                   # TripoSG mesh bottle (y-up — currently used)
      column_emitter.bin                          # captured trajectory (z-up)
      column_emitter_yup.bin                      # captured trajectory (y-up)
      manifest.json / manifest_yup.json
      sim_state_0000.bin .. sim_state_0500.bin    # per-frame replay states (z-up)

pipe_fluid_engine/
  build_gui/PipeFluidEngine                       # built viewer binary
  viewer/main_gui.cpp                             # CLI handlers + throttle accumulator
  src/pipe_fluid_scene.cpp                        # loadBottleSolid / loadBottlePourDemo
  src/fluid_state_loader.cpp                      # binary readers
  include/pipe_fluid/fluid_state_loader.h         # CapturedColumnEmitter + CapturedBottleSolid

paper_journal.md                                  # this file
```

---

## 7. Frustrations log (verbatim, for the "Challenges Encountered" section)

These are direct quotes from project conversations. Useful for the paper to
ground the narrative in real research-process texture.

> "i got it now i get it wow, i have been focusing on such a wrong thing"

(After realising the rectangular outer cage was the simulator's domain border,
not the bottle. ~983 cells of bottle vs ~5076 cells of cage.)

> "i dont think this is presentable at all, we have nothing currently what
> are we even going to write in the paper"

(Around the time the captured-Gaussian bottle was clearly inadequate for the
ablation table. Triggered the image-to-3D pivot.)

> "look the thing is we dont even have a stable simulation i dont know what
> you are talking about but the particles are flying everywhere and its
> leaking eveyrwherre this is nothing, tell me is there a way to get a
> better model than this"

(Just before adopting the factored bottle/column reconstruction approach.)

> "ok i think this sounds like a good plan, its not like i have any other
> option, i still want to keep what i have now, to simply show how wrong
> our current method is, but yes lets do this"

(Approving the factored approach. The "wrong current method" framing became
the ablation baseline.)

> "but no hardcoding im tired of this stupid hardcoding, choosing how much
> water or where the bottle is this is all stupid we should not hardcode
> any of these things we are doing a paper after all this should be derived
> from the video"

(Triggered the data-derived flow throttle in Phase I.)

---

## 8. To-do / parking lot

- [ ] Run extraction on workstation with new throttle code; verify
  `implied volume rate` is physically reasonable (~100-300 mL/s)
- [ ] Re-transpose updated `column_emitter.bin` to `_yup`
- [ ] Rebuild viewer with throttle accumulator
- [ ] Capture a clean before/after pour video for the paper figure
- [ ] Update `bottle_from_mesh.py` to natively output Y-up (currently outputs
  Z-up + we transpose afterwards) — task #104
- [ ] Wait for coworker's ViewCrafter video; integrate as third ablation row
- [ ] Decide ablation table format (rows × columns, metric choices)
- [ ] Write paper sections: Method, Results, Limitations, Implementation Notes
- [ ] Camera-ready figures: bottle reconstruction comparison; pour-rate
  before/after; coordinate-bug illustration

---

## 9. May 2026 submission decision (post-FluidNexus discovery)

*Added May 4, 2026, after discovering [FluidNexus (CVPR 2025 Oral)](https://arxiv.org/abs/2503.04720).
Context for the decision: I never planned a May submission — that's professor
pressure. I want to push the project to 2027. The strategic question is "do
we make the May attempt as strong as possible to satisfy that pressure, or
pivot now?".*

### 9.1 The FluidNexus discovery

FluidNexus, CVPR 2025 Oral, is the paper we're partly trying to write.
Same input (single monocular video). Same key insight (use a video-diffusion
novel-view synthesizer to overcome single-view ambiguity — exactly the
ViewCrafter approach my coworker is implementing). Stronger technical stack
(differentiable simulation + integrated rendering). Their datasets are
real-world fluid videos comparable in scope to ours.

This is direct prior art on the central novelty axis. SIGGRAPH Asia
reviewers will know it.

### 9.2 What's still defensibly ours

After honest comparison:

| Aspect | FluidNexus | Ours |
|---|---|---|
| Headline contribution | end-to-end differentiable fluid reconstruction | factored static-container + dynamic-fluid pipeline |
| Container reconstruction | implicit | explicit (image-to-3D + voxelize), first-class |
| Simulation | differentiable, joint with rendering | classical FLIP/APIC, non-differentiable |
| Compute footprint | heavy (GPU training, large diffusion model) | lightweight (4DGS train + minutes-per-video re-sim on a laptop) |
| Reconstruction novel-view substitute | learned video diffusion | (planned: ViewCrafter — same family) |
| Flow rate estimation | implicit in physics loss | explicit `Q = dV_inside/dt` from per-frame voxel cell counts |

Specific contributions that survive the comparison:

- The **factored static-vs-dynamic split**: image-to-3D for the container,
  4DGS for the fluid. FluidNexus collapses both into one pipeline, which
  means transparent containers with refractive walls are a known weakness
  for them (and for us, but our factoring at least exposes it explicitly).
- The **bottle-fill flow rate estimator** (`Q = dV_inside_bottle/dt`).
  Robust to the deformation MLP's lateral smearing artefacts; doesn't
  trust column geometry. This is small but novel.
- **Negative-results journal**: monocular static-camera 4DGS fails on
  refractive containers (chunky bottle), Hunyuan3D-2 hallucinates on cropped
  partial-object inputs (melted cup), Z-up data + Y-up sim is a silent
  catastrophic bug. These are useful for practitioners and citable.
- **Particle-direct injection from a captured trajectory**: small
  implementation contribution but means our emit is faithful to the data.

### 9.3 The decision options

#### Option A — Push for May 12 with the strongest version we can build
*Realistic outcome: rejection from main Technical Papers track, but
detailed reviewer feedback for free, and "submitted" satisfies the
professor.*

8-day plan, in priority order:

| Day | Task | Why |
|---|---|---|
| 1 | Capture 2 additional pour videos (different fluid + container) | Single-video evidence is anecdotal; 3 videos shows generalisation |
| 2 | Run pipeline end-to-end on each (4DGS train overnight on workstation) | Sanity-check that pipeline isn't bottle-specific |
| 3 | Build ablation table: captured-Gaussian vs TripoSG vs (later) ViewCrafter for bottle; sphere-emit vs particle-direct for column; with vs without Q-throttle | Concrete content for Methods/Results |
| 4 | Implement quantitative metric: render sim from source camera angle, L1/SSIM vs source frames | Gives reviewers a number to grade |
| 5 | Write Methods + Results sections, fill numbers | |
| 6 | Write Introduction (pick FluidNexus differentiator angle), Related Work, Limitations | |
| 7 | Abstract, Conclusion, figure polishing | |
| 8 | Read-through, submit | |

Differentiator angle to pick (one only — don't try to claim both):
- (i) **Container-aware fluid re-simulation** — the static container is a
  first-class reconstruction problem, FluidNexus collapses it into the
  fluid representation
- (ii) **Lightweight, classical-solver pipeline** — minutes-per-video on
  a laptop, no differentiable simulation, no diffusion model

#### Option B — Pivot to a friendlier venue for the same content
*Realistic outcome: acceptance at a workshop / poster track / technical
brief, less prestige but real publication and a foot in the door for the
2027 main paper.*

Candidates:
- **SIGGRAPH Asia 2026 Posters / Technical Briefs** (less prestigious, but
  same venue prestige line on CV; lower bar)
- **SCA 2026 / 2027** (Symposium on Computer Animation; explicitly the
  fluid-sim community)
- **Eurographics 2027 short papers**
- **CVPR 2027** (a year out, gives time to add ViewCrafter integration,
  evaluate against FluidNexus directly)

If the professor cares about *attempting submission* more than about main
track, Option B is strictly better than Option A: we trade SIGGRAPH Asia
main track's near-zero acceptance odds for a venue where the paper has a
real chance.

#### Option C — Withdraw and take the next 8 days to plan the 2027 paper
*Realistic outcome: best long-term outcome but requires the professor
conversation.*

Use the time to:
- Explicitly compare against FluidNexus as the new baseline
- Decide which differentiator (container-aware vs lightweight) is the real
  contribution worth growing
- Plan the experiments needed: more videos, more containers, more fluids,
  proper user study, comparison metrics
- Coordinate with the ViewCrafter coworker on what his contribution looks
  like in the joint paper

This is the option I'd recommend purely on technical grounds. Whether it
flies depends on the professor.

### 9.4 What we'd take to 2027 regardless

Whether we push for May or skip it, the 2027 plan should include:

- **3+ pour videos** with different fluids and containers (water, oil,
  coffee; bottle, mug, glass, jar)
- **Quantitative metrics**: photometric L1/SSIM between sim render and
  source video; flow-rate timing accuracy; bottle-fill accuracy at
  endpoint
- **Direct comparison against FluidNexus** on at least one shared input
  (their public datasets are on GitHub; we can run our pipeline on theirs
  too)
- **ViewCrafter coworker integration** as a third ablation row
- **Refined narrative**: container-aware re-simulation as a complement
  to FluidNexus, not a replacement
- **Negative-results section** elevated from afterthought to first-class
  contribution (the chunky-bottle, hallucinated-cap, coordinate-bug
  case studies are genuinely useful for the community)

### 9.5 The professor conversation (worth scripting before having)

If pushing for B or C instead of A, the case to make:

> "FluidNexus (CVPR 2025 Oral) does most of what we're aiming at, with a
> stronger technical stack. Submitting to SIGGRAPH Asia main track in
> 8 days has near-zero acceptance probability and the rejection will be
> public. I'd rather (i) submit to a workshop/poster track for an actual
> acceptance and a published line on the CV, or (ii) use the next 8 days
> to scope the 2027 paper properly with FluidNexus as the explicit
> baseline. Either is better than a likely-rejected main-track attempt."

Concrete numbers to bring: the FluidNexus arXiv link, their CVPR Oral
status, our actual readiness state (what's working, what isn't, what we
have figures for vs what we don't).

### 9.6 Decision triggers

If by **end of Day 1** of the push (May 5):
- The 2 additional videos are captured AND run cleanly through 4DGS
  training → keep going on Option A.
- Either step fails (capture quality, training crash, voxelization
  artifacts) → seriously consider switching to Option B/C; we won't have
  time to recover and write the paper.

If by **end of Day 4** (May 8):
- Ablation table + metrics are done → write the paper, submit on Day 8.
- Anything substantial still broken → switch to a poster/workshop
  framing for whatever venue accepts late.

These checkpoints exist to prevent throwing good time after bad.
