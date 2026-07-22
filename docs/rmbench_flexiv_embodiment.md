# Dual Flexiv Rizon4s + GN01 embodiment

This is the Stage 2 operational entry point. It starts from the pinned
official `flexiv_description` xacro, generates the ignored RMBench runtime
bundle, and validates a no-task SAPIEN embodiment with a single fixed head
camera.

The simulation installation geometry follows the current station layout
specification: the table long edge is the world-y direction, a raised rack is
outside the negative-x long edge and its top is 0.20 m above the table, the
two base centers are 0.30 m apart, and the left/right base roll angles are
`-45/+45` degrees. The real home joint vectors remain unchanged.

## Build

```bash
conda activate dp3-rmbench
bash scripts/rmbench/flexiv/bootstrap_description.sh --force
```

The helper refuses the wrong branch or environment. Docker is optional; when
it is unavailable the helper uses the local xacro route and records that fact
in `generation_manifest.json`.

## Run the bounded checks

```bash
conda run -n dp3-rmbench python scripts/rmbench/flexiv/validate_embodiment.py \
  --headless --out-dir outputs/rmbench_flexiv_embodiment
conda run -n dp3-rmbench python scripts/rmbench/flexiv/planner_smoke.py \
  --out-dir outputs/rmbench_flexiv_embodiment
conda run -n dp3-rmbench python scripts/rmbench/flexiv/capture_acceptance_artifacts.py \
  --headless --capture-all --output-dir outputs/rmbench_flexiv_acceptance
```

For visual inspection, use the interactive SAPIEN viewport:

```bash
conda run -n dp3-rmbench python scripts/rmbench/flexiv/visualize_embodiment.py \
  --gui --mode home --view head-camera
```

The GUI remains open until the SAPIEN window is closed or `Ctrl-C` is pressed.
Left-drag rotates the view, right-drag pans it, and the mouse wheel zooms;
keyboard camera movement, including W/A/S/D, is disabled. Add `--show-panels`
to enable the debug panels. Their layout is initialized from a fresh
per-process SAPIEN layout, so the user-global `~/.sapien/imgui.ini` cannot
stack panels over the viewport. To run a bounded GUI smoke, add `--seconds 20`.

The reproducible display-free check is:

```bash
conda run -n dp3-rmbench python scripts/rmbench/flexiv/visualize_embodiment.py \
  --headless --view front
```

Acceptance requires non-flat RGB content and valid depth, not only matching
array shapes. A missing Vulkan or CUDA device is recorded as `SKIP`.

## Files and boundaries

- Source YAML: `sim_assets/flexiv_rizon4s_dual_gn01/`.
- Installation fixture: `sim_assets/flexiv_rizon4s_dual_gn01/rack.urdf`; it is a
  fixed-root center upright with two outward 45-degree braces and sloped
  mounting plates matching the left/right base rolls.
- Generator and audits: `scripts/rmbench/flexiv/`.
- Runtime adapters: `3D-Diffusion-Policy/diffusion_policy_3d/sim/flexiv/`.
- No-task smoke env: `third_party/sim/RMBench/envs/flexiv_embodiment_smoke.py`.
- RMBench registration: `third_party/sim/RMBench/task_config/`.
- Tests: `tests/test_flexiv_*.py`.

The bundle is ignored and reproducible. Do not hand-edit its URDF/config
files. Do not modify the official submodule or the `PointCloudBuilder`
gitlink. Stage 3 begins separately with task/data/policy work.
