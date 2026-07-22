# Dual Flexiv Rizon4s + GN01 embodiment

This is the Stage 2 operational entry point. It starts from the pinned
official `flexiv_description` xacro, generates the ignored RMBench runtime
bundle, and validates a no-task SAPIEN embodiment with a single fixed head
camera.

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

For visual inspection, use `visualize_embodiment.py --gui --mode home
--view head-camera`. Use the same command with `--headless` for a display-free
run. A missing Vulkan or CUDA device is recorded as `SKIP`.

## Files and boundaries

- Source YAML: `sim_assets/flexiv_rizon4s_dual_gn01/`.
- Generator and audits: `scripts/rmbench/flexiv/`.
- Runtime adapters: `3D-Diffusion-Policy/diffusion_policy_3d/sim/flexiv/`.
- No-task smoke env: `third_party/sim/RMBench/envs/flexiv_embodiment_smoke.py`.
- RMBench registration: `third_party/sim/RMBench/task_config/`.
- Tests: `tests/test_flexiv_*.py`.

The bundle is ignored and reproducible. Do not hand-edit its URDF/config
files. Do not modify the official submodule or the `PointCloudBuilder`
gitlink. Stage 3 begins separately with task/data/policy work.
