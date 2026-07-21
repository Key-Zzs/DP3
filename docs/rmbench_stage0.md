# RMBench Stage 0

Stage 0 vendors RMBench as a Git subtree and validates a separate SAPIEN 3
runtime without changing the existing `dp3` environment or the
`PointCloudBuilder` submodule.

## Fixed inputs

- Branch: `develop/RMBench`
- RMBench upstream: `https://github.com/RoboTwin-Platform/RMBench.git`
- Subtree pin: `87e0498891073d483d330195c0f160709bd92ff5`
- Runtime environment: `dp3-rmbench`, Python 3.10
- Runtime wheels: PyTorch `2.7.1+cu128`, torchvision `0.22.1+cu128` for RTX
  5080 `sm_120`; PyTorch3D and CuRobo are rebuilt against this ABI
- CuRobo: `v0.7.8`, pinned by `scripts/rmbench/bootstrap_env.sh`
- Warp: `1.6.2`, retained for CuRobo 0.7.8's `warp.torch` API
- Hugging Face dataset: `TianxingChen/RMBench` at immutable asset revision
  `d899d72b53270a89f71d216c08ecbd4d9a7004fd` (the official `refs/pr/8` commit;
  override with `RMBENCH_HF_REVISION` when auditing another official revision)

Only `embodiments/**` and `objects/**` are downloaded. The full dataset and
policy-training assets are out of scope.

## Reproduce

Run from the repository root while no `dp3` environment is active:

```bash
scripts/rmbench/bootstrap_env.sh
scripts/rmbench/fetch_assets.sh --dry-run
scripts/rmbench/fetch_assets.sh
```

The asset command requires an authenticated Hugging Face session. It updates
the generated embodiment config paths after download.

The default revision is the immutable commit behind official `refs/pr/8`
because the current dataset `main` listing omits `franka-panda`; it contains
the required Franka embodiment alongside Aloha and `005_button`. Only the
scoped asset patterns are downloaded.

The helper intentionally checks `aloha-agilex`, `franka-panda`, and
`005_button`. If the current Hugging Face snapshot omits one of these paths,
the command exits with a failure and reports the missing path; an Aloha copy is
never substituted for a missing embodiment.

## Validation levels

```bash
PYTHONNOUSERSITE=1 conda run -n dp3-rmbench \
  python scripts/rmbench/doctor.py --strict --check-assets --check-sim

PYTHONNOUSERSITE=1 conda run -n dp3-rmbench \
  python scripts/rmbench/smoke_test.py --level 0 \
  --json-out environments/reports/rmbench_stage0_level0.json

PYTHONNOUSERSITE=1 conda run -n dp3-rmbench \
  python scripts/rmbench/smoke_test.py --level 1 \
  --json-out environments/reports/rmbench_stage0_level1.json

PYTHONNOUSERSITE=1 conda run -n dp3-rmbench \
  python scripts/rmbench/smoke_test.py --level 2 \
  --json-out environments/reports/rmbench_stage0_level2.json
```

Level 0 checks imports. Level 1 creates a minimal SAPIEN 3 scene and imports
`put_back_block`. Level 2 initializes the task and validates its observation
contract when the scoped assets are present; it reports `SKIP` when assets
have not been authenticated/downloaded.

On the validated RTX 5080 host, Level 1 and Level 2 pass. Warp 1.6.2 prints a
non-fatal `cuDeviceGetUuid` driver API warning during Level 2; keep that output
when diagnosing the host rather than treating it as a clean system log.

The existing `dp3` baseline is captured under
`environments/snapshots/dp3_before_rmbench/`. Run the baseline again with:

```bash
PYTHONNOUSERSITE=1 conda run -n dp3 \
  python scripts/rmbench/baseline_smoke_test.py \
  --json-out environments/reports/dp3_baseline_after_rmbench.json
```

## Common failures and recovery

- **Wrong branch or shadowed source:** `doctor.py --strict` fails before running
  the simulator. Check `git branch --show-current`, `git status`, and
  `PYTHONNOUSERSITE=1`; the DP3 import must resolve to this checkout.
- **Hugging Face authentication/rate limit:** authenticate with the local HF
  session and rerun `fetch_assets.sh`. A failed or incomplete snapshot is a
  reported failure; do not substitute another robot directory or download the
  full dataset.
- **Missing assets:** rerun `fetch_assets.sh`; it is scoped to
  `embodiments/**` and `objects/**` and validates Aloha, Franka, and
  `005_button` explicitly.
- **Vulkan/OIDN/driver warnings:** SAPIEN may print `svulkan2` or OIDN
  diagnostics on this host. They remain in the report. Treat a task-level
  initialization failure as a failure even if imports pass.
- **CuRobo/PyTorch3D extension mismatch:** rerun `bootstrap_env.sh` in
  `dp3-rmbench`; it rebuilds only when the expected CUDA headers, Torch ABI,
  and `sm_120` extension checks are not satisfied.

To rebuild the isolated environment, first preserve any needed report files and
then remove only the named environment with `conda env remove -n dp3-rmbench`.
Rerun `bootstrap_env.sh`, followed by the asset and validation commands above.
The ignored asset directory may be retained or refetched independently.

## Subtree updates

Subtree updates are review-only in Stage 0. Inspect a candidate upstream commit
with `scripts/rmbench/update_subtree.sh --dry-run`, review the resulting pin and
rerun the full validation suite before applying it. Do not update the subtree or
push a remote as part of this completed Stage 0 run.

## Stage 1 entry point

Stage 1 starts only after this report is accepted. Its first bounded slice is a
single-head-camera `put_back_block` collection using the validated Aloha task,
XYZ point clouds, and a small Zarr export with `n_obs_steps` 1 and 2. Policy
training, full-dataset collection, multi-camera capture, Belief DP3, and online
robot control are not Stage 0 deliverables.

## Boundaries

Do not run the upstream `_install.sh` verbatim: it assumes ambient pip state,
patches files without content guards, and downloads the entire asset tree.
Use the repository helper scripts so branch, pin, environment, and asset
scope are checked before mutation. Do not create a new branch/worktree or
push/force-update a remote as part of Stage 0.
