# RMBench Stage 2 Flexiv embodiment audit

This document records the reproducible dual Flexiv Rizon4s + GN01 embodiment
implementation. It covers robot description, simulation assets, planners,
state/action contracts, and a fixed head camera only. Task POMDPs, dataset
conversion, training, inference, and real-robot control are Stage 3 or later.

## Fixed provenance

- Repository branch: `develop/RMBench`.
- RMBench subtree: `87e0498891073d483d330195c0f160709bd92ff5`.
- Official description: `https://github.com/flexivrobotics/flexiv_description.git`.
- Official branch: `humble-v1`.
- Official commit: `92ef7865d76585e6e08d291bdfe652d32f7740f4`.
- Official license: Apache-2.0, retained in the submodule.
- Runtime environment: `dp3-rmbench`, Python 3.10. The existing `dp3`
  environment is not an installation target.

The official submodule is read-only input to
`scripts/rmbench/flexiv/generate_embodiment.py`. The generator retains raw
official xacro output, rewrites only the package mesh URI for a self-contained
runtime bundle, and replaces the upstream zero-mass marker inertials with
small positive inertials required by SAPIEN. It never writes into the official
submodule.

## Real configuration versus simulation defaults

| Item | Source classification | Audit result |
| --- | --- | --- |
| Rizon4s arm topology, joint limits, mesh names, flange, TCP links | official description | verified at the pinned commit |
| GN01 `finger_width_joint` limits and mimic graph | official description | verified; normalized `g=0` is closed and `g=1` is open |
| Left/right seven-joint home vectors | local real runtime config | copied without serials or network data; verified as source values |
| Dual base translations, table height, right-arm 180-degree simulation yaw | simulation default | not claimed as real installation geometry |
| Fixed head-camera pose and D435 intrinsics | simulation default | not present in the real runtime config; not claimed real |
| Dynamics, timestep, drive gains, IK tolerances, action limits | simulation default | explicit simulation overrides only |

Real runtime mount angles are not silently promoted to base pose or camera
extrinsics. Missing physical measurements remain marked `verified: false` in
`sim_assets/flexiv_rizon4s_dual_gn01/`.

## Generated artifacts

The ignored runtime bundle is generated at:

`third_party/sim/RMBench/assets/embodiments/flexiv-rizon4s-dual-gn01/`

It contains the official raw dual output, raw left/right outputs, the
postprocessed dual URDF, two single-articulation URDFs, a generation manifest,
RMBench config files, and per-side CuRobo config files. The combined URDF is
kept for provenance and structural audit. RMBench's existing loader and
planner APIs are used through the two single-articulation entries:

```yaml
embodiment: [flexiv-rizon4s-dual-gn01-left, flexiv-rizon4s-dual-gn01-right, 0.90]
```

This preserves two independent arm entities and two independent planner
instances. The generated cspace order is seven arm joints followed by the
seven active GN01 joints; the retract vector has exactly the same order.

## Contracts

`diffusion_policy_3d/sim/flexiv/` is the explicit adapter boundary:

- `frames.py`: world/base transforms, quaternion convention, and matrix-column
  rotation-6D conversion.
- `state_adapter.py`: exact 34D `flexiv_abs_rot6d_v2` state.
- `action_adapter.py`: exact 14D delta action, per-arm XYZ and rotvec limits,
  and left multiplication `R_target = Exp(delta_rotvec) @ R_current`.
- `gripper_adapter.py`: URDF/manifest-derived GN01 mimic mapping with
  fail-fast range handling.
- `embodiment.py`: SAPIEN arm lookup and damped-least-squares IK.
- `envs/flexiv_embodiment_smoke.py`: no-task, no-reward, no-data smoke
  environment with atomic dual-arm action rejection when either IK solve
  fails.

State field order is:

```text
left.q[7], left.tcp_pos_base[3], left.tcp_rot6d_base[6], left.gripper[1],
right.q[7], right.tcp_pos_base[3], right.tcp_rot6d_base[6], right.gripper[1]
```

Action field order is:

```text
left.delta_xyz[3], left.delta_rotvec[3], right.delta_xyz[3],
right.delta_rotvec[3], left.gripper_cmd[1], right.gripper_cmd[1]
```

## Reproduction and acceptance

Run these commands from the repository root:

```bash
conda run -n dp3-rmbench env CONDA_DEFAULT_ENV=dp3-rmbench \
  python scripts/rmbench/flexiv/generate_embodiment.py --force

conda run -n dp3-rmbench python scripts/rmbench/flexiv/inspect_urdf.py \
  --urdf third_party/sim/RMBench/assets/embodiments/flexiv-rizon4s-dual-gn01/left.urdf \
  --manifest third_party/sim/RMBench/assets/embodiments/flexiv-rizon4s-dual-gn01/generation_manifest.json \
  --side left --out-dir outputs/rmbench_flexiv_embodiment/left_urdf

conda run -n dp3-rmbench python scripts/rmbench/flexiv/inspect_urdf.py \
  --urdf third_party/sim/RMBench/assets/embodiments/flexiv-rizon4s-dual-gn01/right.urdf \
  --manifest third_party/sim/RMBench/assets/embodiments/flexiv-rizon4s-dual-gn01/generation_manifest.json \
  --side right --out-dir outputs/rmbench_flexiv_embodiment/right_urdf

conda run -n dp3-rmbench python scripts/rmbench/flexiv/validate_embodiment.py \
  --headless --out-dir outputs/rmbench_flexiv_embodiment

conda run -n dp3-rmbench python scripts/rmbench/flexiv/planner_smoke.py \
  --out-dir outputs/rmbench_flexiv_embodiment

conda run -n dp3-rmbench python scripts/rmbench/flexiv/capture_acceptance_artifacts.py \
  --headless --capture-all --output-dir outputs/rmbench_flexiv_acceptance
```

`validate_embodiment.py` checks load, 34D state, 1000-step home stability,
repeatable reset, all six signed base-frame translation directions, zero
action, GN01 cycle, head-camera RGB/depth, and contacts. `planner_smoke.py`
structurally checks both CuRobo configs and loads both URDFs through MPlib;
CuRobo construction requires an available CUDA device and is recorded as
`SKIP` when the host has no GPU. No driver error is converted into `PASS`.

The GUI command is:

```bash
conda run -n dp3-rmbench python scripts/rmbench/flexiv/visualize_embodiment.py \
  --gui --mode home --view head-camera
```

It requires a working Vulkan/display stack. If unavailable, run the headless
commands and report the exact driver error; the camera result must remain
`SKIP`, never a fabricated visual pass.

The focused Python tests are the `tests/test_flexiv_*.py` files. Stage 0
smoke tests and the existing cover-blocks import/regression remain separate
checks. `PointCloudBuilder` remains a clean nested gitlink and is not edited.

## Known repository-level baseline conditions

`git submodule status --recursive` currently encounters an existing malformed
nested gitlink under `third_party/sim/RMBench/policy/Mem-0/LlamaFactory` with no
`.gitmodules` mapping. This Stage 2 work does not repair or rewrite that
unrelated subtree. The existing strict doctor also reports its known RMBench
DP3 shadow-import check; the scoped Stage 2 adapters import the project path
explicitly and do not alter the DP3 package.
