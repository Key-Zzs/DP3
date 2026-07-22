# Stage 2 embodiment decisions

## 1. Use the official Flexiv description

The robot model is generated from the official `flexiv_description` repository,
branch `humble-v1`, commit
`92ef7865d76585e6e08d291bdfe652d32f7740f4`. The source is a Git submodule and
the commit, branch, URL, and Apache-2.0 license are recorded in
`third_party/vendor/flexiv_description.md` and the generated manifest.

## 2. Keep both combined and single-arm artifacts

The official dual xacro output is retained as `runtime_dual.urdf` for the
requested complete dual model and audit provenance. The active RMBench route
uses `left.urdf` and `right.urdf` because the existing loader accepts a
three-item embodiment specification and builds separate left/right
articulations. Reusing one articulation would collapse the two planner and
control boundaries.

## 3. Separate verified facts from defaults

The home joint vectors are copied from the local real runtime configuration.
The base translations, table, right-arm mirror yaw, camera extrinsic, camera
intrinsics, dynamics, and IK tolerances are simulation defaults. They are
committed in YAML with `source` and `verified` fields. A later measurement
must update those inputs and regenerate the bundle; it must not edit generated
URDFs by hand.

## 4. Use one fixed head camera

Stage 2 has one fixed world-mounted D435-style head camera. Wrist cameras,
camera randomization, task cues, and dataset capture semantics are disabled.
The camera pose is explicitly a visualization/simulation default until a real
extrinsic is measured.

## 5. Preserve the existing DP3 boundary

The embodiment adapters implement the canonical 34D
`flexiv_abs_rot6d_v2` state and 14D delta action only at the simulation
boundary. They do not change PointCloudBuilder, DP3 training, inference,
dataset schemas, or any source-schema migration. Rotation deltas are applied
in the arm base frame by left multiplication.

## 6. Acceptance policy

Headless validation is the reproducible gate. GUI and GPU-only CuRobo checks
are environment-dependent and must be reported as `SKIP` with their exact
cause when the display, Vulkan driver, or CUDA device is unavailable. A
`SKIP` is not a pass claim and does not authorize moving to Stage 3.
