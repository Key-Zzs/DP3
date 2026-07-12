# Flexiv Dual-Arm DP3 Inference

This runtime is self-contained in this repository: edit the deployment YAML
once, then start policy control with one command. It does not import an external
robot framework or inject an external source tree into `sys.path`. The runtime
mode is named `inference`; there is no staged no-send-to-motion handoff.

## Configuration Files

Two YAML files define the phase boundary:

- Training: `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_train_config.yaml`
- Inference: `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_inference_config.yaml`

The live Flexiv adapter and RealSense RGB-D implementation are under
`third_party/real/dual_flexiv_rizon4s/interface`. Offline conversion from an
existing LeRobot dataset remains supported, but the live runtime does not
require the LeRobot Python package or the external Le-nero checkout.

## Standalone Runtime Setup

Install only the robot-side packages that are not already provided by the DP3
environment:

```bash
python -m pip install -r third_party/real/dual_flexiv_rizon4s/requirements-runtime.txt
```

The list is deliberately limited to NumPy, SciPy, headless OpenCV,
`pyrealsense2`, `flexivrdk`, and `spdlog`; it does not install a separate robot
framework or replace the DP3 Torch/CUDA stack.

Create the private station configuration before running a config check or live
inference:

```bash
cp third_party/real/dual_flexiv_rizon4s/configs/flexiv_runtime.example.yaml \
  third_party/real/dual_flexiv_rizon4s/configs/flexiv_runtime.local.yaml
```

Fill the robot, tool, and head-camera placeholders in the local file. The local
file is gitignored so hardware serial numbers are not committed. To use another
path, set `FLEXIV_DP3_ROBOT_CONFIG=/absolute/path/to/flexiv_runtime.yaml`.

Both support the official `simple_dp3` and `dp3` policy classes through the
`algorithm` field. The default deployment uses `simple_dp3` because the current
checkpoint was trained with `SimpleDP3`.

### Must Match Training and Inference

The following fields describe the learned model contract and are present in
both YAML files:

- policy class: `SimpleDP3` or `DP3`;
- `horizon` and `n_obs_steps`;
- conditioning mode and U-Net structure;
- point-cloud encoder structure and XYZ/XYZRGB channel count;
- diffusion scheduler type, training timesteps, beta schedule, and prediction
  type;
- point-cloud, robot-state, and action shapes;

Before hardware connection, the launcher compares these fields with the Hydra
configuration stored in the checkpoint. A mismatch stops startup with a list of
the differing fields. Editing the current training YAML does not alter an
already-created checkpoint.

`n_action_steps` is not part of the learned-weight contract. DP3 trains the
complete `horizon` trajectory and only slices the returned rollout during
`predict_action()`. Inference may choose any positive value satisfying
`n_action_steps <= horizon - n_obs_steps + 1`. `use_ema` is also an inference
selection: raw weights may always be selected, while EMA requires EMA weights
to exist in the checkpoint. The legacy `policy.use_point_crop` field is consumed
by official simulation environment wrappers, not by the policy model; this real
pipeline obtains cropping from the PointCloudBuilder config. `policy.crop_shape`
and `policy.pointcloud_encoder_cfg.normal_channel` are also accepted but unused
by the current point-cloud encoder, so they are not hard-matched either.
Scheduler `clip_sample` is likewise an inference sampling option and is applied
to the selected checkpoint/DDIM scheduler rather than hard-matched to training.

### Training-Only

The training YAML additionally owns:

- task and zarr dataset selection;
- optimizer and learning-rate scheduler;
- dataloader and validation loader;
- epochs, seed, debug, resume, and gradient accumulation;
- EMA update schedule;
- WandB, Hydra output, and checkpoint-save behavior.

`num_inference_steps` is intentionally absent from the training YAML because it
does not affect the DP3 training loss.

### Inference-Only

The inference YAML owns:

- checkpoint and Flexiv robot-config paths;
- independent Flexiv enable/fault/Home/tool/gripper/mode/servo controls;
- physical GPU selection and policy/PointCloudBuilder devices;
- optional duration limit and policy-loop frequency;
- `receding` versus `chunk` action scheduling;
- inference-only scheduler selection (`checkpoint` or `ddim`);
- reverse-diffusion `num_inference_steps`;
- zero-input `policy_warmup_steps` performed before robot connection;
- action scaling, Cartesian/rotation limits, and runtime watchdogs;
- PointCloudBuilder runtime YAML;
- audit-log directory and stop file;
- Open3D visualization settings.

The checkpoint keeps its DDPM/epsilon training contract. The current deployment
constructs a DDIM scheduler from that stored beta schedule and uses 10 inference
steps; this is an inference-only sampling choice and does not alter checkpoint
tensors. Ten-step DDPM is not interchangeable and leaves large residual noise.
Changing architecture, horizons, channels, or scheduler training semantics
requires a matching checkpoint and normally requires retraining.

`policy_warmup_steps` defaults to 2. These discarded zero-input forwards run
before `robot.connect()` and never send actions. They initialize CUDA kernels so
the first live cycle is measured at steady-state latency instead of cold-start
latency; timing watchdog limits remain unchanged.

## Default Runtime Contract

The checked-in inference YAML currently selects:

```text
policy                 SimpleDP3
horizon                 8
n_obs_steps             2
n_action_steps          4
inference_scheduler  DDIM
num_inference_steps    10
point_cloud       [1024, 3]
agent_pos             [28]
action                [14]
action_mode         chunk
rate_hz                30
duration_seconds     null (until stopped)
```

Both arms and both grippers use the model outputs. The runtime does not apply
task-specific stationary-arm or fixed-gripper overrides.

In the default `chunk` mode the policy returns four actions. The runtime sends
all four in order at 30 Hz, then uses the latest live observation and predicts the
next chunk. This matches Le-nero's DiffusionPolicy queue and the official
`MultiStepWrapper` execution semantics. `receding` remains available for
diagnostics but resamples after every first action.

## Start Inference

Review these fields in `dp3_inference_config.yaml` before starting:

```yaml
checkpoint:
  path: outputs/flexiv_dual_arm_head_xyz-simple_dp3_seed42/checkpoints/latest.ckpt
robot:
  config: ${oc.env:FLEXIV_DP3_ROBOT_CONFIG,third_party/real/dual_flexiv_rizon4s/configs/flexiv_runtime.local.yaml}
  enable_on_connect: true
  clear_fault_on_connect: true
  go_home_on_connect: true
  switch_tool_on_connect: true
  initialize_gripper_on_connect: true
  switch_cartesian_mode_on_connect: true
  use_cartesian_servo_thread: true
inference:
  gpu_id: 0
  duration_seconds: null  # Ctrl+C/stop file; use a positive number for a timed test
  rate_hz: 30
  action_mode: chunk
```

From the repository root, run:

```bash
conda run -n dp3 bash scripts/run_flexiv_dual_arm_dp3_inference.sh
```

When the `dp3` environment is already active:

```bash
bash scripts/run_flexiv_dual_arm_dp3_inference.sh
```

The launcher clears only the configured stale stop file, applies
`CUDA_VISIBLE_DEVICES` from the YAML, validates checkpoint consistency, loads
PointCloudBuilder and the Flexiv adapter, connects, switches both arms to the
configured Cartesian mode, and enters the live inference loop. It does not run
a separate no-send stage and does not require an acknowledgement environment variable.
With `duration_seconds: null`, the loop continues until `Ctrl-C` or the stop file
is created. A positive value enables a timed upper bound for short tests.

## Live Inference Flow

Every 30 Hz control cycle performs:

1. read live Flexiv joints, end-effector poses, gripper widths, RGB, and depth;
2. deproject the RGB-D frame once with PointCloudBuilder;
3. crop in the configured camera-space workspace;
4. sample the exact 1024-point policy input;
5. build and stack the 28D robot-state observation;
6. when the configured action queue is empty, run `policy.predict_action()` and
   enqueue the complete returned chunk;
7. otherwise retain the new observation as part of the two-frame history;
8. scale/clip and validate the next queued action;
9. call `robot.send_action()` and append one JSONL audit row per action.

Repeated RGB-D frames, padded point clouds, non-finite values, stale actions,
or watchdog violations stop the loop. These are runtime safety conditions, not
a separate no-send deployment stage.

## Open3D Monitor

Visualization is enabled in the inference YAML by default. A separate process
shows a 2x2 monitor containing:

1. colorized depth;
2. raw deprojected point cloud;
3. cropped point cloud;
4. the complete sampled policy input.

PointCloudBuilder runs once in the control process. Raw and cropped clouds are
decimated only for display. Capacity-one non-blocking queues keep the newest
frame and discard stale visualization frames. The default display rate is 2 Hz,
and closing the window does not stop robot inference.

## Stop and Logs

The default run stops normally when `Ctrl-C` is pressed in its terminal. The
launcher also prints a stop command that can be run from a second terminal:

```bash
touch /tmp/stop_flexiv_dp3_inference
```

The loop checks the file before inference and again immediately before sending.
`Ctrl-C` also enters robot cleanup. Hardware stop controls remain the primary
emergency stop mechanism.

Each run creates a fresh file such as:

```text
logs/flexiv_dp3_inference_xyz_<timestamp>_until_stopped.jsonl
```

Rows include raw and filtered actions, point-cloud counts, device selection,
camera identity and age, policy latency, send duration, loop timing, artifact
hashes, and `send_status`.

## Optional Configuration Check

This check is useful after changing a YAML or checkpoint, but is not part of the
normal one-command deployment sequence:

```bash
conda run -n dp3 bash scripts/run_flexiv_dual_arm_dp3_inference.sh --check-config
```

It loads the checkpoint, PointCloudBuilder, and Flexiv adapter configuration,
then exits before `robot.connect()`.

The configuration check still requires the referenced checkpoint, point-cloud
YAML, and private robot YAML to exist. It constructs camera objects but does not
open a RealSense pipeline, discover a named camera, or contact a robot.

## Validation Boundary

Code-only validation can cover imports, configuration, coherent RGB-D/IR frame
conversion with fake framesets, feature schemas, cleanup, and fake RDK action
dispatch. RealSense-only connection, Flexiv connection, and closed-loop policy
inference remain operator-run hardware tests. Normal inference can move the
robot; it must not be treated as a software-only smoke test. Codex did not run
any hardware connection, camera pipeline, or live inference command during the
standalone-runtime migration.
