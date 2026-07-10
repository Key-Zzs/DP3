# Flexiv Dual-Arm DP3 Inference

This runtime follows the same operator model as Le-nero `run_policy`: edit the
deployment YAML once, then start policy control with one command. The runtime
mode is named `inference`; there is no staged no-send-to-motion handoff.

## Configuration Files

Two YAML files define the phase boundary:

- Training: `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_train_config.yaml`
- Inference: `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_inference_config.yaml`

Both support the official `simple_dp3` and `dp3` policy classes through the
`algorithm` field. The default deployment uses `simple_dp3` because the current
checkpoint was trained with `SimpleDP3`.

### Must Match Training and Inference

The following fields describe the learned model contract and are present in
both YAML files:

- policy class: `SimpleDP3` or `DP3`;
- `horizon`, `n_obs_steps`, and `n_action_steps`;
- conditioning mode and U-Net structure;
- point-cloud encoder structure and XYZ/XYZRGB channel count;
- DDIM scheduler type, training timesteps, beta schedule, clipping, and
  prediction type;
- point-cloud, robot-state, and action shapes;
- EMA-versus-raw checkpoint weight selection.

Before hardware connection, the launcher compares these fields with the Hydra
configuration stored in the checkpoint. A mismatch stops startup with a list of
the differing fields. Editing the current training YAML does not alter an
already-created checkpoint.

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
- physical GPU selection and policy/PointCloudBuilder devices;
- optional duration limit and policy-loop frequency;
- `receding` versus `chunk` action scheduling;
- reverse-diffusion `num_inference_steps`;
- action scaling, Cartesian/rotation limits, and runtime watchdogs;
- PointCloudBuilder runtime YAML;
- audit-log directory and stop file;
- Open3D visualization settings.

Changing `num_inference_steps` trades policy latency against diffusion sampling
quality without changing checkpoint tensor shapes. Changing architecture,
horizons, channels, or scheduler training semantics requires a matching
checkpoint and normally requires retraining.

## Default Runtime Contract

The checked-in inference YAML currently selects:

```text
policy                 SimpleDP3
horizon                4
n_obs_steps             2
n_action_steps          3
num_inference_steps    10
point_cloud       [1024, 3]
agent_pos             [28]
action                [14]
action_mode      receding
rate_hz                 5
duration_seconds     null (until stopped)
```

In `receding` mode the policy still returns three actions, but the runtime sends
only index 0 and predicts again from the next live observation. In `chunk` mode
all three queued actions are sent before the next policy prediction.

## Start Inference

Review these fields in `dp3_inference_config.yaml` before starting:

```yaml
checkpoint:
  path: outputs/flexiv_dual_arm_head_xyz-simple_dp3_seed42/checkpoints/latest.ckpt
robot:
  config: ~/flexiv_ws/dual_arm_teleop/scripts/config/robots/flexiv_config.yaml
inference:
  gpu_id: 0
  duration_seconds: null  # Ctrl+C/stop file; use a positive number for a timed test
  rate_hz: 5
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

Each receding inference cycle performs:

1. read live Flexiv joints, end-effector poses, gripper widths, RGB, and depth;
2. deproject the RGB-D frame once with PointCloudBuilder;
3. crop in the configured camera-space workspace;
4. sample the exact 1024-point policy input;
5. build and stack the 28D robot-state observation;
6. run `policy.predict_action()` with the configured reverse-diffusion steps;
7. scale and clip the selected 14D action;
8. verify frame age, point-cloud validity, and timing watchdogs;
9. call `robot.send_action()` and append one JSONL audit row.

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
