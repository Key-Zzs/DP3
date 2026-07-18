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

For the Flexiv real task, startup additionally requires
`state_schema=flexiv_abs_rot6d_v2`, `state_dim=34`,
`state_rotation_representation=rotation_6d`,
`rotation6d_convention=matrix_columns_0_1`,
`normalizer_schema=flexiv_abs_rot6d_v2`, `action_dim=14`, and
`action_rotation_representation=rotvec`. State orientation is absolute RDK
world/base TCP orientation encoded as the first two matrix columns
`[R[:, 0], R[:, 1]]`; it is not Home-relative. The action remains delta rotvec.
An old v1 checkpoint is rejected before robot connection.

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
- process-isolated Rerun monitor settings.

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
point_cloud       [2048, 3]
agent_pos             [34]
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

The PointCloudBuilder YAML is the only source of the live camera/depth contract.
`depth_source.mode: frame` selects native depth, keeps the adapter camera at
`use_depth=true`, and leaves IR disabled. `depth_source.mode: ffs_stereo`
selects any of the four Builder backends, enables IR, and keeps RGB, left IR,
and right IR in one RealSense frameset. The adapter publishes the canonical
keys `sidecar.head_left_ir` and `sidecar.head_right_ir` plus
`head_rgbd_timestamp`, `head_rgbd_frame_index`, and paired IR timestamp/frame
index metadata. The live frame adapter maps those fields to the Builder's
configured `left_key` and `right_key`; for `xyzrgb` it also supplies RGB. It
deliberately omits native `depth` from an FFS Builder frame, so FFS cannot
silently fall back to native depth.

Before `robot.connect()`, startup parses the Builder contract, preflights the
backend-specific artifact manifest/files, checks camera dimensions and FFS
geometry, and requires exact checkpoint compatibility: `xyz` is 3 channels,
`xyzrgb` is 6 channels, and `sampling.num_points` must match the checkpoint.
The `--check-config` branch exercises the same native/FFS adapter feature
contract without opening the camera or connecting the robot.

## Start Inference

Review these fields in `dp3_inference_config.yaml` before starting:

```yaml
checkpoint:
  path: outputs/flexiv_dual_arm_head_xyz-simple_dp3-abs-rot6d-v2_seed1000/checkpoints/latest.ckpt
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

Every control cycle performs:

1. read live Flexiv joints, end-effector poses, gripper widths, and one coherent
   camera frameset; native mode uses its depth sidecar, while FFS mode uses its
   left/right IR pair (and RGB when configured);
2. resolve the configured native or FFS depth source once with PointCloudBuilder;
3. crop in the configured camera-space workspace;
4. sample the exact 2048-point policy input;
5. build and stack the 34D absolute rotation-6D robot-state observation;
6. when the configured action queue is empty, run `policy.predict_action()` and
   enqueue the complete returned chunk;
7. otherwise retain the new observation as part of the two-frame history;
8. scale/clip and validate the next queued action;
9. call `robot.send_action()` and append one JSONL audit row per action.

Repeated RGB-D frames, padded point clouds, non-finite values, stale actions,
or watchdog violations stop the loop. These are runtime safety conditions, not
a separate no-send deployment stage.

## Rerun Telemetry Monitor

The live monitor is deliberately outside the control process:

```text
DP3 inference/control
        | non-blocking fixed-size writes
        v
multiprocessing.shared_memory latest-only rings (capacity 3)
        v
independent telemetry process
        +--> local Viewer (spawn)
        +--> manual or remote Viewer (connect_grpc)
```

Core code is in `visualizer/visualizer/monitor/`:

- `config.py` and `schema.py` validate the monitor YAML and fixed payload
  contract;
- `shared_ring.py` provides independent control, camera, sampled point-cloud,
  and optional stage rings;
- `client.py` is the producer-side frequency gate and best-effort publisher;
- `process.py` owns the spawned telemetry child;
- `rerun_sink.py` and `blueprint.py` are child-only Rerun code;
- `benchmark.py` runs synthetic transport/resource measurements.

The inference process does not import Rerun or call a Rerun logging API. RGB,
depth, point clouds, state, actions, and policy horizons use fixed shared-memory
arrays. A producer tries slot locks with `block=False`; if all slots are busy,
the record is dropped immediately. The consumer copies only the newest committed
slot into a private buffer, releases the lock, and then logs to Rerun. Stale
frames are never replayed and there is no unbounded backlog.

The checked-in configuration publishes control at `inference.rate_hz`, camera
and sampled point clouds at 2 Hz, and raw/cropped stages at 1 Hz. Raw/cropped
display is capped at 5,000 points per stage because it adds a low-frequency
Builder/D2H and rendering cost. The Builder still runs exactly once per control
cycle: normal cycles use `from_live_frame()`, while a due stage cycle uses the
same-pass `from_live_frame_with_stages()` instead.

The Viewer time panel defaults to the `log_time` timeline in `Following` state.
`activate_blueprint_on_start: true` actively applies it for every recording, so
an older paused cursor cannot remain before the first RGB-D sample.

Rerun display rates are telemetry rates, not camera capture rates. The current
profile deliberately uses `min_bulk_slack_ms: 0`: publication occurs only after
action send, and measured normal monitor transport is below 0.4 ms p99. A
positive slack threshold suppressed almost every bulk frame when the existing
10 Hz PointCloudBuilder loop took about 126 ms. If another deployment has less
watchdog margin, disable raw/cropped first or restore a positive threshold.

Time-series units are grouped in the default Blueprint:

- joints and action rotation vectors: radians;
- TCP xyz and action delta xyz: metres;
- state rotation-6D matrix-column components: unitless;
- normalized gripper state/command: unitless `[0, 1]`;
- timing: milliseconds;
- telemetry health: counters or boolean status represented as `0/1`.

Point-cloud xyz axes and decoded depth are metres. RGB values are `uint8`
intensities. For raw/cropped diagnostics, keep their display limits small
(the default is 5,000 points per stage), hide unused 3D/time-series views, and
restart the Viewer after a heavy recording to release its retained history.

Install the optional monitor dependencies in the active DP3 environment:

```bash
python -m pip install -e "visualizer[monitor]"
```

Start a local Viewer automatically from the telemetry child:

```yaml
monitor:
  enabled: true
  viewer:
    mode: spawn
    port: 9876
    memory_limit: 2GB
    detach_process: true
    activate_blueprint_on_start: true
```

The detached Viewer remains open after inference stops. The next inference run
reuses the existing listener on port 9876. Close the Viewer explicitly when its
retained history is no longer needed.

For a manually started local Viewer:

```bash
rerun --port 9876 --memory-limit 2GB
```

```yaml
monitor:
  viewer:
    mode: connect
    url: rerun+http://127.0.0.1:9876
```

The same `connect` mode supports a remote Viewer. On the Viewer host run
`rerun --bind 0.0.0.0 --port 9876 --memory-limit 2GB`, and set
`monitor.viewer.url` to `rerun+http://<viewer-ip>:9876` on the inference host.
Use a trusted LAN/VPN or firewall rule for that port. Closing, disconnecting,
or crashing the Viewer/telemetry child is fail-open and does not stop the
inference loop. `monitor.enabled: false` is a true no-op: no shared memory,
child process, or Rerun import is created.

Set `activate_blueprint_on_start: false` to preserve a hand-tuned active layout;
in that mode, select Following manually in the time panel.

Run the synthetic benchmark without hardware:

```bash
python tools/benchmark_dp3_monitor.py
```

It writes `logs/monitor_benchmark/<timestamp>/config_resolved.yaml`,
`system_info.json`, `samples.csv`, `resource_report.json`, and
`resource_report.md`. The report separates baseline, shared-memory/null-sink,
and local Viewer CPU/RSS, publish latency, cycle jitter, deadline misses,
drops/overwrites, effective rates, consumer heartbeat/lag, and Viewer
startup/shutdown. The tested optional dependency is `rerun-sdk==0.34.1` with
`psutil==7.2.2`. Headless runs
must be read as “headless viewer; not equivalent to visible native rendering”.

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
