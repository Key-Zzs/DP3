# LeRobot to DP3 zarr Workflow

This repository is being extended around a Flexiv dual-arm RGB-D workflow for
DP3 / 3D-Diffusion-Policy. The current README focuses on task 2: offline
conversion from a local LeRobot dataset to a DP3-compatible zarr replay buffer.
The original upstream DP3 README is kept as `README_DP3.md`.

## Current Scope

- Convert a local LeRobot dataset path to DP3 zarr.
- Use the shared `PointCloudBuilder` pipeline for RGB-D to point-cloud
  generation.
- Default camera: `head`.
- Depth source: native RealSense depth from `sidecar.*_depth`.
- Alignment: no `rs.align`; generated configs set
  `camera.aligned_depth_to_color: false`.
- `xyz` mode: deprojects native depth with depth intrinsics.
- `xyzrgb` mode: projects depth-frame XYZ into the color camera with
  `depth_to_color` extrinsics, then samples RGB from color pixels.
- Output point cloud frame: selected camera/depth frame.
- No three-view fusion.
- No world-frame or robot-base transform.
- No FFS or FoundationStereo.
- No Flexiv realtime control in the offline converter.

Task 5 online inference should reuse the same `PointCloudBuilder` package and
the same YAML schema, but should not call the offline export script.

## Environment

Use the `dp3` conda environment for export, inspection, visualization, and
training smoke tests:

```bash
conda activate dp3
cd 3D-Diffusion-Policy
export PYTHONPATH=$PWD/PointCloudBuilder:$PWD/3D-Diffusion-Policy:$PYTHONPATH
```

## Default Output Path

If `--output-zarr` is omitted, the exporter writes to:

```text
~/.cache/dp3_zarr/<lerobot_repo_id>_<camera>_<pointcloud-mode>.zarr
```

The script first reads `repo_id` from `meta/info.json`. If the local dataset
does not store `repo_id`, it falls back to the path relative to
`~/.cache/huggingface/lerobot`. Path separators and unsupported
filename characters in the repo id are replaced with `_`.

For the example dataset below, the default `xyz` output is:

```text
~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr
```

Pass `--output-zarr` only when you need to override this location.

## Export xyz

```bash
python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --camera head \
  --pointcloud-mode xyz \
  --num-points 1024 \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --overwrite
```

## Export xyzrgb

```bash
python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --camera head \
  --pointcloud-mode xyzrgb \
  --num-points 1024 \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml \
  --overwrite
```

If `--builder-config` is omitted, the script writes a generated
`*.pointcloud_builder.yaml` next to the output zarr. The generated config stores
depth intrinsics, color intrinsics, depth scale, and for `xyzrgb` the
`depth_to_color` transform.

Exports are committed atomically. Frames are first written to a hidden sibling
directory, then `state`, `action`, and `point_cloud` checksums are verified. The
final `.zarr` path appears only after `export_status=complete` and matching
`expected_total_frames` / `converted_frames` metadata have been written. An
interrupted export therefore cannot be mistaken for a complete training set.

## Inspect zarr

```bash
python tools/inspect_dp3_zarr.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr
```

The inspector verifies the completion metadata and stored SHA-256 checksums,
checks `data/state`, `data/action`, `data/point_cloud`, and
`meta/episode_ends`, prints shapes and ranges, rejects NaN/Inf, checks that
`episode_ends[-1] == T`, and prints zarr attributes. Flexiv training performs
the same completion and checksum checks before loading samples.

## Visualize zarr Point Clouds

Use the Open3D zarr point-cloud viewer:

[visualize_zarr_pointcloud.py](visualizer/visualizer/visualize_zarr_pointcloud.py)

You can pass either the zarr root or the direct `data/point_cloud` array path.
Use an absolute path for zarr inputs; examples use `~` to avoid machine-specific
home directories.

Visualize one frame from the zarr root:

```bash
python visualizer/visualizer/visualize_zarr_pointcloud.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr \
  --frame 0
```

Visualize one frame from the direct point-cloud array:

```bash
python visualizer/visualizer/visualize_zarr_pointcloud.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr/data/point_cloud \
  --frame 0
```

Useful options:

```bash
--point-size 4
--background 1 1 1
--max-points 1024
--no-show
```

The visualizer automatically detects:

```text
N x 3 -> xyz point cloud, colored by z height
N x 6 -> xyzrgb point cloud, RGB normalized from [0,1] or [0,255]
```

## Debug Point-Cloud Stages

Use these tools when the final zarr point cloud looks wrong and you need to
inspect the exact preprocessing stages used by `export_lerobot_to_dp3_zarr.py`.
Both scripts rebuild one frame through the same `PointCloudBuilder` path:
raw depth deprojection, crop, then sampling. The Open3D window shows `raw`,
`cropped`, and `sampled` point clouds side by side; each pane supports mouse
rotate, pan, and zoom.

Debug a frame from an exported zarr. This reads `source_lerobot_path`,
`camera`, `pointcloud_mode`, `num_points`, and the stored
`pointcloud_builder_config` from zarr attrs, then replays the source LeRobot
RGB-D frame:

```bash
python tools/debug_zarr_pointcloud_stages.py \
  --dp3-zarr ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyzrgb.zarr \
  --frame-index 0
```

By default, `debug_zarr_pointcloud_stages.py` uses the builder config snapshot
stored inside `.zattrs`, so it reproduces the exported zarr even if the YAML file
on disk has changed. To test the currently edited config, pass it explicitly:

```bash
python tools/debug_zarr_pointcloud_stages.py \
  --dp3-zarr ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyzrgb.zarr \
  --frame-index 0 \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml
```

Debug directly from a LeRobot dataset without reading zarr attrs:

```bash
python tools/debug_lerobot_pointcloud_stages.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --frame-index 0 \
  --camera head \
  --pointcloud-mode xyzrgb \
  --num-points 1024 \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml
```

Add `--no-show` to print stage shapes and metadata without opening the Open3D
GUI.

## DP3 zarr Structure

The exported zarr has:

```text
data/state       (T, 28) float32
data/action      (T, 14) float32
data/point_cloud (T, N, 3) float32 for xyz
data/point_cloud (T, N, 6) float32 for xyzrgb
meta/episode_ends cumulative int64 episode ends
```

In the current DP3 dataset code, `data/state` is loaded as
`obs["agent_pos"]`, and `data/point_cloud` is loaded as `obs["point_cloud"]`.

The repository already provides real-task YAMLs for XYZ and XYZRGB. The unified
training config derives point-cloud color usage and encoder input channels from
the selected task's `expected_pointcloud_dim`.

## Train Flexiv Dual-Arm DP3

All Flexiv training parameters now live in
`3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_train_config.yaml`.
At minimum, review these fields before starting:

```yaml
defaults:
  - task: real/flexiv_dual_arm_head_xyz  # or ..._xyzrgb

launcher:
  gpu_id: 0
  overwrite: false

algorithm: simple_dp3  # simple_dp3 or dp3
task:
  dataset:
    zarr_path: /absolute/path/to/flexiv_head_xyz.zarr
    max_train_episodes: 90

training:
  seed: 42
  resume: false

logging:
  mode: online  # online, offline, or disabled
```

`launcher.gpu_id` selects the physical GPU through `CUDA_VISIBLE_DEVICES`; keep
`training.device: cuda:0` so the selected device is addressed correctly inside
the process. For XYZRGB, select `real/flexiv_dual_arm_head_xyzrgb` and update
the zarr path. The color flag and six encoder channels are resolved
automatically.

Activate the environment, then training is a zero-argument command:

```bash
conda activate dp3
bash scripts/train_flexiv_dual_arm_dp3.sh
```

The output directory is controlled by `run_dir` in the same YAML. Its default
resolves to:

```text
outputs/<exp_name>_seed<seed>/checkpoints/
```

If the target output directory already exists, the wrapper aborts by default to
avoid mixing old checkpoints, Hydra configs, and WandB files. Set
`launcher.overwrite: true` only when the entire existing run directory should
be deleted. To continue an interrupted run, keep `overwrite: false`, set
`training.resume: true`, and ensure `<run_dir>/checkpoints/latest.ckpt` exists.
Resume continues at the next epoch instead of repeating the configured epoch
count. The final epoch is always saved even when it is not aligned with
`checkpoint_every`. The script no longer accepts positional training arguments
or environment-based hyperparameter overrides.

For a short sanity run, temporarily set the corresponding YAML values:

```yaml
task:
  dataset:
    max_train_episodes: 1
dataloader:
  batch_size: 1
  num_workers: 0
val_dataloader:
  batch_size: 1
  num_workers: 0
training:
  num_epochs: 1
  max_train_steps: 1
  use_ema: false
logging:
  mode: disabled
checkpoint:
  save_ckpt: false
```

## Flexiv Dual-Arm DP3 Inference

Inference parameters live in
`3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_inference_config.yaml`.
Edit that file to select the checkpoint, robot config, GPU, optional duration limit,
control rate, action scheduling, independent Flexiv startup/servo controls,
inference-only scheduler and diffusion steps, safety limits, pre-connection
policy warmup, point-cloud config, and Open3D visualization. The default runtime
executes each configured action chunk at 30 Hz and enables the 200 Hz Flexiv
Cartesian servo thread.

The current epsilon checkpoint is trained with DDPM but deployed with a DDIM
scheduler reconstructed from the checkpoint beta schedule. DDIM uses 10 steps
at roughly 39--40 ms for batch-1 inference; do not replace it with 10-step DDPM.
Both arms and both grippers use the model outputs without task-specific
stationary-arm or fixed-gripper overrides.

The model architecture, horizon, observation history, scheduler-training
semantics, point-cloud shape, and state/action shape are duplicated in the train
and inference YAMLs. The launcher compares those fields with the checkpoint
payload before connecting. Inference `n_action_steps` may differ within the
official DP3 slice bound, and `use_ema` selects an available checkpoint weight
set. Other inference-specific fields may differ as documented below.

The live runtime is standalone in this repository. It uses the local Flexiv
adapter and RealSense RGB-D implementation under
`third_party/real/dual_flexiv_rizon4s/interface`; it does not require an
external Le-nero checkout or the LeRobot Python package. This is separate from
the offline LeRobot dataset compatibility documented above.

Install the minimal robot-side dependencies without changing the DP3
Torch/CUDA stack:

```bash
python -m pip install -r third_party/real/dual_flexiv_rizon4s/requirements-runtime.txt
```

Create a private, gitignored station configuration and replace all hardware
placeholders:

```bash
cp third_party/real/dual_flexiv_rizon4s/configs/flexiv_runtime.example.yaml \
  third_party/real/dual_flexiv_rizon4s/configs/flexiv_runtime.local.yaml
```

Set `FLEXIV_DP3_ROBOT_CONFIG=/absolute/path/to/config.yaml` to use another
private config path. Never commit real robot or camera serial numbers.

Run the complete policy deployment with one command:

```bash
conda run -n dp3 bash scripts/run_flexiv_dual_arm_dp3_inference.sh
```

When the `dp3` environment is already active:

```bash
bash scripts/run_flexiv_dual_arm_dp3_inference.sh
```

This is the motion-producing `inference` path. It directly runs live RGB-D
deprojection, crop, 1024-point sampling, policy prediction, action filtering,
and `robot.send_action()`; there is no separate no-send stage or mode handoff. The default
Open3D monitor runs in a separate process at 2 Hz with capacity-one latest-frame
queues, so visualization cannot block the control loop.

The default configuration runs closed-loop inference until `Ctrl+C`. The launcher
also prints the generated JSONL path and stop-file command, so another terminal can stop it with:

```bash
touch /tmp/stop_flexiv_dp3_inference
```

An optional no-hardware configuration check is available but is not part of the
normal deployment sequence:

```bash
conda run -n dp3 bash scripts/run_flexiv_dual_arm_dp3_inference.sh --check-config
```

The check requires the configured checkpoint and local YAMLs, but exits before
`robot.connect()` and does not open the RealSense pipeline. Normal inference can
move the robot. The software migration and automated tests do not replace the
operator-run RealSense-only, Flexiv connection, and final closed-loop tests.
Codex did not run any hardware connection, camera pipeline, or live inference
command while implementing this migration.

See [docs/flexiv_dual_arm_inference.md](docs/flexiv_dual_arm_inference.md) for
the parameter classification, runtime flow, and hardware stop behavior.
