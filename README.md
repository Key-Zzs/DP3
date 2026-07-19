# LeRobot to DP3 zarr Workflow

This repository is being extended around a Flexiv dual-arm RGB-D workflow for
DP3 / 3D-Diffusion-Policy. The current README focuses on task 2: offline
conversion from a local LeRobot dataset to a DP3-compatible zarr replay buffer.
The original upstream DP3 README is kept as `README_DP3.md`.

## Current Scope

- Convert a local LeRobot dataset path to DP3 zarr.
- Use the shared `PointCloudBuilder` pipeline for RGB-D to point-cloud
  generation.
- Camera, point-cloud format, sampling, and depth source are read exclusively
  from the required `--builder-config` YAML. `depth_source.mode: frame` means
  native depth; `depth_source.mode: ffs_stereo` selects the configured FFS route.
- Alignment: no `rs.align`; the Builder YAML sets
  `camera.aligned_depth_to_color: false`.
- `xyz` mode: deprojects native depth with depth intrinsics.
- `xyzrgb` mode: projects depth-frame XYZ into the color camera with
  `depth_to_color` extrinsics, then samples RGB from color pixels.
- Output point cloud frame: selected camera/depth frame.
- No three-view fusion.
- No world-frame or robot-base transform.
- FFS depth is metric depth produced by PointCloudBuilder after its validated
  calibration/rectification contract; the exporter does not duplicate FFS
  inference or point-cloud geometry.
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

## Raw Sidecar Zarr vs DP3 Zarr

There are two different Zarr schemas in this workflow:

1. A new LeRobot recording can preserve raw sensor data in the acquisition
   sidecar declared by `meta/rgbd_sidecar.json` and stored at
   `sidecars/realsense.zarr`. It contains native depth, lossless left/right IR,
   per-camera timestamp/reused values, scalar join keys, robot timestamps, and
   episode ends for all three cameras.
2. This offline exporter creates a derived DP3 replay buffer containing
   `data/state`, `data/action`, `data/point_cloud`, `meta/episode_ends`, and
   optional `data/img`. It is not a raw-sensor archive.

The exporter and source debug tool accept:

```text
--rgbd-sidecar-source auto|zarr|parquet
```

`auto` is the default. If `meta/rgbd_sidecar.json` exists, `auto` must use it
and validate the complete Zarr v2 store. An incomplete/corrupt status,
unsupported schema/version, calibration hash mismatch, missing array, wrong
dtype/shape/chunk/compressor, count mismatch, malformed episode ends, or scalar
join mismatch fails before any point cloud is generated. It never silently
falls back to Parquet when a manifest exists. Only a dataset with no manifest
is detected as the legacy Parquet layout. Explicit `zarr` or `parquet` rejects
a conflicting layout.

Validation compares `index`, `episode_index`, `frame_index`,
`global_frame_index`, `robot_timestamp`, the selected camera's
`rgbd_timestamp`, and `rgbd_reused` in bounded batches. Zarr is opened once and
native depth is read in frame chunks; a full multi-episode sidecar is not
loaded into memory. Raw IR remains in the LeRobot sidecar and is not copied to
the DP3 replay buffer.

The reader exposes exact left/right IR pairs and calibration references. In
`ffs_stereo` mode the exporter requests those pairs, validates their shape,
dtype/range, timestamp, global-frame join, and calibration SHA, then passes
`left_ir`, `right_ir`, `timestamp`, `global_frame_index`, and optional RGB to
the same `PointCloudBuilder.from_recorded_frame()` call used by native depth.
The native `depth` field is never passed to the FFS builder.

The dedicated builder-side backend and artifact guide is
[PointCloudBuilder FFS guide](PointCloudBuilder/ffs_reproduction/README.md).

## Default Output Path

If `--output-zarr` is omitted, the exporter writes to a path whose camera and
point-cloud format components come from the Builder YAML:

```text
~/.cache/dp3_zarr/<lerobot_repo_id>_<camera>_<pointcloud-mode>_state_abs_rot6d_v2.zarr
```

The script first reads `repo_id` from `meta/info.json`. If the local dataset
does not store `repo_id`, it falls back to the path relative to
`~/.cache/huggingface/lerobot`. Path separators and unsupported
filename characters in the repo id are replaced with `_`.

For the example dataset below, the default `xyz` output is:

```text
~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz_state_abs_rot6d_v2.zarr
```

Pass `--output-zarr` only when you need to override this location.

## Export xyz

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --rgbd-sidecar-source auto \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --overwrite
```

## Export xyzrgb

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --rgbd-sidecar-source auto \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml \
  --overwrite
```

## Explicit Builder-config exports

Complete native-depth export for the v05 dataset (the explicit legacy-state
converter is required because this recording stores the recognized 28D source
schema):

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --rgbd-sidecar-source zarr \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --allow-legacy-state-conversion \
  --output-zarr ~/.cache/dp3_zarr/flexiv_dual_arm_3d_pick_place_20260713_v05_head_xyz_native.zarr
```

One-frame FFS exports use the same dataset, `head` camera, first global frame,
and 1024 fixed points. All four routes use the same canonical Builder config;
before each command, comment out the active native `depth_source` block in that
file and uncomment only the matching FFS block:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --rgbd-sidecar-source zarr --max-frames 1 \
  --allow-legacy-state-conversion \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --output-zarr outputs/ffs_export_acceptance/ffs_pytorch.zarr

PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --rgbd-sidecar-source zarr --max-frames 1 \
  --allow-legacy-state-conversion \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --output-zarr outputs/ffs_export_acceptance/ffs_tensorrt_single.zarr

PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --rgbd-sidecar-source zarr --max-frames 1 \
  --allow-legacy-state-conversion \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --output-zarr outputs/ffs_export_acceptance/ffs_tensorrt_two_stage.zarr

PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --rgbd-sidecar-source zarr --max-frames 1 \
  --allow-legacy-state-conversion \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --output-zarr outputs/ffs_export_acceptance/ffs_tensorrt_plugin.zarr
```

FFS outputs record `depth_source=ffs_stereo`, backend/artifact contract,
normalization, disparity settings, calibration and manifest SHA-256 values,
portable artifact filenames/relative paths, the resolved Builder config and its
hash, and PointCloudBuilder timing/count metadata. `native_depth_used_for_builder`
is `false` for every FFS export. Any missing IR pair, invalid calibration,
artifact/config/manifest hash mismatch, or backend initialization/inference
failure aborts and removes the temporary output; FFS never silently falls back
to native depth.

Both modes use the same downstream chain:
`depth -> PointCloudBuilder -> deprojection -> RGB mapping -> crop -> fixed-size sampling`.

`--builder-config` is required. It is the single source of truth for camera,
point-cloud format, sampling, and depth source; the exporter does not generate
or accept CLI overrides for those values. The two canonical Flexiv Builder
configs are:

- `third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml` — native/FFS + xyz
- `third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml` — native/FFS + xyzrgb

The checked-in files currently select the validated FFS routes: `data_config.yaml`
uses `tensorrt_two_stage` with `xyz`, while `data_rgb_config.yaml` uses
`tensorrt_plugin` with `xyzrgb`. Their native-depth `depth_source.mode: frame`
blocks remain commented. To select native depth, comment out the active FFS
mapping and uncomment the native block. All four FFS backend groups remain in
the canonical files so a route switch changes only the Builder YAML.

The exporter resolves relative artifact paths against the original YAML
directory before writing its output-side resolved config. Backend, artifact id,
precision, optimization level, and workspace are taken from the YAML.

Use `--rgbd-sidecar-source zarr` when a command must require the new raw
sidecar, or `--rgbd-sidecar-source parquet` when it must require a legacy
dataset with no raw-sidecar manifest. Existing commands that omit the option
remain compatible because `auto` is the default.

Exports are committed atomically. Frames are first written to a hidden sibling
directory, then `state`, `action`, and `point_cloud` checksums are verified. The
final `.zarr` path appears only after `export_status=complete` and matching
`expected_total_frames` / `converted_frames` metadata have been written. An
interrupted export therefore cannot be mistaken for a complete training set.

To reuse a recognized legacy Flexiv recording, require the explicit converter:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/legacy_lerobot \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --allow-legacy-state-conversion \
  --target-state-schema flexiv_abs_rot6d_v2 \
  --output-zarr ~/.cache/dp3_zarr/legacy_state_abs_rot6d_v2.zarr
```

The exporter accepts legacy data only when its exact 28D absolute-rotvec names
and action names match the supported Flexiv v1 contract. Unknown 28D data and
metadata conflicts fail fast; the old Zarr and old checkpoint are never
silently reused.

## Inspect zarr

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/inspect_dp3_zarr.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz_state_abs_rot6d_v2.zarr
```

The inspector verifies the completion metadata and stored SHA-256 checksums,
checks `data/state`, `data/action`, `data/point_cloud`, and
`meta/episode_ends`, prints shapes and ranges, rejects NaN/Inf, checks that
`episode_ends[-1] == T`, validates recorded native-depth or FFS source provenance when present, and
prints zarr attributes. Flexiv training performs the same completion and
checksum checks before loading samples.

## Visualize zarr Point Clouds

Use the Open3D zarr point-cloud viewer:

[visualize_zarr_pointcloud.py](visualizer/visualizer/visualize_zarr_pointcloud.py)

You can pass either the zarr root or the direct `data/point_cloud` array path.
Use an absolute path for zarr inputs; examples use `~` to avoid machine-specific
home directories.

Visualize one frame from the zarr root:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python visualizer/visualizer/visualize_zarr_pointcloud.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz_state_abs_rot6d_v2.zarr \
  --frame 0
```

Visualize one frame from the direct point-cloud array:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python visualizer/visualizer/visualize_zarr_pointcloud.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz_state_abs_rot6d_v2.zarr/data/point_cloud \
  --frame 0
```

Useful options:

```bash
--point-size 4
--background 1 1 1
--max-points 2048
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
native or FFS stereo depth, deprojection, crop, then sampling. The Open3D window shows `raw`,
`cropped`, and `sampled` point clouds side by side; each pane supports mouse
rotate, pan, and zoom.

Debug a frame from an exported zarr. This reads `source_lerobot_path`,
`camera`, `pointcloud_mode`, `num_points`, and the stored
`pointcloud_builder_config` from zarr attrs, then replays the source LeRobot
RGB-D frame:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/debug_zarr_pointcloud_stages.py \
  --dp3-zarr ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyzrgb_state_abs_rot6d_v2.zarr \
  --frame-index 0
```

By default, `debug_zarr_pointcloud_stages.py` uses the builder config snapshot
stored inside `.zattrs`, so it reproduces the exported zarr even if the YAML file
on disk has changed. To test the currently edited config, pass it explicitly:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/debug_zarr_pointcloud_stages.py \
  --dp3-zarr ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyzrgb_state_abs_rot6d_v2.zarr \
  --frame-index 0 \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml
```

Debug directly from a LeRobot dataset without reading zarr attrs:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/debug_lerobot_pointcloud_stages.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --frame-index 0 \
  --rgbd-sidecar-source auto \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml
```

Both stage-debug tools accept the same Builder YAML contract as the exporter.
For an FFS frame, use an explicit FFS Builder YAML; the debugger
requests `left_ir`/`right_ir`, validates the recorded IR/calibration join, and
passes only those IR fields to the shared builder:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/debug_lerobot_pointcloud_stages.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --frame-index 0 \
  --rgbd-sidecar-source zarr \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --no-show
```

When the input is an FFS-derived zarr, `debug_zarr_pointcloud_stages.py`
uses the Builder config snapshot in `.zattrs`; it therefore reproduces the FFS
route without a native-depth fallback. Pass `--builder-config` only when you
intentionally want to inspect a different YAML.

Add `--no-show` to print stage shapes and metadata without opening the Open3D
GUI.

## DP3 zarr Structure

The exported zarr has:

```text
data/state       (T, 34) float32
data/action      (T, 14) float32
data/point_cloud (T, N, 3) float32 for xyz
data/point_cloud (T, N, 6) float32 for xyzrgb
meta/episode_ends cumulative int64 episode ends
```

Optional `data/img` keeps its existing RGB semantics. Raw `depth`, `left_ir`,
and `right_ir` arrays are not copied into this derived DP3 Zarr. Source storage,
manifest/calibration hashes and paths, committed counts, selected camera,
native-depth units/scale, and the PointCloudBuilder config/source are recorded
in root attributes.

In the current DP3 dataset code, `data/state` is loaded as
`obs["agent_pos"]`, and `data/point_cloud` is loaded as `obs["point_cloud"]`.

The Flexiv real-task state contract is `flexiv_abs_rot6d_v2`: 34 values in the
recorded order of seven joints, absolute TCP `xyz`, six absolute rotation-6D
values, and the normalized gripper state for each arm. Rotation-6D is
`[R[:, 0], R[:, 1]]`, i.e. the first two columns of the absolute RDK world/base
TCP rotation matrix; it is not the first two rows and does not depend on Home,
Quest, or camera reference frames. The action contract remains exactly 14
values: left/right delta `xyz`, left/right delta rotvec, then the two gripper
commands.

The exporter validates exact LeRobot state/action names and persisted schema
metadata. A recognized legacy Flexiv v1 28D absolute-rotvec dataset can be
converted offline with the explicit `--allow-legacy-state-conversion` flag;
the converter uses `Rotation.from_rotvec(...).as_matrix()` and the same two
matrix columns, so rotvec sign jumps near pi do not propagate. Unknown 28D
data is rejected, and the output name includes `state_abs_rot6d_v2` so it
cannot be confused with the old Zarr. Existing v1 checkpoints are incompatible
with this runtime and require retraining.

The acquisition-side LeRobot source may also use
`flexiv_abs_rot6d_raw_force_v3` with `observation.state` shape `(48,)`. DP3
still consumes only the target `flexiv_abs_rot6d_v2` `(34,)` state and the
unchanged `(14,)` action. The shared source contract validates the exact schema,
shape, dtype, finite values, and complete ordered names before rows are read;
it builds the v3-to-v2 projection by matching each target name, never by
assuming the first 34 positions. The following 14 source fields are dropped:

```text
left_ee_ext_wrench_in_tcp_raw.fx/fy/fz/mx/my/mz
left_gripper_force
right_ee_ext_wrench_in_tcp_raw.fx/fy/fz/mx/my/mz
right_gripper_force
```

The derived Zarr records `source_state_schema`, `source_state_dim`, the full
`source_state_names`, `state_transform=drop_raw_force_fields_v3_to_v2_by_name`,
and `dropped_state_names`. `raw_source_state_sha256` covers all 48 source
values, while `derived_state_sha256` covers the projected 34 values; different
hashes are expected for v3. Force/wrench values never enter DP3 Zarr
`data/state`, normalizer statistics, model inputs, checkpoints, training, or
online inference.

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
    zarr_path: ~/.cache/dp3_zarr/flexiv_head_xyz_state_abs_rot6d_v2.zarr
    max_train_episodes: 90

training:
  seed: 42
  resume: false

logging:
  mode: online  # online, offline, or disabled
```

The Flexiv dataset uses the `flexiv_abs_rot6d_v2` normalization contract. It
replays the collection adapter's `0.02 m` translation and `0.04 rad` rotation
norm limits in memory (the source zarr is not modified), applies symmetric
physical action scales to both arms, maps both grippers through `[0,1]`, and
uses robust state quantiles with range floors only for joints and absolute
`xyz`. The twelve dimensionless rotation-6D components always use fixed
`scale=1, offset=0`; they are never stretched by a low-variance quantile or a
radian floor. Training prints a `[FlexivNormalizer]` audit line; do not deploy
a checkpoint that lacks the v2 schema, fixed rotation-6D scales, and contract
metadata. Changing any of these normalizer settings requires training a new
checkpoint rather than resuming an older run.

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
policy warmup, point-cloud config, and the process-isolated Rerun monitor. The
checked-in YAML uses the actual 10 Hz inference rate and leaves the Cartesian
servo thread disabled unless explicitly enabled in the robot section.

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

For synchronous `action_mode: chunk`, `inference.temporal_ensemble_coeff`
controls deployment-aligned overlap blending. `0.0` preserves the original
queue exactly. Values in `(0, 1]` are the new-chunk weight; only the 12
Cartesian pose channels are blended and both grippers always use the new
chunk. The overlap is derived from `horizon`, `n_obs_steps`, and the configured
`n_action_steps`, so it is not tied to a four-action chunk. The checked-in
configuration uses the offline-tested `0.5` setting.

The live runtime is standalone in this repository. It uses the local Flexiv
adapter and RealSense RGB-D implementation under
`third_party/real/dual_flexiv_rizon4s/interface`; it does not require an
external Le-nero checkout or the LeRobot Python package. This is separate from
the offline LeRobot dataset compatibility documented above.

Install the minimal robot-side dependencies without changing the DP3
Torch/CUDA stack:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python -m pip install -r third_party/real/dual_flexiv_rizon4s/requirements-runtime.txt
```

Create a private, gitignored station configuration and replace all hardware
placeholders:

```bash
cp third_party/real/dual_flexiv_rizon4s/configs/flexiv_runtime.example.yaml \
  third_party/real/dual_flexiv_rizon4s/configs/flexiv_runtime.local.yaml
```

Set `FLEXIV_DP3_ROBOT_CONFIG=~/.config/flexiv_dp3/config.yaml` to use another
private config path. Never commit real robot or camera serial numbers.

Run the independent perception-only check before enabling robot motion:

```bash
conda run -n dp3 bash scripts/run_flexiv_dp3_perception_only.sh
```

When the `dp3` environment is already active:

```bash
bash scripts/run_flexiv_dp3_perception_only.sh
```

This program opens only the `head_rgb` RealSense and `PointCloudBuilder`; it
does not import Flexiv RDK, connect either arm, or send actions. By default it
discards 60 warmup frames, measures 300 frames, displays the raw/cropped/sampled
perception stages, and writes per-frame JSONL plus a summary JSON under `logs/`.
It exits with code 2 when the recent valid-depth median is below `0.75`, its
range exceeds `0.08`, sampling pads a cloud, or a depth array does not own its
memory. Add `--no-visualize` on a headless host.

The perception-only entry point also supports the explicit FFS route. Set
`depth_source.mode: ffs_stereo` in a complete FFS Builder YAML (including
live-camera intrinsics and artifact paths), then pass that YAML with
`--builder-config`; startup enables both IR streams and performs the
manifest/artifact preflight before opening the camera. For example:

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/run_flexiv_dp3_perception_only.py \
  --builder-config PointCloudBuilder/ffs_reproduction/configs/v05_ffs.yaml \
  --frames 30 \
  --no-visualize
```

Run the complete policy deployment with one command:

```bash
conda run -n dp3 bash scripts/run_flexiv_dual_arm_dp3_inference.sh
```

When the `dp3` environment is already active:

```bash
bash scripts/run_flexiv_dual_arm_dp3_inference.sh
```

This is the motion-producing `inference` path. It directly runs live RGB-D
deprojection, crop, 2048-point sampling, policy prediction, action filtering,
and `robot.send_action()`; it is separate from the no-motion perception-only
entry point above. The default Rerun telemetry child runs separately from the
control loop. It receives fixed-shape latest-only samples through bounded
shared-memory rings, so a slow or unavailable viewer cannot block policy
prediction or action send.

The formal launcher takes the live perception contract exclusively from the
selected PointCloudBuilder YAML. In `native_depth` mode it keeps
`use_depth=true` and leaves IR disabled. In `ffs_stereo` mode it enables both
IR streams and reads one coherent RGB/depth/left-IR/right-IR frameset; the
adapter publishes `sidecar.head_left_ir`, `sidecar.head_right_ir`,
`head_rgbd_timestamp`, `head_rgbd_frame_index`, and paired IR timestamp/frame
index fields. The launcher remaps those canonical observation keys to the
Builder's configured `left_key`/`right_key`, passes RGB for `xyzrgb`, and never
passes native depth to the FFS Builder or falls back to it. Startup checks the
backend artifacts, camera geometry, frame metadata contract, checkpoint
point-cloud dimension (`xyz=3`, `xyzrgb=6`), and fixed point count before any
robot connection.

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

### Rerun monitor and synthetic benchmark

The inference producer writes fixed-size RGB/depth, sampled point-cloud, state,
policy-horizon, raw-action, filtered-action, commanded-action, and timing data
to bounded `multiprocessing.shared_memory` rings. An independent non-daemon
telemetry process owns Rerun initialization, Blueprint construction, Viewer
management, serialization, and gRPC. Queue/Pipe/Event are control-only; large
payloads do not pass through them. The consumer copies only the newest slot, so
a slow or closed Viewer cannot stop robot inference.

Core implementation:
`visualizer/visualizer/monitor/{config,schema,shared_ring,client,process,rerun_sink,blueprint,benchmark}.py`.
The old `tools/flexiv_dp3_live_viewer.py` path is only a deprecation shim.

Install the optional dependencies:

```bash
python -m pip install -e "visualizer[monitor]"
```

Automatic local Viewer (default, port 9876):

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

The detached Viewer remains open after inference exits. A later inference run
detects the listener on port 9876 and reconnects to the same window instead of
spawning a duplicate. Close the Viewer window or its terminal explicitly when
it is no longer needed.

Manual local Viewer:

```bash
rerun --port 9876 --memory-limit 2GB
```

```yaml
monitor:
  viewer:
    mode: connect
    url: rerun+http://127.0.0.1:9876/proxy
```

Remote Viewer:

```bash
rerun --bind 0.0.0.0 --port 9876 --memory-limit 2GB
```

```yaml
monitor:
  viewer:
    mode: connect
    url: rerun+http://<viewer-ip>:9876/proxy
```

Use a trusted LAN/VPN or firewall restriction for a remote Viewer. Set
`monitor.enabled: false` for a true no-op. The checked-in deployment profile
publishes RGB/depth and the sampled cloud at 2 Hz and raw/cropped clouds at
1 Hz. Raw/cropped display buffers are capped at 5,000 points each; this cap does
not change the 2,048-point policy input. The sampled policy input is reused from
the existing CPU NumPy result and the policy horizon is reused from the existing
CPU NumPy action sequence. The Builder runs at most once per cycle.
Viewer/telemetry failure is fail-open.

The default Blueprint follows the newest `log_time` sample and separates state
and actions by physical unit: joints `[rad]`, TCP/action xyz `[m]`, rotation-6D
`[unitless]`, action rotvec `[rad]`, grippers `[0..1]`, and timing `[ms]`.
Actions are logged in three stages: policy-selected
(`/control/action_selected_raw`), safety-filtered
(`/control/action_filtered`), and actually commanded
(`/robot/action_commanded`). Keep `activate_blueprint_on_start: true` to apply
the `log_time / Following` default at the start of every recording.

The checked-in profile uses `min_bulk_slack_ms: 0` because publication happens
after `robot.send_action()` and the measured normal monitor hot path is below
0.4 ms p99. This prevents an already-overrunning 10 Hz inference loop from
silently suppressing every RGB/depth/point-cloud update. The 1 Hz stage path can
cost several milliseconds and is therefore rate-limited and display-truncated.
If watchdog margin is insufficient on another machine, disable raw/cropped
first or restore a positive slack threshold.

`activate_blueprint_on_start: true` guarantees Following and makes all five
image/point-cloud views active, but it reapplies the generated layout for each
recording. To retain a hand-tuned layout, save an `.rbl` file, set
`activate_blueprint_on_start: false`, and select Following manually. A manually
started Viewer can load the saved Blueprint on every run:

```bash
conda run -n dp3 rerun /absolute/path/dp3_layout.rbl \
  --port 9876 --memory-limit 2GB --persist-state
```

Set `monitor.viewer.mode: connect` and keep the same
`monitor.recording.application_id` as the saved Blueprint, then run the normal
inference script. With `activate_blueprint_on_start: false`, reconnecting does
not replace the already-active Viewer layout.

Run a synthetic benchmark (no Flexiv, RealSense, or action send):

```bash
python tools/benchmark_dp3_monitor.py
```

Reports are under `logs/monitor_benchmark/<timestamp>/` and include resolved
configuration, system/GPU information, 5 Hz process samples, JSON/Markdown
summary, p50/p95/p99/max publish latency and cycle jitter, deadline misses,
drops/overwrites, effective rates, consumer heartbeat/lag, and
producer/telemetry/Viewer CPU/RSS. The validated DP3 optional dependency is
`rerun-sdk==0.34.1` with `psutil==7.2.2`.
When a headless Viewer is used, the report is explicitly marked “headless
viewer; not equivalent to visible native rendering”.

See [docs/flexiv_dual_arm_inference.md](docs/flexiv_dual_arm_inference.md) for
the parameter classification, runtime flow, and hardware stop behavior.
