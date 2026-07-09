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

## Inspect zarr

```bash
python tools/inspect_dp3_zarr.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr
```

The inspector checks `data/state`, `data/action`, `data/point_cloud`, and
`meta/episode_ends`, prints shapes and ranges, rejects NaN/Inf, checks that
`episode_ends[-1] == T`, and prints zarr attributes.

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

For `simple_dp3` training, add a task YAML whose `shape_meta` matches the
exported zarr. For the current Flexiv dual-arm dataset, `agent_pos` should be
`[28]`, action should be `[14]`, and point cloud should be `[1024, 3]` or
`[1024, 6]` depending on the export mode.

For `xyzrgb` training, also set the policy to use point-cloud color and make
the point-cloud encoder input channels match:

```yaml
policy:
  use_pc_color: true
  pointcloud_encoder_cfg:
    in_channels: 6
```

## Train Flexiv Dual-Arm DP3

The wrapper trains from an exported DP3 zarr and lets you choose the physical
GPU explicitly as the fourth positional argument. The script sets
`CUDA_VISIBLE_DEVICES=<gpu_id>`, so training uses `cuda:0` inside the selected
visible device.

XYZ point cloud:

```bash
conda run -n dp3 bash scripts/train_flexiv_dual_arm_dp3.sh \
  xyz \
  /path/to/flexiv_head_xyz.zarr \
  simple_dp3 \
  0 \
  42
```

XYZRGB point cloud:

```bash
conda run -n dp3 bash scripts/train_flexiv_dual_arm_dp3.sh \
  xyzrgb \
  /path/to/flexiv_head_xyzrgb.zarr \
  simple_dp3 \
  0 \
  42
```

Arguments are:

```text
<xyz|xyzrgb> <zarr_path> [simple_dp3|dp3] [gpu_id] [seed] [hydra_overrides...] [--overwrite]
```

By default, checkpoints are written to this repository-relative path:

```text
outputs/<exp_name>_seed<seed>/checkpoints/
```

Set `RUN_DIR=/custom/output/dir` to override the whole Hydra output directory.
If the target output directory already exists, the wrapper aborts by default to
avoid mixing old checkpoints, Hydra configs, and WandB files with a new run. Add
`--overwrite` only when you want to delete the entire target output directory
before training starts.

Useful environment overrides include `SAVE_CKPT=True|False`,
`WANDB_MODE=offline|online|disabled`, `BATCH_SIZE`, `NUM_WORKERS`,
`MAX_TRAIN_EPISODES`, and `EXP_NAME`.

Short sanity run:

```bash
DEBUG=False SAVE_CKPT=False WANDB_MODE=disabled MAX_TRAIN_EPISODES=1 \
BATCH_SIZE=1 NUM_WORKERS=0 \
conda run -n dp3 bash scripts/train_flexiv_dual_arm_dp3.sh \
  xyz \
  /path/to/flexiv_head_xyz.zarr \
  simple_dp3 \
  0 \
  42 \
  training.num_epochs=1 \
  training.max_train_steps=1 \
  training.use_ema=False \
  training.sample_every=999999 \
  policy.num_inference_steps=1
```
