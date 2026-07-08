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
cd /home/deepcybo/workspace/3D-Diffusion-Policy
export PYTHONPATH=$PWD/PointCloudBuilder:$PWD/3D-Diffusion-Policy:$PYTHONPATH
```

## Export xyz

```bash
python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path /home/deepcybo/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --output-zarr data/flexiv_pick_place_head_xyz.zarr \
  --camera head \
  --pointcloud-mode xyz \
  --num-points 1024 \
  --overwrite
```

## Export xyzrgb

```bash
python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path /home/deepcybo/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --output-zarr data/flexiv_pick_place_head_xyzrgb.zarr \
  --camera head \
  --pointcloud-mode xyzrgb \
  --num-points 1024 \
  --overwrite
```

If `--builder-config` is omitted, the script writes a generated
`*.pointcloud_builder.yaml` next to the output zarr. The generated config stores
depth intrinsics, color intrinsics, depth scale, and for `xyzrgb` the
`depth_to_color` transform.

## Inspect zarr

```bash
python tools/inspect_dp3_zarr.py \
  --zarr-path data/flexiv_pick_place_head_xyz.zarr
```

The inspector checks `data/state`, `data/action`, `data/point_cloud`, and
`meta/episode_ends`, prints shapes and ranges, rejects NaN/Inf, checks that
`episode_ends[-1] == T`, and prints zarr attributes.

## Visualize zarr Point Clouds

Use the Open3D zarr point-cloud viewer:

[visualize_zarr_pointcloud.py](/home/deepcybo/workspace/3D-Diffusion-Policy/visualizer/visualizer/visualize_zarr_pointcloud.py)

You can pass either the zarr root or the direct `data/point_cloud` array path.
The path must be absolute.

Visualize one frame from the zarr root:

```bash
python visualizer/visualizer/visualize_zarr_pointcloud.py \
  --zarr-path /home/deepcybo/workspace/3D-Diffusion-Policy/data/flexiv_pick_place_head_xyz.zarr \
  --frame 0
```

Visualize one frame from the direct point-cloud array:

```bash
python visualizer/visualizer/visualize_zarr_pointcloud.py \
  --zarr-path /home/deepcybo/workspace/3D-Diffusion-Policy/data/flexiv_pick_place_head_xyz.zarr/data/point_cloud \
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
