# Task 2: LeRobot to DP3 zarr Export

This directory contains the offline converter for turning a local LeRobot RGB-D
dataset into a DP3-compatible zarr replay buffer.

This is not the task 5 online inference path. Online inference should reuse the
same `PointCloudBuilder` package and the same YAML schema, but should not call
this offline export script.

## Scope

- Default camera: `head`
- Depth source: native RealSense depth from `sidecar.*_depth`
- Alignment: no `rs.align`; `camera.aligned_depth_to_color: false`
- `xyz` mode: deprojects native depth with depth intrinsics
- `xyzrgb` mode: projects depth-frame XYZ into the color camera using
  `depth_to_color` extrinsics, then samples RGB in color pixels
- Output point cloud frame: head/depth camera frame
- No three-view fusion
- No world-frame or robot-base transform
- No FFS or FoundationStereo
- No Flexiv realtime control

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

## Inspect

```bash
python tools/inspect_dp3_zarr.py \
  --zarr-path data/flexiv_pick_place_head_xyz.zarr
```

The inspector checks `data/state`, `data/action`, `data/point_cloud`, and
`meta/episode_ends`, prints shapes and ranges, rejects NaN/Inf, checks that
`episode_ends[-1] == T`, and prints zarr attributes.

## DP3 Shape Contract

The exported zarr has:

```text
data/state       (T, 28) float32
data/action      (T, 14) float32
data/point_cloud (T, N, 3) float32 for xyz
data/point_cloud (T, N, 6) float32 for xyzrgb
meta/episode_ends cumulative int64 episode ends
```

For `simple_dp3` training, add a task YAML whose `shape_meta` matches the
exported zarr. For the current Flexiv dual-arm dataset, `agent_pos` should be
`[28]`, action should be `[14]`, and point cloud should be `[1024, 3]` or
`[1024, 6]` depending on the export mode.
