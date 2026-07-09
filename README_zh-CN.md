# LeRobot 到 DP3 zarr 工作流

本仓库正在围绕 Flexiv 双臂 RGB-D 数据流扩展 DP3 /
3D-Diffusion-Policy。当前 README 重点覆盖任务 2：将本地 LeRobot 数据集离线转换为
DP3 可读取的 zarr replay buffer。原始上游 DP3 README 保留为 `README_DP3.md`。

## 当前范围

- 将本地 LeRobot 数据集路径转换为 DP3 zarr。
- 使用统一的 `PointCloudBuilder` 管线完成 RGB-D 到点云的生成。
- 默认相机：`head`。
- 深度来源：`sidecar.*_depth` 中的 RealSense 原生深度。
- 对齐方式：不做 `rs.align`；自动生成的配置使用
  `camera.aligned_depth_to_color: false`。
- `xyz` 模式：用 depth intrinsics 反投影原生深度。
- `xyzrgb` 模式：用 `depth_to_color` 外参把 depth-frame XYZ 投影到 color
  相机，再从 color 像素取 RGB。
- 输出点云坐标系：所选相机的 depth/camera frame。
- 不做三视角融合。
- 不做 world-frame 或 robot-base 坐标变换。
- 不接入 FFS 或 FoundationStereo。
- 离线转换脚本不接入 Flexiv 实时控制。

任务 5 的在线推理应复用同一个 `PointCloudBuilder` 包和同一套 YAML schema，但不应调用离线导出脚本。

## 环境

导出、检查、可视化和训练 smoke test 建议使用 `dp3` conda 环境：

```bash
conda activate dp3
cd /home/deepcybo/workspace/3D-Diffusion-Policy
export PYTHONPATH=$PWD/PointCloudBuilder:$PWD/3D-Diffusion-Policy:$PYTHONPATH
```

## 默认输出路径

如果不传 `--output-zarr`，导出脚本会默认写入：

```text
/home/deepcybo/.cache/dp3_zarr/<lerobot_repo_id>_<camera>_<pointcloud-mode>.zarr
```

脚本会优先读取 `meta/info.json` 里的 `repo_id`。如果本地数据集没有保存
`repo_id`，则回退为相对 `/home/deepcybo/.cache/huggingface/lerobot` 的路径。
repo id 里的路径分隔符和不适合作为文件名的字符会被替换成 `_`。

以下示例数据集的默认 `xyz` 输出路径是：

```text
/home/deepcybo/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr
```

只有需要改写输出位置时，才传入 `--output-zarr`。

## 导出 xyz 点云

```bash
python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path /home/deepcybo/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --camera head \
  --pointcloud-mode xyz \
  --num-points 1024 \
  --builder-config /home/deepcybo/workspace/3D-Diffusion-Policy/third_party/real/flexiv-GN01/configs/data_config.yaml \
  --overwrite
```

## 导出 xyzrgb 点云

```bash
python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path /home/deepcybo/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --camera head \
  --pointcloud-mode xyzrgb \
  --num-points 1024 \
  --builder-config /home/deepcybo/workspace/3D-Diffusion-Policy/third_party/real/flexiv-GN01/configs/data_rgb_config.yaml \
  --overwrite
```

如果不提供 `--builder-config`，脚本会在输出 zarr 旁边写入自动生成的
`*.pointcloud_builder.yaml`。该配置保存 depth intrinsics、color intrinsics、
depth scale；在 `xyzrgb` 模式下还会保存 `depth_to_color` 变换。

## 检查 zarr

```bash
python tools/inspect_dp3_zarr.py \
  --zarr-path /home/deepcybo/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr
```

检查脚本会验证 `data/state`、`data/action`、`data/point_cloud` 和
`meta/episode_ends`，打印 shape 和数值范围，拒绝 NaN/Inf，检查
`episode_ends[-1] == T`，并打印 zarr attrs。

## 可视化 zarr 点云

使用 Open3D zarr 点云可视化脚本：

[visualize_zarr_pointcloud.py](/home/deepcybo/workspace/3D-Diffusion-Policy/visualizer/visualizer/visualize_zarr_pointcloud.py)

可以传入 zarr 根目录，也可以直接传入 `data/point_cloud` 数组路径。路径必须是绝对路径。

从 zarr 根目录可视化一帧：

```bash
python visualizer/visualizer/visualize_zarr_pointcloud.py \
  --zarr-path /home/deepcybo/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr \
  --frame 0
```

从 point-cloud 数组路径可视化一帧：

```bash
python visualizer/visualizer/visualize_zarr_pointcloud.py \
  --zarr-path /home/deepcybo/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr/data/point_cloud \
  --frame 0
```

常用选项：

```bash
--point-size 4
--background 1 1 1
--max-points 1024
--no-show
```

可视化脚本会自动判断：

```text
N x 3 -> xyz 点云，按 z 高度着色
N x 6 -> xyzrgb 点云，RGB 自动兼容 [0,1] 或 [0,255]
```

## 调试点云处理阶段

当最终 zarr 点云看起来不对，需要检查 `export_lerobot_to_dp3_zarr.py`
实际使用的预处理阶段时，使用下面两个脚本。它们都会让指定帧走同一条
`PointCloudBuilder` 路径：原始 depth 反投影、裁剪、采样。Open3D 大窗口会并排显示
`raw`、`cropped`、`sampled` 三个点云视图，每个视图都支持鼠标旋转、平移和缩放。

从已导出的 zarr 调试一帧。脚本会从 zarr attrs 读取
`source_lerobot_path`、`camera`、`pointcloud_mode`、`num_points` 和保存的
`pointcloud_builder_config`，再回放原始 LeRobot RGB-D 帧：

```bash
python tools/debug_zarr_pointcloud_stages.py \
  --dp3-zarr /home/deepcybo/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyzrgb.zarr \
  --frame-index 0
```

默认情况下，`debug_zarr_pointcloud_stages.py` 会使用 `.zattrs` 中保存的 builder
config 快照，因此即使磁盘上的 YAML 已经修改，也能复现导出当时的 zarr。若要测试当前正在编辑的配置，显式传入：

```bash
python tools/debug_zarr_pointcloud_stages.py \
  --dp3-zarr /home/deepcybo/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyzrgb.zarr \
  --frame-index 0 \
  --builder-config third_party/real/flexiv-GN01/configs/data_rgb_config.yaml
```

不读取 zarr attrs，直接从 LeRobot 数据集调试：

```bash
python tools/debug_lerobot_pointcloud_stages.py \
  --lerobot-path /home/deepcybo/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --frame-index 0 \
  --camera head \
  --pointcloud-mode xyzrgb \
  --num-points 1024 \
  --builder-config third_party/real/flexiv-GN01/configs/data_rgb_config.yaml
```

加上 `--no-show` 可以只打印三阶段 shape 和 metadata，不打开 Open3D GUI。

## DP3 zarr 结构

导出的 zarr 结构如下：

```text
data/state       (T, 28) float32
data/action      (T, 14) float32
data/point_cloud (T, N, 3) float32，对应 xyz
data/point_cloud (T, N, 6) float32，对应 xyzrgb
meta/episode_ends 累积 episode 结束位置，int64
```

在当前 DP3 dataset 代码中，`data/state` 会被加载为
`obs["agent_pos"]`，`data/point_cloud` 会被加载为 `obs["point_cloud"]`。

使用 `simple_dp3` 训练时，需要新增一个 task YAML，并让其中的 `shape_meta`
与导出的 zarr 一致。对于当前 Flexiv 双臂数据集，`agent_pos` 应为 `[28]`，
action 应为 `[14]`，point cloud 应根据导出模式设置为 `[1024, 3]` 或
`[1024, 6]`。

如果使用 `xyzrgb` 训练，还需要让策略使用点云颜色，并匹配点云 encoder 的输入通道数：

```yaml
policy:
  use_pc_color: true
  pointcloud_encoder_cfg:
    in_channels: 6
```
