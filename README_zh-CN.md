# LeRobot 到 DP3 zarr 工作流

本仓库正在围绕 Flexiv 双臂 RGB-D 数据流扩展 DP3 /
3D-Diffusion-Policy。当前 README 重点覆盖任务 2：将本地 LeRobot 数据集离线转换为
DP3 可读取的 zarr replay buffer。原始上游 DP3 README 保留为 `README_DP3.md`。

## 当前范围

- 将本地 LeRobot 数据集路径转换为 DP3 zarr。
- 使用统一的 `PointCloudBuilder` 管线完成 RGB-D 到点云的生成。
- 相机、点云格式、采样参数和深度来源全部只从必填的
  `--builder-config` YAML 读取。`depth_source.mode: frame` 表示 native depth；
  `depth_source.mode: ffs_stereo` 表示选择 YAML 中配置的 FFS 路线。
- 对齐方式：不做 `rs.align`；Builder YAML 使用
  `camera.aligned_depth_to_color: false`。
- `xyz` 模式：用 depth intrinsics 反投影原生深度。
- `xyzrgb` 模式：用 `depth_to_color` 外参把 depth-frame XYZ 投影到 color
  相机，再从 color 像素取 RGB。
- 输出点云坐标系：所选相机的 depth/camera frame。
- 不做三视角融合。
- 不做 world-frame 或 robot-base 坐标变换。
- FFS 深度由 PointCloudBuilder 在经过校验的 calibration/rectification contract
  后生成；exporter 不复制 FFS 推理或点云几何逻辑。
- 离线转换脚本不接入 Flexiv 实时控制。

任务 5 的在线推理应复用同一个 `PointCloudBuilder` 包和同一套 YAML schema，但不应调用离线导出脚本。

## 环境

导出、检查、可视化和训练 smoke test 建议使用 `dp3` conda 环境：

```bash
conda activate dp3
cd 3D-Diffusion-Policy
export PYTHONPATH=$PWD/PointCloudBuilder:$PWD/3D-Diffusion-Policy:$PYTHONPATH
```

## Raw sidecar Zarr 与 DP3 Zarr

本流程中有两种用途和 schema 都不同的 Zarr：

1. 新的 LeRobot 录制数据可以用 `meta/rgbd_sidecar.json` 声明 raw acquisition
   sidecar，并把三台相机的原生 depth、无损左右 IR、每相机 timestamp/reused、
   标量 join key、robot timestamp 和 episode ends 保存到
   `sidecars/realsense.zarr`。
2. 本离线导出器生成派生的 DP3 replay buffer，只包含 `data/state`、
   `data/action`、`data/point_cloud`、`meta/episode_ends` 和可选
   `data/img`。它不是原始传感器归档。

导出器和 source debug 工具支持：

```text
--rgbd-sidecar-source auto|zarr|parquet
```

默认值是 `auto`。只要 `meta/rgbd_sidecar.json` 存在，`auto` 就必须使用它并
完整验证 Zarr v2 store。status 非 complete、schema/version 不支持、calibration
hash 不一致、数组缺失、dtype/shape/chunk/compressor 错误、计数不一致、
episode ends 非法或标量 join 错位，都会在生成任何点云前失败。manifest 存在时
绝不静默回退到 Parquet；只有完全没有 manifest 的数据集才会自动识别为旧
Parquet 布局。显式选择 `zarr` 或 `parquet` 时，若与实际布局冲突也会直接报错。

校验会分批比较 `index`、`episode_index`、`frame_index`、
`global_frame_index`、`robot_timestamp`、所选相机的 `rgbd_timestamp` 和
`rgbd_reused`。Zarr 只打开一次，原生 depth 按 frame chunk 读取，不会把完整的
多 episode sidecar 一次性载入内存。原始 IR 保留在 LeRobot sidecar 中，不会复制
进 DP3 replay buffer。

reader 可以无损返回同一帧的左右 IR 和 calibration reference。在 `ffs_stereo`
模式下，exporter 请求并校验 IR 的 shape、dtype/range、timestamp、global-frame
join 和 calibration SHA，然后把 `left_ir`、`right_ir`、`timestamp`、
`global_frame_index` 以及可选 RGB 传给和 native depth 相同的
`PointCloudBuilder.from_recorded_frame()`。原生 `depth` 不会传入 FFS builder。

Builder 侧 backend 和 artifact 指南见
[PointCloudBuilder FFS 指南](PointCloudBuilder/ffs_reproduction/README_zh-CN.md)。

## 默认输出路径

如果不传 `--output-zarr`，导出脚本会根据 Builder YAML 中的相机和点云格式默认写入：

```text
~/.cache/dp3_zarr/<lerobot_repo_id>_<camera>_<pointcloud-mode>_state_abs_rot6d_v2.zarr
```

脚本会优先读取 `meta/info.json` 里的 `repo_id`。如果本地数据集没有保存
`repo_id`，则回退为相对 `~/.cache/huggingface/lerobot` 的路径。
repo id 里的路径分隔符和不适合作为文件名的字符会被替换成 `_`。

以下示例数据集的默认 `xyz` 输出路径是：

```text
~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz_state_abs_rot6d_v2.zarr
```

只有需要改写输出位置时，才传入 `--output-zarr`。

## 导出 xyz 点云

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --rgbd-sidecar-source auto \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --overwrite
```

## 导出 xyzrgb 点云

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --rgbd-sidecar-source auto \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml \
  --overwrite
```

## 显式 Builder config 导出

v05 数据集的完整 native-depth 导出（该录制保存的是已识别的 28D source
schema，所以必须显式启用 legacy-state converter）：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --rgbd-sidecar-source zarr \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --allow-legacy-state-conversion \
  --output-zarr ~/.cache/dp3_zarr/flexiv_dual_arm_3d_pick_place_20260713_v05_head_xyz_native.zarr
```

四条 FFS 路线都使用同一数据集、`head` 相机、首个 global frame 和固定 1024
点；四条路线共用 canonical Builder config。执行每条命令前，先把该文件中 active
的 native `depth_source` 块注释掉，再只解开对应的一个 FFS 完整配置块：

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

FFS 输出会记录 `depth_source=ffs_stereo`、backend/artifact contract、
normalization、disparity 参数、calibration 和 manifest SHA-256、可迁移的
artifact 文件名/相对路径、resolved Builder config 及其 hash，以及
PointCloudBuilder timing/count metadata。所有 FFS 输出的
`native_depth_used_for_builder` 都是 `false`。IR 缺失或非法、calibration 无效、
artifact/config/manifest hash 不一致，或 backend 初始化/推理失败时，都会中止并
清理临时输出；FFS 不会静默回退到 native depth。

两种模式都使用同一条下游链路：
`depth -> PointCloudBuilder -> deprojection -> RGB mapping -> crop -> fixed-size sampling`。

`--builder-config` 为必填参数，是 camera、点云格式、sampling 和 depth source 的
唯一来源；导出器不会再自动生成配置，也不接受这些配置项的 CLI 覆盖。两个
canonical Flexiv Builder config 为：

- `third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml` — native/FFS + xyz
- `third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml` — native/FFS + xyzrgb

当前仓库中的两个文件已经选择经过验证的 FFS 路线：`data_config.yaml` 使用
`tensorrt_two_stage` + `xyz`，`data_rgb_config.yaml` 使用
`tensorrt_plugin` + `xyzrgb`。两个文件中的 native-depth
`depth_source.mode: frame` 仍保持注释。若要选择 native depth，先注释 active 的
FFS mapping，再解开 native 块。四种 FFS backend 的完整配置仍集中在 canonical
文件中，因此切换路线只需要改 Builder YAML。

exporter 会按照原始 YAML 所在目录解析相对 artifact 路径，然后在输出旁写入
最终 resolved config。backend、artifact id、precision、optimization level 和
workspace 均从 YAML 读取。

需要强制使用新 raw sidecar 时传 `--rgbd-sidecar-source zarr`；需要强制使用没有
raw-sidecar manifest 的旧数据集时传 `--rgbd-sidecar-source parquet`。旧命令若省略
该参数仍保持兼容，因为默认值是 `auto`。

导出采用原子提交：各帧先写入同目录下的隐藏临时目录，随后校验
`state`、`action`、`point_cloud` 的 SHA-256。只有写入
`export_status=complete` 且 `expected_total_frames` 与 `converted_frames`
一致后，最终 `.zarr` 路径才会出现。因此中断的导出不会再被误认为完整训练集。

若要复用能够明确识别的旧 Flexiv 录制，必须显式启用 converter：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/legacy_lerobot \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --allow-legacy-state-conversion \
  --target-state-schema flexiv_abs_rot6d_v2 \
  --output-zarr ~/.cache/dp3_zarr/legacy_state_abs_rot6d_v2.zarr
```

Exporter 只接受 names/order 完全匹配的 Flexiv v1 28D absolute-rotvec 数据和 14D
action。未知 28D 数据或 metadata 冲突会 fail-fast；不会静默复用旧 Zarr 或旧
checkpoint。

## 检查 zarr

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/inspect_dp3_zarr.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz_state_abs_rot6d_v2.zarr
```

检查脚本会验证完成标记和保存的 SHA-256，再检查 `data/state`、
`data/action`、`data/point_cloud` 和 `meta/episode_ends`，打印 shape 和
数值范围，拒绝 NaN/Inf，检查 `episode_ends[-1] == T`，并打印 zarr
attrs；若 attrs 中保存了 source provenance，还会验证并单独显示。Flexiv 训练在
读取样本前也会执行相同的完成标记和校验和检查。

## 可视化 zarr 点云

使用 Open3D zarr 点云可视化脚本：

[visualize_zarr_pointcloud.py](visualizer/visualizer/visualize_zarr_pointcloud.py)

可以传入 zarr 根目录，也可以直接传入 `data/point_cloud` 数组路径。zarr 输入需要使用绝对路径；示例用 `~` 避免暴露机器相关 home 目录。

从 zarr 根目录可视化一帧：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python visualizer/visualizer/visualize_zarr_pointcloud.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz_state_abs_rot6d_v2.zarr \
  --frame 0
```

从 point-cloud 数组路径可视化一帧：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python visualizer/visualizer/visualize_zarr_pointcloud.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz_state_abs_rot6d_v2.zarr/data/point_cloud \
  --frame 0
```

常用选项：

```bash
--point-size 4
--background 1 1 1
--max-points 2048
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
`PointCloudBuilder` 路径：native 或 FFS stereo depth 反投影、裁剪、采样。Open3D 大窗口会并排显示
`raw`、`cropped`、`sampled` 三个点云视图，每个视图都支持鼠标旋转、平移和缩放。

从已导出的 zarr 调试一帧。脚本会从 zarr attrs 读取
`source_lerobot_path`、`camera`、`pointcloud_mode`、`num_points` 和保存的
`pointcloud_builder_config`，再回放原始 LeRobot RGB-D 帧：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/debug_zarr_pointcloud_stages.py \
  --dp3-zarr ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyzrgb_state_abs_rot6d_v2.zarr \
  --frame-index 0
```

默认情况下，`debug_zarr_pointcloud_stages.py` 会使用 `.zattrs` 中保存的 builder
config 快照，因此即使磁盘上的 YAML 已经修改，也能复现导出当时的 zarr。若要测试当前正在编辑的配置，显式传入：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/debug_zarr_pointcloud_stages.py \
  --dp3-zarr ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyzrgb_state_abs_rot6d_v2.zarr \
  --frame-index 0 \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml
```

不读取 zarr attrs，直接从 LeRobot 数据集调试：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/debug_lerobot_pointcloud_stages.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --frame-index 0 \
  --rgbd-sidecar-source auto \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml
```

两个阶段调试脚本都支持与 exporter 相同的 Builder YAML contract。调试 FFS
帧时必须传入显式 FFS Builder YAML；脚本会请求 `left_ir`/`right_ir`，校验录制的
IR/calibration join，再只把这些 IR 字段交给共享的 builder：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/debug_lerobot_pointcloud_stages.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_3d/pick_place_20260713_v05 \
  --frame-index 0 \
  --rgbd-sidecar-source zarr \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --no-show
```

如果输入是 FFS 派生 zarr，`debug_zarr_pointcloud_stages.py` 会使用 `.zattrs` 中的
Builder config snapshot，因此会复现 FFS 路线，不会回退到 native depth。只有在
有意检查另一份 YAML 时，才传入 `--builder-config`。

加上 `--no-show` 可以只打印三阶段 shape 和 metadata，不打开 Open3D GUI。

## DP3 zarr 结构

导出的 zarr 结构如下：

```text
data/state       (T, 34) float32
data/action      (T, 14) float32
data/point_cloud (T, N, 3) float32，对应 xyz
data/point_cloud (T, N, 6) float32，对应 xyzrgb
meta/episode_ends 累积 episode 结束位置，int64
```

可选 `data/img` 保持现有 RGB 语义。原始 `depth`、`left_ir`、`right_ir` 不会复制
进这个派生 DP3 Zarr。root attrs 会记录 source storage、manifest/calibration 的
路径和 hash、committed 计数、所选相机、原生 depth units/scale，以及
PointCloudBuilder config 和其来源。

在当前 DP3 dataset 代码中，`data/state` 会被加载为
`obs["agent_pos"]`，`data/point_cloud` 会被加载为 `obs["point_cloud"]`。

Flexiv real task 的状态契约是 `flexiv_abs_rot6d_v2`：每侧严格按七个关节、
absolute TCP `xyz`、六个 absolute rotation-6D 分量、归一化夹爪状态排列，共
34D。rotation-6D 是绝对 RDK world/base TCP 旋转矩阵的前两列
`[R[:, 0], R[:, 1]]`，不是前两行，也不依赖 Home、Quest 或相机坐标系。action
契约严格保持 14D：左右两侧 delta `xyz`、左右两侧 delta rotvec、最后两个夹爪
命令。

Exporter 会严格验证 LeRobot 的 state/action names/order 和 schema metadata。能够
被 exact names 识别的旧 Flexiv v1 28D absolute-rotvec 数据，可显式使用
`--allow-legacy-state-conversion` 离线转换；转换按
`Rotation.from_rotvec(...).as_matrix()` 取前两列，因此 π 附近 rotvec 的符号跳变
不会传播。未知 28D 数据会直接拒绝，输出名称包含 `state_abs_rot6d_v2`，不会和旧
Zarr 混淆。旧 v1 checkpoint 与新 runtime 不兼容，必须重新训练。

采集侧 LeRobot source 也可以使用 `flexiv_abs_rot6d_raw_force_v3`，其
`observation.state` shape 为 `(48,)`。DP3 target 仍严格是
`flexiv_abs_rot6d_v2` 的 `(34,)` state，action 仍为 `(14,)`。共享 source
contract 会在读取数据行之前严格验证 schema、shape、dtype、有限值以及完整且有序
的字段名；v3 到 v2 的 projection 按每个 target 字段名建立索引，绝不假设前 34
个位置就是 target。以下 14 个 source 字段会被删除：

```text
left_ee_ext_wrench_in_tcp_raw.fx/fy/fz/mx/my/mz
left_gripper_force
right_ee_ext_wrench_in_tcp_raw.fx/fy/fz/mx/my/mz
right_gripper_force
```

派生 Zarr 会记录 `source_state_schema`、`source_state_dim`、完整的
`source_state_names`、`state_transform=drop_raw_force_fields_v3_to_v2_by_name`
和 `dropped_state_names`。`raw_source_state_sha256` 覆盖完整 48D source，
`derived_state_sha256` 覆盖投影后的 34D；v3 时两者不同是预期行为。力/力矩字段
不会进入 DP3 Zarr 的 `data/state`、normalizer statistics、模型输入、checkpoint、
训练或在线推理。

仓库已经提供 XYZ 和 XYZRGB 两个真实任务 YAML。统一训练配置会根据所选任务的
`expected_pointcloud_dim` 自动决定是否使用颜色，并自动设置点云 encoder 输入通道数。

## 训练 Flexiv 双臂 DP3

Flexiv 训练的所有参数现在都放在
`3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_train_config.yaml`。
启动前至少检查这些字段：

```yaml
defaults:
  - task: real/flexiv_dual_arm_head_xyz  # 或 ..._xyzrgb

launcher:
  gpu_id: 0
  overwrite: false

algorithm: simple_dp3  # simple_dp3 或 dp3
task:
  dataset:
    zarr_path: /绝对路径/flexiv_head_xyz_state_abs_rot6d_v2.zarr
    max_train_episodes: 90

training:
  seed: 42
  resume: false

logging:
  mode: online  # online、offline 或 disabled
```

Flexiv Dataset 使用 `flexiv_abs_rot6d_v2` 归一化契约：在内存中复现采集 adapter 的
`0.02 m` 平移和 `0.04 rad` 旋转范数限幅（不会改写源 Zarr），为左右臂使用对称的
物理 action 尺度，两个夹爪统一按 `[0,1]` 映射；稳健 state 分位数和范围下限只用于
关节与 absolute `xyz`。12 个无量纲 rotation-6D 分量固定使用 `scale=1, offset=0`，
不会因低方差分位数或弧度 floor 被放大。训练启动时会打印 `[FlexivNormalizer]`
审计行；不要部署缺少 v2 schema、固定 rotation-6D scale 或完整 contract metadata
的 checkpoint。修改任一 normalizer 参数后必须创建新的训练运行，不能从旧 checkpoint
续训。

`launcher.gpu_id` 通过 `CUDA_VISIBLE_DEVICES` 选择物理 GPU；请保持
`training.device: cuda:0`，使训练进程正确使用映射后的显卡。训练 XYZRGB 时，
把 task 改成 `real/flexiv_dual_arm_head_xyzrgb` 并更新 zarr 路径即可，颜色开关和
6 通道 encoder 会自动解析。

激活环境后，训练只需要一条零参数命令：

```bash
conda activate dp3
bash scripts/train_flexiv_dual_arm_dp3.sh
```

输出目录由同一个 YAML 中的 `run_dir` 控制，默认解析为：

```text
outputs/<exp_name>_seed<seed>/checkpoints/
```

如果目标输出目录已经存在，脚本默认报错退出，避免混入旧 checkpoint、Hydra 配置和
WandB 文件。只有确认要删除整个旧运行目录时，才设置 `launcher.overwrite: true`。
如需继续中断的训练，保持 `overwrite: false`，设置 `training.resume: true`，并确认
`<run_dir>/checkpoints/latest.ckpt` 存在。恢复训练会从下一个 epoch 继续，不会重新执行
完整的 `num_epochs`；最后一个 epoch 即使不落在 `checkpoint_every` 周期上也会强制保存。
脚本不再接收位置训练参数或环境变量形式的超参数覆盖。

最小 sanity 训练可以临时在 YAML 中设置：

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

## Flexiv 双臂 DP3 推理

推理参数统一放在
`3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_inference_config.yaml`。
在该文件中设置 checkpoint、机器人配置、GPU、可选运行时长上限、控制频率、
动作队列模式、Flexiv 启动/servo 独立开关、推理专用 scheduler 与反向扩散步数、连接
机器人前的策略 warmup、动作限幅、点云配置和进程隔离的 Rerun 监控。当前默认以
10 Hz 依次执行配置的 action chunk；Flexiv 笛卡尔 servo thread 只有显式配置后才启用。

当前 epsilon checkpoint 使用 DDPM 训练，但部署时根据 checkpoint 的 beta schedule
重建 DDIM scheduler。DDIM 10 步的 batch-1 推理约为 39--40 ms；不要替换成 DDPM
10 步。双臂和双夹爪都使用模型输出，不做固定臂或固定夹爪的任务特定覆盖。

训练和推理 YAML 都显式包含模型结构、horizon、观测历史、扩散训练语义、点云 shape
和 state/action shape。推理入口会在连接机器人前，将这些权重契约参数与 checkpoint
内保存的训练配置逐项比较。`n_action_steps` 可在官方 DP3 切片范围内独立设置；
`use_ema` 用于选择 checkpoint 中实际存在的 EMA 或原始权重。

同步 `action_mode: chunk` 下，`inference.temporal_ensemble_coeff` 控制按未来时刻对齐的
chunk 重叠融合。设为 `0.0` 时完全保留原始队列逻辑；设为 `(0, 1]` 时表示新 chunk
权重。融合只作用于双臂 12 维笛卡尔位姿，两个夹爪始终使用新 chunk。重叠长度根据
`horizon`、`n_obs_steps` 和当前 `n_action_steps` 动态计算，并不限定四步 chunk。仓库
默认使用离线测试推荐值 `0.5`。

实时推理运行时已经完整收回本仓库：Flexiv adapter 和 RealSense RGB-D 实现位于
`third_party/real/dual_flexiv_rizon4s/interface`，不需要外部 Le-nero checkout，也不依赖
LeRobot Python 包。这与前文保留的离线 LeRobot 数据集格式兼容是两个独立边界。

安装最小机器人侧依赖，不要改变 DP3 的 Torch/CUDA 依赖栈：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python -m pip install -r third_party/real/dual_flexiv_rizon4s/requirements-runtime.txt
```

从脱敏模板创建本机私有配置，并填写所有硬件占位符：

```bash
cp third_party/real/dual_flexiv_rizon4s/configs/flexiv_runtime.example.yaml \
  third_party/real/dual_flexiv_rizon4s/configs/flexiv_runtime.local.yaml
```

本地配置已加入 `.gitignore`，不要提交真实机器人或相机序列号。如需使用其他路径，设置
`FLEXIV_DP3_ROBOT_CONFIG=~/.config/flexiv_dp3/config.yaml`。

真机动作前先运行独立的 perception-only 检查：

```bash
conda run -n dp3 bash scripts/run_flexiv_dp3_perception_only.sh
```

已激活 `dp3` 环境时：

```bash
bash scripts/run_flexiv_dp3_perception_only.sh
```

该程序只打开 `head_rgb` RealSense 和 `PointCloudBuilder`，不会导入 Flexiv RDK、连接
左右臂或发送动作。默认丢弃 60 帧 warmup，再检查 300 帧并显示 raw/cropped/sampled
感知视图；逐帧 JSONL 和汇总 JSON 写入 `logs/`。最近 15 帧的有效深度比例中位数低于
`0.75`、波动范围超过 `0.08`、点云发生 padding 或深度数组不拥有独立内存时，程序以
退出码 2 报告质量失败。无桌面环境时附加 `--no-visualize`。

perception-only 入口也支持显式 FFS 路线。在包含实时相机内参和 artifact 路径的
完整 FFS Builder YAML 中设置 `depth_source.mode: ffs_stereo`，然后通过
`--builder-config` 传入该 YAML；程序启动时会打开左右 IR 流，并在打开相机前完成
manifest/artifact 预检。例如：

```bash
PYTHONNOUSERSITE=1 ~/miniconda3/envs/dp3/bin/python tools/run_flexiv_dp3_perception_only.py \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --frames 30 \
  --no-visualize
```

完整推理部署只运行一条命令：

```bash
conda run -n dp3 bash scripts/run_flexiv_dual_arm_dp3_inference.sh
```

已经激活 `dp3` 环境时：

```bash
bash scripts/run_flexiv_dual_arm_dp3_inference.sh
```

这是会产生机器人运动的 `inference` 流程，会直接执行实时 RGB-D 反投影、裁剪、
2048 点采样、策略预测、动作过滤和 `robot.send_action()`；它与上述独立的无动作
perception-only 检查是两个入口。默认 Rerun telemetry 子进程通过固定大小的
shared-memory latest-only ring 接收数据，Viewer 卡顿或退出不会阻塞策略预测和 action send。

正式 launcher 的 live perception contract 完全来自所选 PointCloudBuilder YAML。
`native_depth` 模式保持 `use_depth=true` 且不启用 IR；`ffs_stereo` 模式打开左右
IR，并从同一次采集取得同步的 RGB/depth/left-IR/right-IR frameset。adapter 发布
`sidecar.head_left_ir`、`sidecar.head_right_ir`、`head_rgbd_timestamp`、
`head_rgbd_frame_index` 以及成对的 IR timestamp/frame index 字段。launcher 会把
这些 canonical observation key 映射到 Builder 配置的 `left_key`/`right_key`，
`xyzrgb` 时附带 RGB；native depth 不会传给 FFS Builder，也不会作为 fallback。
启动阶段会在连接机器人前校验 backend artifact、相机几何、帧 metadata、checkpoint
点云维度（`xyz=3`、`xyzrgb=6`）和固定点数。

### Rerun 实时监控

监控架构为：

```text
DP3 inference/control
        | 非阻塞固定大小写入
        v
multiprocessing.shared_memory latest-only rings（容量 3）
        v
独立 telemetry process
        +--> 本机 Viewer（spawn）
        +--> 手动或远端 Viewer（connect_grpc）
```

推理进程不导入 Rerun，也不调用 `rr.log`。RGB、depth、state、action、policy horizon
和点云只通过预分配共享内存传输；Queue/Pipe 仅传递 ready、stop、error 和 heartbeat
等少量控制消息。消费者只读取最新 slot，不回放积压帧。

安装可选依赖：

```bash
conda activate dp3
python -m pip install -e "visualizer[monitor]"
```

默认自动启动本机 Viewer：

```yaml
monitor:
  enabled: true
  min_bulk_slack_ms: 0.0
  viewer:
    mode: spawn
    port: 9876
    memory_limit: 2GB
    detach_process: true
    activate_blueprint_on_start: true
```

`detach_process: true` 表示推理结束后只断开本次 recording，不关闭 Rerun 窗口；下次
推理检测到 9876 已有 Viewer 后会直接复用。只有手动关闭 Viewer 窗口或其启动终端时
才会退出。`activate_blueprint_on_start: true` 会在每次 recording 开始时应用默认布局，
并把时间线设为 `log_time / Following`，防止运行期间停在空白旧时间点。

当前低负载实时显示配置为：

```yaml
monitor:
  rates:
    control_hz: ${inference.rate_hz}
    camera_hz: 2.0
    sampled_pointcloud_hz: 2.0
    stage_pointcloud_hz: 1.0
  payloads:
    rgb: true
    depth: true
    sampled_pointcloud: true
    raw_pointcloud: true
    cropped_pointcloud: true
  display:
    max_raw_points: 5000
    max_cropped_points: 5000
```

因此 RGB、depth、sampled point cloud 以最高 2 Hz 更新，raw/cropped 以最高 1 Hz
更新。5000 点上限只影响显示，不改变策略实际使用的 2048 点输入。当前使用
`min_bulk_slack_ms: 0`，是因为所有 bulk publish 都发生在 `robot.send_action()` 之后，
且普通监控发布实测 p99 小于 0.4 ms；之前设为 5 ms 时，现有约 126 ms 的推理周期
没有剩余 slack，导致 RGB/depth/点云几乎全部被跳过。如果另一台机器的 watchdog
余量不足，优先关闭 raw/cropped，或恢复正的 slack 阈值。

默认 Blueprint 中各纵轴单位为：joint 和 action rotvec 使用弧度，TCP/action xyz 使用
米，rotation-6D 无量纲，夹爪为 `[0,1]`，timing 为毫秒。若要保留手工调整后的布局，
先保存 `.rbl`，再设置 `activate_blueprint_on_start: false`，并在 Viewer 中手动选择
Following。

本机手动 Viewer：

```bash
rerun --port 9876 --memory-limit 2GB
```

```yaml
monitor:
  viewer:
    mode: connect
    url: rerun+http://127.0.0.1:9876/proxy
```

远端 Viewer 主机运行：

```bash
rerun --bind 0.0.0.0 --port 9876 --memory-limit 2GB
```

推理主机设置 `monitor.viewer.mode: connect` 和
`monitor.viewer.url: rerun+http://<viewer-ip>:9876/proxy`。只应通过可信局域网、VPN
或防火墙受限端口连接。

完全 synthetic、不会连接 Flexiv/RealSense 或发送 action 的资源基准：

```bash
python tools/benchmark_dp3_monitor.py
```

报告写入 `logs/monitor_benchmark/<timestamp>/`。其中分别统计 producer、telemetry 和
Viewer 的 CPU/RSS，以及 publish 延迟、周期抖动、deadline miss、drop/overwrite 和
实际发布率。

默认配置会持续闭环推理，直到在当前终端按 `Ctrl+C`。启动器也会打印 JSONL 日志位置
和停止命令；也可以在另一个终端执行：

```bash
touch /tmp/stop_flexiv_dp3_inference
```

还保留一个可选的无硬件配置检查，但它不属于正常部署流程：

```bash
conda run -n dp3 bash scripts/run_flexiv_dual_arm_dp3_inference.sh --check-config
```

配置检查要求 checkpoint、点云 YAML 和本地机器人 YAML 均存在，但会在
`robot.connect()` 之前退出，也不会打开 RealSense pipeline。正常 inference 会移动机器人；
自动化测试不能替代后续由操作者执行的 RealSense-only、Flexiv 连接和最终闭环测试。
Codex 在本次迁移中没有执行任何硬件连接、相机 pipeline 或实时 inference 命令。

参数分类、完整运行链路和停止行为见
[docs/flexiv_dual_arm_inference.md](docs/flexiv_dual_arm_inference.md)。
