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
cd 3D-Diffusion-Policy
export PYTHONPATH=$PWD/PointCloudBuilder:$PWD/3D-Diffusion-Policy:$PYTHONPATH
```

## 默认输出路径

如果不传 `--output-zarr`，导出脚本会默认写入：

```text
~/.cache/dp3_zarr/<lerobot_repo_id>_<camera>_<pointcloud-mode>.zarr
```

脚本会优先读取 `meta/info.json` 里的 `repo_id`。如果本地数据集没有保存
`repo_id`，则回退为相对 `~/.cache/huggingface/lerobot` 的路径。
repo id 里的路径分隔符和不适合作为文件名的字符会被替换成 `_`。

以下示例数据集的默认 `xyz` 输出路径是：

```text
~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr
```

只有需要改写输出位置时，才传入 `--output-zarr`。

## 导出 xyz 点云

```bash
python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --camera head \
  --pointcloud-mode xyz \
  --num-points 1024 \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_config.yaml \
  --overwrite
```

## 导出 xyzrgb 点云

```bash
python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --camera head \
  --pointcloud-mode xyzrgb \
  --num-points 1024 \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml \
  --overwrite
```

如果不提供 `--builder-config`，脚本会在输出 zarr 旁边写入自动生成的
`*.pointcloud_builder.yaml`。该配置保存 depth intrinsics、color intrinsics、
depth scale；在 `xyzrgb` 模式下还会保存 `depth_to_color` 变换。

## 检查 zarr

```bash
python tools/inspect_dp3_zarr.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr
```

检查脚本会验证 `data/state`、`data/action`、`data/point_cloud` 和
`meta/episode_ends`，打印 shape 和数值范围，拒绝 NaN/Inf，检查
`episode_ends[-1] == T`，并打印 zarr attrs。

## 可视化 zarr 点云

使用 Open3D zarr 点云可视化脚本：

[visualize_zarr_pointcloud.py](visualizer/visualizer/visualize_zarr_pointcloud.py)

可以传入 zarr 根目录，也可以直接传入 `data/point_cloud` 数组路径。zarr 输入需要使用绝对路径；示例用 `~` 避免暴露机器相关 home 目录。

从 zarr 根目录可视化一帧：

```bash
python visualizer/visualizer/visualize_zarr_pointcloud.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr \
  --frame 0
```

从 point-cloud 数组路径可视化一帧：

```bash
python visualizer/visualizer/visualize_zarr_pointcloud.py \
  --zarr-path ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyz.zarr/data/point_cloud \
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
  --dp3-zarr ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyzrgb.zarr \
  --frame-index 0
```

默认情况下，`debug_zarr_pointcloud_stages.py` 会使用 `.zattrs` 中保存的 builder
config 快照，因此即使磁盘上的 YAML 已经修改，也能复现导出当时的 zarr。若要测试当前正在编辑的配置，显式传入：

```bash
python tools/debug_zarr_pointcloud_stages.py \
  --dp3-zarr ~/.cache/dp3_zarr/flexiv_dual_arm_test_pick_place_20260708_v02_head_xyzrgb.zarr \
  --frame-index 0 \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml
```

不读取 zarr attrs，直接从 LeRobot 数据集调试：

```bash
python tools/debug_lerobot_pointcloud_stages.py \
  --lerobot-path ~/.cache/huggingface/lerobot/flexiv_dual_arm_test/pick_place_20260708_v02 \
  --frame-index 0 \
  --camera head \
  --pointcloud-mode xyzrgb \
  --num-points 1024 \
  --builder-config third_party/real/dual_flexiv_rizon4s/configs/data_rgb_config.yaml
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

## 训练 Flexiv 双臂 DP3

训练封装脚本会从已导出的 DP3 zarr 启动训练，并允许通过第 4 个位置参数显式选择物理 GPU。
脚本内部会设置 `CUDA_VISIBLE_DEVICES=<gpu_id>`，因此训练进程内部仍使用
`cuda:0` 指向被选中的那张卡。

XYZ 点云：

```bash
conda run -n dp3 bash scripts/train_flexiv_dual_arm_dp3.sh \
  xyz \
  /path/to/flexiv_head_xyz.zarr \
  simple_dp3 \
  0 \
  42
```

XYZRGB 点云：

```bash
conda run -n dp3 bash scripts/train_flexiv_dual_arm_dp3.sh \
  xyzrgb \
  /path/to/flexiv_head_xyzrgb.zarr \
  simple_dp3 \
  0 \
  42
```

参数格式：

```text
<xyz|xyzrgb> <zarr_path> [simple_dp3|dp3] [gpu_id] [seed] [hydra_overrides...] [--overwrite]
```

默认 checkpoint 输出到当前仓库内的相对路径：

```text
outputs/<exp_name>_seed<seed>/checkpoints/
```

如需指定整个 Hydra 输出目录，设置 `RUN_DIR=/custom/output/dir`。如果目标输出目录已经存在，
训练封装脚本默认会报错退出，避免旧 checkpoint、Hydra 配置和 WandB 文件与新训练混在一起。
只有显式添加 `--overwrite` 时，脚本才会先删除整个目标输出目录再开始训练。

常用环境变量包括
`SAVE_CKPT=True|False`、`WANDB_MODE=offline|online|disabled`、`BATCH_SIZE`、
`NUM_WORKERS`、`MAX_TRAIN_EPISODES` 和 `EXP_NAME`。

Flexiv 训练统一读取
`3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_train_config.yaml`。
其中 `algorithm` 可选 `simple_dp3` 或 `dp3`，训练脚本也会根据第三个位置参数覆盖该字段。
模型结构、观测/动作时域和扩散调度器属于训练推理一致参数；优化器、EMA 更新、
dataloader、epoch、日志和 checkpoint 保存属于训练专用参数。

最小 sanity 训练：

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
  training.sample_every=999999
```

## Flexiv 双臂 DP3 推理

推理参数统一放在
`3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_inference_config.yaml`。
在该文件中设置 checkpoint、机器人配置、GPU、可选运行时长上限、控制频率、
receding/chunk 动作模式、反向扩散步数、动作限幅、点云配置和 Open3D 可视化。

训练和推理 YAML 都显式包含模型结构、时域、扩散调度器、点云 shape、state/action
shape 和 EMA 选择。推理入口会在连接机器人前，将这些参数与 checkpoint 内保存的训练
配置逐项比较；只有推理专用参数允许不同。

实时推理运行时已经完整收回本仓库：Flexiv adapter 和 RealSense RGB-D 实现位于
`third_party/real/dual_flexiv_rizon4s/interface`，不需要外部 Le-nero checkout，也不依赖
LeRobot Python 包。这与前文保留的离线 LeRobot 数据集格式兼容是两个独立边界。

安装最小机器人侧依赖，不要改变 DP3 的 Torch/CUDA 依赖栈：

```bash
python -m pip install -r third_party/real/dual_flexiv_rizon4s/requirements-runtime.txt
```

从脱敏模板创建本机私有配置，并填写所有硬件占位符：

```bash
cp third_party/real/dual_flexiv_rizon4s/configs/flexiv_runtime.example.yaml \
  third_party/real/dual_flexiv_rizon4s/configs/flexiv_runtime.local.yaml
```

本地配置已加入 `.gitignore`，不要提交真实机器人或相机序列号。如需使用其他路径，设置
`FLEXIV_DP3_ROBOT_CONFIG=/absolute/path/to/config.yaml`。

完整推理部署只运行一条命令：

```bash
conda run -n dp3 bash scripts/run_flexiv_dual_arm_dp3_inference.sh
```

已经激活 `dp3` 环境时：

```bash
bash scripts/run_flexiv_dual_arm_dp3_inference.sh
```

这是会产生机器人运动的 `inference` 流程，会直接执行实时 RGB-D 反投影、裁剪、
1024 点采样、策略预测、动作过滤和 `robot.send_action()`；不再包含独立的无动作阶段或
模式切换流程。默认 Open3D 显示进程以 2 Hz 运行，通过容量为 1 的 latest-frame
队列接收数据，不会阻塞控制循环。

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
