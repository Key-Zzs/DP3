# 代码仓库中文解析报告

本报告面向刚接触具身智能、机器人控制和深度学习策略学习的读者。阅读范围以当前仓库真实文件为准，排除了 `.git`、缓存、日志、输出、数据集、权重和 `third_party` 大目录。本次重点阅读约 66 个文件，并全量扫描了 `diffusion_policy_3d/config/task/` 下 61 个任务配置文件的命名和组织方式。

几个术语先说明：

- 模仿学习：用专家示范数据学习从 observation 到 action 的策略。
- Diffusion Policy：把动作轨迹看成要从噪声中逐步去噪生成的序列，训练时给动作加噪，模型学习还原目标。
- Point cloud：点云，通常是 `(N, 3)` 的 xyz 坐标，或 `(N, 6)` 的 xyzrgb。
- Proprioception：机器人自身状态，例如关节角、末端执行器位置、夹爪状态等。本仓库主要用 `agent_pos` 表示。
- Horizon：时间窗口长度。这里 `horizon` 是模型预测的轨迹长度，`n_obs_steps` 是输入观察步数，`n_action_steps` 是每次执行的动作步数。

## 1. 仓库总览

| 项目 | 结论 | 依据 |
| --- | --- | --- |
| 仓库名称 | 3D Diffusion Policy，简称 DP3 | `README.md` 标题和简介 |
| 主要任务 | 3D 点云视觉模仿学习：从点云和机器人状态预测连续控制动作 | `README.md`，`policy/dp3.py`，`model/vision/pointnet_extractor.py` |
| 所属方向 | imitation learning、diffusion policy、3D point cloud policy、robot manipulation、sim-to-real 数据训练 | `README.md`、`config/task/*.yaml`、`env_runner/*.py` |
| 主要输入 | `point_cloud`、`agent_pos`，DexArt 还可用 `imagin_robot`；环境还产生 `image`、`depth`，但当前 DP3Encoder 不读取 RGB/depth | `dataset/*.py`，`model/vision/pointnet_extractor.py`，`env/*/*wrapper.py` |
| 主要输出 | 连续动作轨迹 chunk，形状为 `(B, n_action_steps, action_dim)` | `policy/dp3.py:predict_action()` |
| 支持环境/数据 | Adroit、DexArt、MetaWorld、RealDex zarr 数据 | `config/task/*.yaml`，`dataset/*.py`，`env_runner/*.py` |
| 训练入口 | `3D-Diffusion-Policy/train.py`，外层脚本 `scripts/train_policy.sh` | `train.py`，`scripts/train_policy.sh` |
| 评估/推理入口 | `3D-Diffusion-Policy/eval.py`，外层脚本 `scripts/eval_policy.sh` | `eval.py`，`scripts/eval_policy.sh` |
| 数据处理入口 | 仿真演示生成脚本和真实机器人数据转换示例 | `scripts/gen_demonstration_*.sh`，`scripts/convert_real_robot_data.py` |
| 主要依赖 | PyTorch、diffusers、Hydra/OmegaConf、zarr、wandb、PyTorch3D、Open3D、MuJoCo/mujoco_py、gym、MetaWorld、DexArt、mj_envs/mjrl | `INSTALL.md`，各源码 import |

仓库有一个容易混淆的结构：当前仓库根目录下还有一个同名子目录 `3D-Diffusion-Policy/`。真正的 Python 包 `diffusion_policy_3d`、训练入口 `train.py` 和评估入口 `eval.py` 都在这个同名子目录里。外层根目录主要放 README、INSTALL、脚本和 `visualizer` 小包。

整体任务可以概括为：

```text
专家示范 zarr 数据或仿真环境 observation
-> 点云和机器人状态编码
-> 条件 1D U-Net 扩散模型生成动作轨迹
-> 执行动作 chunk
-> 在环境中统计成功率、回报和视频
```

## 2. 目录结构与文件功能

### 2.1 核心目录总表

| 路径 | 类型 | 功能 | 初学者是否需要重点阅读 | 备注 |
| --- | --- | --- | --- | --- |
| `README.md` | 文档 | 项目背景、数据、训练和评估命令、真实机器人数据格式 | 是 | 有些路径/脚本细节需要和源码交叉检查 |
| `INSTALL.md` | 文档 | 旧版环境安装流程 | 是 | 面向 Python 3.8 和较旧 CUDA 生态，现代 GPU 可能要调整 |
| `ERROR_CATCH.md` | 文档 | 常见安装/运行报错 | 可选 | 复现时很有用 |
| `scripts/` | 脚本目录 | 训练、评估、演示生成、真实数据转换 | 是 | 入口命令都在这里 |
| `3D-Diffusion-Policy/` | 源码根目录 | Python 包、Hydra 配置、训练/评估入口 | 是 | 进入该目录后运行 `python train.py` |
| `3D-Diffusion-Policy/diffusion_policy_3d/config/` | 配置 | Hydra 主配置和任务配置 | 是 | 决定模型、数据集、环境 runner |
| `3D-Diffusion-Policy/diffusion_policy_3d/policy/` | 策略 | DP3 和 SimpleDP3 策略类 | 是 | 核心算法第一入口 |
| `3D-Diffusion-Policy/diffusion_policy_3d/model/vision/` | 模型 | 点云和状态编码器 | 是 | `DP3Encoder` 是 observation 进模型的位置 |
| `3D-Diffusion-Policy/diffusion_policy_3d/model/diffusion/` | 模型 | 1D 条件 U-Net、mask、EMA、扩散时间 embedding | 是 | 动作轨迹去噪网络 |
| `3D-Diffusion-Policy/diffusion_policy_3d/model/common/` | 工具 | 归一化器、scheduler、device/dtype 工具 | 是 | 归一化是训练/推理一致性的关键 |
| `3D-Diffusion-Policy/diffusion_policy_3d/dataset/` | 数据集 | zarr 数据读取、采样、转换成 batch | 是 | 理解数据流必读 |
| `3D-Diffusion-Policy/diffusion_policy_3d/common/` | 通用工具 | replay buffer、sequence sampler、checkpoint、日志统计 | 是 | 数据切片和 checkpoint 在这里 |
| `3D-Diffusion-Policy/diffusion_policy_3d/env/` | 环境包装 | Adroit、DexArt、MetaWorld 环境 observation/action 封装 | 是 | 看 observation 如何从仿真中产生 |
| `3D-Diffusion-Policy/diffusion_policy_3d/env_runner/` | 评估 runner | rollout、policy inference、环境 step、指标和视频 | 是 | 评估/推理调用链核心 |
| `3D-Diffusion-Policy/diffusion_policy_3d/gym_util/` | 环境工具 | 多步 observation/action wrapper、点云生成、视频记录 | 是 | 动作 chunk 执行和点云坐标处理在这里 |
| `visualizer/` | 小包 | Plotly/Flask 点云可视化 | 可选 | 用于调试点云，不参与训练 |
| `third_party/sim/` | 仿真外部依赖 | DexArt、MetaWorld、mujoco_py、gym 等外部代码 | 本报告未深入 | 用户要求排除大型外部目录 |

### 2.2 关键文件功能表

| 路径 | 类型 | 功能 | 初学者是否需要重点阅读 | 备注 |
| --- | --- | --- | --- | --- |
| `3D-Diffusion-Policy/train.py` | Python 入口 | `TrainDP3Workspace` 负责训练全流程 | 是 | dataset、model、optimizer、EMA、rollout、checkpoint 都在这里串起来 |
| `3D-Diffusion-Policy/eval.py` | Python 入口 | 创建 `TrainDP3Workspace` 并调用 `eval()` | 是 | 不单独实现评估逻辑 |
| `3D-Diffusion-Policy/setup.py` | 安装脚本 | 安装 `diffusion_policy_3d` 包 | 可选 | 只有 `find_packages()` |
| `diffusion_policy_3d/config/dp3.yaml` | Hydra 主配置 | 原始 DP3 配置，较大的 U-Net | 是 | `_target_` 指向 `policy.dp3.DP3` |
| `diffusion_policy_3d/config/simple_dp3.yaml` | Hydra 主配置 | 简化版 DP3，较小 U-Net | 是 | `_target_` 指向 `policy.simple_dp3.SimpleDP3` |
| `diffusion_policy_3d/config/task/*.yaml` | Hydra 任务配置 | 61 个任务的 observation/action 维度、dataset、env_runner | 是 | 建议先读代表文件：`adroit_hammer.yaml`、`dexart_bucket.yaml`、`metaworld_assembly.yaml`、`realdex_drill.yaml` |
| `diffusion_policy_3d/policy/base_policy.py` | 抽象类 | 定义 `predict_action()` 和 `set_normalizer()` 接口 | 是 | env_runner 只依赖这个接口 |
| `diffusion_policy_3d/policy/dp3.py` | 策略 | DP3 策略，含初始化、采样推理、loss | 是 | 最核心文件之一 |
| `diffusion_policy_3d/policy/simple_dp3.py` | 策略 | SimpleDP3 策略，逻辑几乎同 DP3 | 是 | 使用更简化的 U-Net 实现 |
| `diffusion_policy_3d/model/vision/pointnet_extractor.py` | 模型 | `DP3Encoder` 编码点云和 `agent_pos` | 是 | 当前没有真正使用 RGB/depth |
| `diffusion_policy_3d/model/diffusion/conditional_unet1d.py` | 模型 | DP3 使用的条件 1D U-Net | 是 | 支持 FiLM、add、cross attention 等条件方式 |
| `diffusion_policy_3d/model/diffusion/simple_conditional_unet1d.py` | 模型 | SimpleDP3 使用的轻量 1D U-Net | 是 | 少了一些 residual block |
| `diffusion_policy_3d/model/diffusion/mask_generator.py` | 模型工具 | 生成条件 mask，用于 inpainting 风格扩散 | 是 | 默认 `obs_as_global_cond=True` 时 action 全部不 visible |
| `diffusion_policy_3d/model/diffusion/ema_model.py` | 训练工具 | 指数滑动平均模型 | 是 | 训练默认 `use_ema=True` |
| `diffusion_policy_3d/model/diffusion/conv1d_components.py` | 模型组件 | 1D 卷积、下采样、上采样模块 | 可选 | U-Net 基础块 |
| `diffusion_policy_3d/model/diffusion/positional_embedding.py` | 模型组件 | 扩散 timestep 的正弦位置编码 | 可选 | 和 Transformer 常见 positional embedding 类似 |
| `diffusion_policy_3d/model/common/normalizer.py` | 数据处理 | `LinearNormalizer` 归一化/反归一化 | 是 | action 输出必须反归一化后执行 |
| `diffusion_policy_3d/model/common/lr_scheduler.py` | 训练工具 | 封装 diffusers scheduler | 可选 | `train.py` 调用 |
| `diffusion_policy_3d/model/common/module_attr_mixin.py` | 工具 | 提供 `device` 和 `dtype` 属性 | 可选 | policy/env_runner 会用 |
| `diffusion_policy_3d/dataset/base_dataset.py` | 抽象类 | dataset 接口 | 是 | 定义 batch 格式 |
| `diffusion_policy_3d/dataset/adroit_dataset.py` | 数据集 | Adroit zarr 数据读取 | 是 | keys: `state/action/point_cloud/img` |
| `diffusion_policy_3d/dataset/dexart_dataset.py` | 数据集 | DexArt zarr 数据读取 | 是 | 额外读取 `imagin_robot` |
| `diffusion_policy_3d/dataset/metaworld_dataset.py` | 数据集 | MetaWorld zarr 数据读取 | 是 | keys: `state/action/point_cloud` |
| `diffusion_policy_3d/dataset/realdex_dataset.py` | 数据集 | 真实机器人 RealDex zarr 数据读取 | 是 | keys: `state/action/point_cloud/img` |
| `diffusion_policy_3d/common/replay_buffer.py` | 数据结构 | zarr replay buffer 复制、保存、访问 | 是 | 约定 `/data/*` 和 `/meta/episode_ends` |
| `diffusion_policy_3d/common/sampler.py` | 数据采样 | 按 episode 生成固定长度序列，支持边界 padding | 是 | dataset 的 `__getitem__()` 依赖它 |
| `diffusion_policy_3d/common/checkpoint_util.py` | checkpoint | Top-K checkpoint 管理 | 是 | 按 `test_mean_score` 保存最优 |
| `diffusion_policy_3d/common/logger_util.py` | 日志工具 | 记录 largest-K 成功率均值 | 可选 | env_runner 使用 |
| `diffusion_policy_3d/common/pytorch_util.py` | 工具 | 递归 dict tensor 转换、optimizer device 转移 | 是 | train 和 runner 都频繁调用 |
| `diffusion_policy_3d/env_runner/adroit_runner.py` | 评估 | Adroit rollout、成功率、视频 | 是 | 调用 `policy.predict_action()` |
| `diffusion_policy_3d/env_runner/dexart_runner.py` | 评估 | DexArt rollout、成功率、视频 | 是 | 输入包含 `imagin_robot` |
| `diffusion_policy_3d/env_runner/metaworld_runner.py` | 评估 | MetaWorld rollout、成功率、视频 | 是 | `save_video` 默认 false |
| `diffusion_policy_3d/env/adroit/adroit.py` | 环境 | Adroit 像素和状态封装 | 是 | 点云由外层 wrapper 添加 |
| `diffusion_policy_3d/env/dexart/dexart_wrapper.py` | 环境 | DexArt 环境封装，读取 RGB/depth/state/point_cloud/imagination_robot | 是 | 动作维度来自 robot dof |
| `diffusion_policy_3d/env/metaworld/metaworld_wrapper.py` | 环境 | MetaWorld 环境封装，生成点云和 state | 是 | 有相机位置、点云旋转/裁剪 |
| `diffusion_policy_3d/gym_util/multistep_wrapper.py` | wrapper | 将单步 env 包成多步 observation 和 action chunk 接口 | 是 | 推理时一次执行 `n_action_steps` |
| `diffusion_policy_3d/gym_util/mujoco_point_cloud.py` | 点云 | MuJoCo depth 到点云的相机内参/外参处理 | 是 | 坐标系分析重点 |
| `diffusion_policy_3d/gym_util/mjpc_diffusion_wrapper.py` | 点云 wrapper | Adroit 点云裁剪、变换、采样并加入 obs | 是 | 当前 runner 使用这个文件 |
| `diffusion_policy_3d/gym_util/mjpc_wrapper.py` | 点云 wrapper | 旧式 dm_env 风格点云 wrapper | 可选 | 与上一个文件功能相近 |
| `diffusion_policy_3d/gym_util/video_recording_wrapper.py` | wrapper | rollout 视频帧记录 | 可选 | env_runner 上传 wandb |
| `scripts/train_policy.sh` | shell | 封装训练命令 | 是 | 参数：算法、任务、附加名、seed、GPU |
| `scripts/eval_policy.sh` | shell | 封装评估命令 | 是 | 存在变量未定义风险，见第 13 节 |
| `scripts/gen_demonstration_adroit.sh` | shell | 调用 VRL3 生成 Adroit expert zarr | 可选 | 依赖 `third_party/sim/VRL3` |
| `scripts/gen_demonstration_dexart.sh` | shell | 调用 DexArt expert 生成示范 | 可选 | 依赖 DexArt assets 和 checkpoint |
| `scripts/gen_demonstration_metaworld.sh` | shell | 调用 MetaWorld expert 生成示范 | 可选 | 依赖 `third_party/sim/Metaworld` |
| `scripts/convert_real_robot_data.py` | Python | 真实机器人 pickle 数据转 zarr 示例 | 可选 | 路径硬编码，不是通用 CLI |
| `scripts/find_gpu.sh` | shell | 用 `nvidia-smi` 选显存占用最小 GPU | 可选 | 当前 `train_policy.sh` 注释掉了自动调用 |
| `visualizer/visualizer/pointcloud.py` | 工具 | Plotly/Flask 点云可视化 | 可选 | 暴露 `visualize_pointcloud()` 和 `Visualizer` |

### 2.3 任务配置分组

`diffusion_policy_3d/config/task/` 下共有 61 个 `.yaml`：

- Adroit: `adroit_door.yaml`、`adroit_hammer.yaml`、`adroit_pen.yaml`。
- DexArt: `dexart_bucket.yaml`、`dexart_faucet.yaml`、`dexart_laptop.yaml`、`dexart_toilet.yaml`。
- MetaWorld: 50 个左右任务，例如 `metaworld_assembly.yaml`、`metaworld_basketball.yaml`、`metaworld_pick-place.yaml`、`metaworld_window-open.yaml` 等。
- RealDex: `realdex_drill.yaml`、`realdex_dumpling.yaml`、`realdex_pour.yaml`、`realdex_roll.yaml`。

它们都遵循类似结构：

```yaml
name: ...
task_name: ...
shape_meta:
  obs:
    point_cloud: {shape: [...], type: point_cloud}
    agent_pos: {shape: [...], type: low_dim}
  action:
    shape: [...]
env_runner:
  _target_: ...
dataset:
  _target_: ...
  zarr_path: ...
```

## 3. 核心算法与数据流

### 3.1 总体数据流

```text
zarr 示范数据或仿真环境
-> ReplayBuffer.copy_from_path()
-> SequenceSampler.sample_sequence()
-> Dataset.__getitem__()
-> batch = {"obs": {...}, "action": ...}
-> LinearNormalizer.normalize()
-> DP3Encoder(point_cloud, agent_pos, optional imagin_robot)
-> ConditionalUnet1D(noisy action trajectory, diffusion timestep, obs global condition)
-> MSE loss
-> optimizer.step(), EMA update
-> checkpoint 保存 normalizer/model/optimizer/global_step/epoch
-> eval 加载 checkpoint
-> policy.predict_action(obs_dict)
-> conditional_sample() 从随机噪声迭代去噪
-> action 反归一化
-> MultiStepWrapper.step(action_chunk)
```

### 3.2 数据集读取

对应文件：

- `diffusion_policy_3d/common/replay_buffer.py`
- `diffusion_policy_3d/common/sampler.py`
- `diffusion_policy_3d/dataset/*.py`

zarr 数据格式约定：

```text
zarr_root/
  data/
    state
    action
    point_cloud
    img
    imagin_robot   # DexArt
    depth          # convert_real_robot_data.py 会保存，但当前 dataset 不读取 depth
  meta/
    episode_ends
```

`ReplayBuffer.copy_from_path(zarr_path, keys=...)` 会把 zarr 中指定 key 复制到内存型 root。`SequenceSampler` 根据 `episode_ends` 把连续 episode 切成长度为 `horizon` 的样本，并用 `pad_before`、`pad_after` 在 episode 边界补齐。

各 dataset 的输出格式一致：

```python
{
    "obs": {
        "point_cloud": Tensor[T, N, C],
        "agent_pos": Tensor[T, D],
        # DexArt 额外有:
        "imagin_robot": Tensor[T, 96, 7],
    },
    "action": Tensor[T, action_dim],
}
```

初学者容易误解的点：

- zarr 中 `state` 会被 dataset 重命名成模型输入里的 `agent_pos`。
- `img` 会被读取到 replay buffer，但当前 `_sample_to_data()` 没把它放进 `obs`，所以 DP3 主模型并没有用 RGB。
- `depth` 在真实机器人转换脚本会保存，但 `RealDexDataset` 当前没有读取。
- DexArt 的 `point_cloud` 和 `imagin_robot` 归一化器是 identity，而不是用数据 min/max 归一化。

### 3.3 模型输入构造

对应文件：

- `diffusion_policy_3d/policy/dp3.py`
- `diffusion_policy_3d/policy/simple_dp3.py`
- `diffusion_policy_3d/model/vision/pointnet_extractor.py`

`DP3.__init__()` 从 `shape_meta` 读取动作维度和 observation 形状：

- `shape_meta["action"]["shape"]` 决定 `action_dim`。
- `shape_meta["obs"]` 决定 `DP3Encoder` 的 `observation_space`。

默认 `obs_as_global_cond=True`，所以 observation 不和动作拼在同一 trajectory 里，而是先被 `DP3Encoder` 编成全局条件：

```text
point_cloud + optional imagin_robot
-> PointNetEncoderXYZ 或 PointNetEncoderXYZRGB
-> pn_feat

agent_pos
-> state_mlp
-> state_feat

concat(pn_feat, state_feat)
-> obs_feature

n_obs_steps 个 obs_feature 展平
-> global_cond
```

`DP3Encoder.forward()` 的核心逻辑：

```python
points = observations["point_cloud"]
if "imagin_robot" in observations:
    points = concat(points, observations["imagin_robot"][..., :points.shape[-1]], dim=1)
pn_feat = pointnet(points)
state_feat = state_mlp(observations["agent_pos"])
return concat(pn_feat, state_feat)
```

### 3.4 扩散策略网络

对应文件：

- `diffusion_policy_3d/model/diffusion/conditional_unet1d.py`
- `diffusion_policy_3d/model/diffusion/simple_conditional_unet1d.py`
- `diffusion_policy_3d/model/diffusion/conv1d_components.py`
- `diffusion_policy_3d/model/diffusion/positional_embedding.py`

训练时 trajectory 是规范化动作 `nactions`，形状约为：

```text
B: batch size
T: horizon
Da: action_dim
trajectory: (B, T, Da)
```

`ConditionalUnet1D.forward(sample, timestep, global_cond)` 会：

1. 把 `(B, T, Da)` rearrange 为 `(B, Da, T)` 以适配 1D 卷积。
2. 用 `SinusoidalPosEmb` 编码扩散 timestep。
3. 把 timestep embedding 与 observation global condition 拼接。
4. 通过下采样 residual block、中间 block、上采样 block。
5. 输出同形状 `(B, T, Da)` 的预测。

DP3 和 SimpleDP3 的差别：

- `DP3` 使用 `model/diffusion/conditional_unet1d.py`，每个 down/up stage 有更多 residual block。
- `SimpleDP3` 使用 `model/diffusion/simple_conditional_unet1d.py`，结构更轻量。
- 配置上 `dp3.yaml` 的 `down_dims` 是 `[512, 1024, 2048]`，`simple_dp3.yaml` 是 `[128, 256, 384]`。

### 3.5 Loss 计算

对应函数：

- `DP3.compute_loss(batch)`
- `SimpleDP3.compute_loss(batch)`

伪代码：

```python
def compute_loss(batch):
    nobs = normalizer.normalize(batch["obs"])
    nactions = normalizer["action"].normalize(batch["action"])

    if not use_pc_color:
        nobs["point_cloud"] = nobs["point_cloud"][..., :3]

    obs_features = obs_encoder(first_n_obs_steps(nobs))
    global_cond = flatten_time(obs_features)

    trajectory = nactions
    condition_mask = mask_generator(trajectory.shape)
    noise = randn_like(trajectory)
    timesteps = random_int(0, num_train_timesteps, size=B)
    noisy_trajectory = noise_scheduler.add_noise(trajectory, noise, timesteps)
    noisy_trajectory[condition_mask] = trajectory[condition_mask]

    pred = model(noisy_trajectory, timesteps, global_cond)

    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "sample":
        target = trajectory
    elif prediction_type == "v_prediction":
        target = velocity_target(...)

    loss = mse(pred, target) masked by ~condition_mask
    return loss, {"bc_loss": loss.item()}
```

当前 `dp3.yaml` 和 `simple_dp3.yaml` 的 scheduler 配置是 `prediction_type: sample`，所以默认监督目标是干净动作轨迹，而不是噪声 `epsilon`。

### 3.6 推理采样

对应函数：

- `DP3.predict_action(obs_dict)`
- `DP3.conditional_sample(...)`

推理时：

1. 对输入 observation 用 checkpoint 中保存的 normalizer 归一化。
2. 只取前 `n_obs_steps` 步 observation。
3. 用 `DP3Encoder` 得到 global condition。
4. 从随机高斯噪声初始化 `trajectory`。
5. `scheduler.set_timesteps(num_inference_steps)`，默认 `num_inference_steps=10`。
6. 每个 timestep 调一次 U-Net 和 scheduler step。
7. 得到规范化动作 `naction_pred`。
8. 用 `normalizer["action"].unnormalize()` 反归一化。
9. 返回从 `start = n_obs_steps - 1` 开始的 `n_action_steps` 个动作。

输出：

```python
{
    "action": Tensor[B, n_action_steps, action_dim],
    "action_pred": Tensor[B, horizon, action_dim],
}
```

初学者容易误解的点：

- 模型不是只预测一个动作，而是预测整段 horizon 动作轨迹。
- 环境实际执行的是动作 chunk，也就是 `action_pred[:, n_obs_steps-1:n_obs_steps-1+n_action_steps]`。
- 输出动作必须反归一化后才能交给环境。

## 4. 训练调用链

### 4.1 外层脚本入口

README 示例：

```bash
bash scripts/train_policy.sh dp3 adroit_hammer 0112 0 0
```

真实脚本调用链：

```text
scripts/train_policy.sh
-> cd 3D-Diffusion-Policy
-> export HYDRA_FULL_ERROR=1
-> export CUDA_VISIBLE_DEVICES=${gpu_id}
-> python train.py --config-name=${config_name}.yaml ...
```

脚本参数含义：

| 位置 | 脚本变量 | 示例 | 含义 |
| --- | --- | --- | --- |
| `$1` | `alg_name` | `dp3` | 主配置名，对应 `dp3.yaml` |
| `$2` | `task_name` | `adroit_hammer` | Hydra task override |
| `$3` | `addition_info` | `0112` | 实验名后缀 |
| `$4` | `seed` | `0` | 随机种子 |
| `$5` | `gpu_id` | `0` | `CUDA_VISIBLE_DEVICES` |

### 4.2 `train.py` 函数级调用链

```text
3D-Diffusion-Policy/train.py
-> @hydra.main(config_path="diffusion_policy_3d/config")
-> main(cfg)
-> TrainDP3Workspace(cfg)
   -> set seed: torch/numpy/random
   -> hydra.utils.instantiate(cfg.policy)
      -> DP3.__init__() 或 SimpleDP3.__init__()
         -> DP3Encoder(...)
         -> ConditionalUnet1D(...)
         -> LowdimMaskGenerator(...)
         -> LinearNormalizer()
   -> optional deepcopy EMA model
   -> hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
-> workspace.run()
   -> optional load_checkpoint() if cfg.training.resume
   -> hydra.utils.instantiate(cfg.task.dataset)
      -> AdroitDataset/DexArtDataset/MetaworldDataset/RealDexDataset
      -> ReplayBuffer.copy_from_path()
      -> SequenceSampler(...)
   -> DataLoader(dataset, **cfg.dataloader)
   -> dataset.get_normalizer()
   -> dataset.get_validation_dataset()
   -> model.set_normalizer(normalizer)
   -> get_scheduler(...)
   -> optional EMAModel(...)
   -> hydra.utils.instantiate(cfg.task.env_runner, output_dir=...)
   -> wandb.init(...)
   -> TopKCheckpointManager(...)
   -> model.to(device), optimizer_to(...)
   -> for epoch:
      -> for batch in train_dataloader:
         -> dict_apply(batch, tensor.to(device))
         -> model.compute_loss(batch)
         -> loss.backward()
         -> optimizer.step()
         -> optimizer.zero_grad()
         -> lr_scheduler.step()
         -> ema.step(model)
         -> wandb.log(step_log)
      -> optional env_runner.run(policy)
      -> optional policy.predict_action(train_sampling_batch["obs"])
      -> optional save_checkpoint()
      -> wandb.log(epoch_log)
```

### 4.3 Loss、optimizer、scheduler、checkpoint、logging

| 环节 | 文件/函数 | 实现 |
| --- | --- | --- |
| loss | `policy/dp3.py:compute_loss()` | 扩散 MSE loss，返回 `loss` 和 `{"bc_loss": ...}` |
| optimizer | `train.py:TrainDP3Workspace.__init__()` | Hydra 实例化 `cfg.optimizer`，默认 `torch.optim.AdamW` |
| scheduler | `model/common/lr_scheduler.py:get_scheduler()` | 封装 diffusers scheduler，默认 `cosine` |
| backward | `train.py:run()` | `loss.backward()` 后按 `gradient_accumulate_every` step |
| EMA | `model/diffusion/ema_model.py:EMAModel.step()` | 指数滑动平均，训练默认开启 |
| checkpoint | `train.py:save_checkpoint()` | 保存 `cfg`、model/optimizer/EMA state_dict、`global_step`、`epoch` |
| Top-K | `common/checkpoint_util.py:TopKCheckpointManager` | 根据 `test_mean_score` 保留 top-k |
| logging | `train.py:wandb.init/log()` | 记录 loss、lr、rollout 指标、视频 |

注意：`train.py` 内部把 `RUN_VALIDATION = False` 写死，所以虽然构造了 `val_dataloader`，默认不会跑 validation loss。

## 5. 推理 / 评估调用链

### 5.1 外层脚本入口

README 示例：

```bash
bash scripts/eval_policy.sh dp3 adroit_hammer 0112 0 0
```

真实脚本意图：

```text
scripts/eval_policy.sh
-> cd 3D-Diffusion-Policy
-> export HYDRA_FULL_ERROR=1
-> export CUDA_VISIBLE_DEVICES=${gpu_id}
-> python eval.py --config-name=${config_name}.yaml task=${task_name} hydra.run.dir=${run_dir} ...
```

但当前 `scripts/eval_policy.sh` 没有定义 `wandb_mode` 和 `save_ckpt`，仍把 `logging.mode=${wandb_mode}`、`checkpoint.save_ckpt=${save_ckpt}` 传给 Hydra。普通 shell 不会报变量未定义，但会展开为空，可能导致 Hydra 解析空 override 出错。更稳妥的是直接运行 `eval.py` 并显式给这些 override。

### 5.2 `eval.py` 函数级调用链

```text
3D-Diffusion-Policy/eval.py
-> main(cfg)
-> TrainDP3Workspace(cfg)
-> workspace.eval()
   -> get_checkpoint_path(tag="latest")
   -> load_checkpoint(path=...)
   -> hydra.utils.instantiate(cfg.task.env_runner, output_dir=...)
   -> policy = ema_model if use_ema else model
   -> policy.eval()
   -> policy.cuda()
   -> env_runner.run(policy)
   -> print runner_log float metrics
```

### 5.3 Runner 内部调用链

以 MetaWorld 为例：

```text
env_runner/metaworld_runner.py:MetaworldRunner.run(policy)
-> obs = env.reset()
   -> MultiStepWrapper.reset()
   -> SimpleVideoRecordingWrapper.reset()
   -> MetaWorldEnv.reset()
      -> get_rgb()
      -> get_robot_state()
      -> get_point_cloud()
-> while not done:
   -> obs_dict = torch.from_numpy(obs).to(policy.device)
   -> obs_dict_input = {
        "point_cloud": obs_dict["point_cloud"].unsqueeze(0),
        "agent_pos": obs_dict["agent_pos"].unsqueeze(0)
      }
   -> policy.predict_action(obs_dict_input)
   -> action = action_dict["action"].cpu().numpy().squeeze(0)
   -> obs, reward, done, info = env.step(action)
      -> MultiStepWrapper.step(action_chunk)
      -> single env step for each action in chunk
-> mean_success_rates, mean_traj_rewards, test_mean_score
-> optional wandb.Video
```

Adroit 和 DexArt 的区别：

- `AdroitRunner` 使用 `AdroitEnv` 加 `MujocoPointcloudWrapperAdroit`，输入 `point_cloud` 和 `agent_pos`。
- `DexArtRunner` 使用 `DexArtEnv`，输入 `point_cloud`、`imagin_robot`、`agent_pos`。
- `MetaworldRunner` 使用 `MetaWorldEnv`，输入 `point_cloud` 和 `agent_pos`。

## 6. 数据处理调用链

### 6.1 仿真演示生成

| 脚本 | 调用外部文件 | 输出位置 | 说明 |
| --- | --- | --- | --- |
| `scripts/gen_demonstration_adroit.sh` | `third_party/sim/VRL3/src/gen_demonstration_expert.py` | `3D-Diffusion-Policy/data/` | 生成 Adroit 示范，默认 10 episodes |
| `scripts/gen_demonstration_dexart.sh` | `third_party/sim/dexart-release/examples/gen_demonstration_expert.py` | `3D-Diffusion-Policy/data/` | 生成 DexArt 示范，默认 100 episodes |
| `scripts/gen_demonstration_metaworld.sh` | `third_party/sim/Metaworld/gen_demonstration_expert.py` | `3D-Diffusion-Policy/data/` | 生成 MetaWorld 示范 |

这些脚本依赖 `third_party` 和外部资产/专家 checkpoint，本报告按要求没有深入 `third_party`。

### 6.2 真实机器人数据转换

对应文件：`scripts/convert_real_robot_data.py`。

真实数据原始格式按 README 描述为每个 episode 一个 dict：

- `point_cloud`: `(T, Np, 6)`，xyzrgb。
- `image`: `(T, H, W, 3)`。
- `depth`: `(T, H, W)`。
- `agent_pos`: `(T, Nd)`。
- `action`: `(T, Nd)`。

转换脚本流程：

```text
hard-coded expert_data_path
-> 找每个 episode 目录下的 data.pkl
-> 读取 point_cloud/image/depth/agent_pos/action
-> preprocess_point_cloud()
   -> xyz 乘固定 scale
   -> 使用固定 extrinsics_matrix 做坐标变换
   -> 按 WORK_SPACE 裁剪
   -> PyTorch3D FPS 下采样到 1024 点
-> preproces_image()
   -> resize 到 84x84
-> 写入 zarr:
   data/img
   data/point_cloud
   data/depth
   data/action
   data/state
   meta/episode_ends
```

限制：

- `expert_data_path` 和 `save_data_path` 是硬编码的 `/home/zhanggu/...`，不能直接当通用命令使用。
- 覆盖已有 zarr 时使用 `os.system('rm -rf ...')`，需要人工确认路径安全。
- 当前 `RealDexDataset` 读取 `state/action/point_cloud/img`，没有用 `depth`。

## 7. 机器人 observation / action / 坐标系解析

### 7.1 Observation 字段

| 字段 | 是否进入当前 DP3 模型 | 形状示例 | 来源文件 | 说明 |
| --- | --- | --- | --- | --- |
| `point_cloud` | 是 | Adroit `(512,3/6)`，MetaWorld `(512,3)`，DexArt/RealDex `(1024,3/6)` | `dataset/*.py`，`env/*/*wrapper.py` | 模型默认只取 xyz，除非 `use_pc_color=true` |
| `agent_pos` | 是 | Adroit 24，DexArt 33，MetaWorld 9，RealDex 22 | `config/task/*.yaml`，`dataset/*.py` | 由 zarr `state` 映射而来 |
| `imagin_robot` | DexArt 使用 | `(96,7)` | `dexart_dataset.py`，`dexart_wrapper.py` | `DP3Encoder` 会把其前几维对齐后拼到点云上 |
| `image` | 当前核心模型不使用 | `(3,84,84)` | env wrappers、zarr `img` | 配置和环境里存在，但 `DP3Encoder.forward()` 没读取 |
| `depth` | 当前核心模型不使用 | `(84,84)` | env wrappers、转换脚本 | 用于点云生成或保存，但 dataset 不送入模型 |
| `full_state` | 不进入当前策略 | MetaWorld raw obs | `metaworld_wrapper.py` | runner 没传给 policy |
| language/force/tactile | 未看到 | 无 | 无 | 仓库未提供 |

### 7.2 Action 表示

| 任务族 | action 维度 | 配置依据 | 执行方式 |
| --- | --- | --- | --- |
| Adroit hammer | 26 | `config/task/adroit_hammer.yaml` | 交给 Adroit env action space |
| Adroit door | 28 | `config/task/adroit_door.yaml` | 交给 Adroit env action space |
| Adroit pen | 24 | `config/task/adroit_pen.yaml` | 交给 Adroit env action space |
| DexArt | 22 | `config/task/dexart_*.yaml` | `DexArtEnv.step(action)` |
| MetaWorld | 4 | `config/task/metaworld_*.yaml` | `MetaWorldEnv.step(action)` |
| RealDex | 22 | `config/task/realdex_*.yaml` | 仓库未提供真实机器人部署接口 |

动作在模型中是轨迹：

```text
训练 batch action: (B, horizon, action_dim)
推理输出 action_pred: (B, horizon, action_dim)
实际执行 action: (B, n_action_steps, action_dim)
```

动作归一化：

- `dataset.get_normalizer()` 对 `action` 拟合 `LinearNormalizer`。
- `compute_loss()` 中用 `normalizer["action"].normalize()`。
- `predict_action()` 中用 `normalizer["action"].unnormalize()`。

README 对真实机器人动作的描述是：机械臂用 relative end-effector position control，灵巧手用 relative joint-angle position control。但当前仓库没有真实机器人控制代码，只有数据格式和训练入口。

### 7.3 坐标系和点云处理

| 逻辑 | 文件/函数 | 说明 |
| --- | --- | --- |
| MuJoCo depth 到点云 | `gym_util/mujoco_point_cloud.py:PointCloudGenerator.generateCroppedPointCloud()` | 用 camera fovy 构造内参，用 camera body pose 构造外参，Open3D 反投影 |
| 四元数到旋转矩阵 | `mujoco_point_cloud.py:quat2Mat()` | 生成相机位姿时使用 |
| Adroit 点云旋转 | `gym_util/mjpc_diffusion_wrapper.py:ADROIT_PC_TRANSFORM` | 固定绕 x 轴 45 度旋转 |
| Adroit 点云裁剪/平移 | `ENV_POINT_CLOUD_CONFIG` | 每个 Adroit 任务配置 min/max bound、scale、offset |
| MetaWorld 相机位置 | `env/metaworld/metaworld_wrapper.py` | 修改 `cam_pos[2]`，相机 `corner2` |
| MetaWorld 点云旋转 | `MetaWorldEnv.__init__()` 的 `pc_transform` | 用 x/y 角度构造旋转矩阵 |
| MetaWorld 点云裁剪 | `MetaWorldEnv.get_point_cloud()` | 按 `TASK_BOUDNS` 或 default bounds 裁剪 |
| FPS 下采样 | `mjpc_diffusion_wrapper.py:point_cloud_sampling()`、`dexart_wrapper.py:downsample_with_fps()` | 依赖 `pytorch3d.ops.sample_farthest_points` |
| RealDex 外参 | `scripts/convert_real_robot_data.py:preprocess_point_cloud()` | 使用硬编码 `extrinsics_matrix` 和 `WORK_SPACE` |
| IK/FK/controller/real robot interface | 仓库未提供明确实现 | README 只建议参考 iDP3 |

## 8. 核心类与函数解析

| 类 / 函数 | 文件路径 | 作用 | 输入 | 输出 | 被谁调用 | 初学者阅读重点 |
| --- | --- | --- | --- | --- | --- | --- |
| `TrainDP3Workspace` | `train.py` | 训练/评估工作区 | Hydra `cfg` | workspace 对象 | `main()` | 全流程总控 |
| `TrainDP3Workspace.run()` | `train.py` | 训练循环 | 无显式输入，使用 `self.cfg` | wandb 日志、checkpoint | `main()` | dataset/model/optimizer 如何串起来 |
| `TrainDP3Workspace.eval()` | `train.py` | 加载 latest checkpoint 并 rollout | `self.cfg` | 打印指标 | `eval.py` | 评估复用训练 workspace |
| `TrainDP3Workspace.save_checkpoint()` | `train.py` | 保存模型状态 | path/tag | `.ckpt` | `run()` | checkpoint 包含 normalizer |
| `BasePolicy.predict_action()` | `policy/base_policy.py` | 策略推理接口 | obs dict | action dict | env_runner | runner 只依赖此接口 |
| `DP3.__init__()` | `policy/dp3.py` | 构建编码器、U-Net、scheduler、normalizer | `shape_meta` 等配置 | `DP3` | Hydra | 维度从配置进入模型 |
| `DP3.predict_action()` | `policy/dp3.py` | 推理生成动作 | obs dict | action/action_pred | env_runner | 归一化、采样、反归一化 |
| `DP3.conditional_sample()` | `policy/dp3.py` | 从噪声迭代去噪 | condition data/mask/global_cond | trajectory | `predict_action()` | diffusion inference 主循环 |
| `DP3.compute_loss()` | `policy/dp3.py` | 扩散训练 loss | batch | loss/loss_dict | `TrainDP3Workspace.run()` | 加噪、target、MSE |
| `SimpleDP3` | `policy/simple_dp3.py` | 轻量版 DP3 | 同 DP3 | 同 DP3 | Hydra | 和 DP3 差在 U-Net |
| `DP3Encoder` | `model/vision/pointnet_extractor.py` | 点云和状态编码 | observation dict | feature tensor | policy | 当前真正用的模态 |
| `PointNetEncoderXYZ` | `model/vision/pointnet_extractor.py` | xyz 点云编码 | `(B,N,3)` | `(B,out_channels)` | `DP3Encoder` | MLP 加 max pooling |
| `ConditionalUnet1D` | `model/diffusion/conditional_unet1d.py` | 条件 1D U-Net | noisy trajectory, timestep, cond | trajectory prediction | policy | 动作序列主干网络 |
| `ConditionalResidualBlock1D` | `model/diffusion/conditional_unet1d.py` | FiLM/add/cross attention 条件 residual block | conv feature + cond | conv feature | U-Net | 条件如何调制网络 |
| `LowdimMaskGenerator` | `model/diffusion/mask_generator.py` | 生成 condition mask | trajectory shape | bool mask | policy | observation 是否作为 inpainting |
| `LinearNormalizer` | `model/common/normalizer.py` | 字典字段归一化 | data dict/tensor | normalized tensor | dataset/policy | 训练推理一致性 |
| `ReplayBuffer.copy_from_path()` | `common/replay_buffer.py` | 读取 zarr 数据 | zarr path、keys | ReplayBuffer | dataset | zarr 格式 |
| `SequenceSampler.sample_sequence()` | `common/sampler.py` | 从 episode 中取定长序列 | idx | key->sequence | dataset | horizon 和 padding |
| `AdroitDataset.__getitem__()` | `dataset/adroit_dataset.py` | 返回 Adroit batch sample | idx | dict tensor | DataLoader | state 改名 agent_pos |
| `DexArtDataset.get_normalizer()` | `dataset/dexart_dataset.py` | DexArt normalizer | 无 | LinearNormalizer | train.py | 点云 identity normalizer |
| `MultiStepWrapper.step()` | `gym_util/multistep_wrapper.py` | 执行动作 chunk | `(n_action_steps, action_dim)` | stacked obs/reward/done/info | env_runner | chunk action 如何落到单步 env |
| `PointCloudGenerator.generateCroppedPointCloud()` | `gym_util/mujoco_point_cloud.py` | MuJoCo depth 生成点云 | camera render | point cloud/depth | env wrappers | 相机内外参 |
| `MujocoPointcloudWrapperAdroit.get_point_cloud()` | `gym_util/mjpc_diffusion_wrapper.py` | Adroit 点云变换裁剪采样 | 无 | point_cloud/depth | Adroit runner env | 坐标变换和 FPS |
| `MetaWorldEnv.get_point_cloud()` | `env/metaworld/metaworld_wrapper.py` | MetaWorld 点云生成 | 无 | point_cloud/depth | MetaWorldEnv step/reset | 裁剪和 transform |
| `DexArtEnv.step()` | `env/dexart/dexart_wrapper.py` | DexArt 单步交互 | action | obs/reward/done/info | MultiStepWrapper | obs 字段来源 |
| `TopKCheckpointManager.get_ckpt_path()` | `common/checkpoint_util.py` | 选择是否保存 top-k checkpoint | metric dict | path 或 None | train.py | monitor_key 是 `test_mean_score` |

核心训练 step 伪代码：

```python
for batch in train_dataloader:
    batch = to_device(batch)
    raw_loss, loss_dict = model.compute_loss(batch)
    loss = raw_loss / gradient_accumulate_every
    loss.backward()
    if should_step:
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
    if use_ema:
        ema.step(model)
    wandb.log(loss_dict)
```

核心评估 step 伪代码：

```python
obs = env.reset()
while not done:
    obs_dict = torch_from_numpy(obs)
    action_dict = policy.predict_action(obs_dict_input)
    action_chunk = action_dict["action"].cpu().numpy()[0]
    obs, reward, done, info = env.step(action_chunk)
```

## 9. 配置系统解析

### 9.1 Hydra 配置入口

`train.py` 和 `eval.py` 都使用：

```python
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        "diffusion_policy_3d", "config"))
)
```

因此配置根目录是：

```text
3D-Diffusion-Policy/diffusion_policy_3d/config/
```

主配置：

- `dp3.yaml`
- `simple_dp3.yaml`

任务配置：

- `task/*.yaml`

命令行覆盖示例：

```bash
cd 3D-Diffusion-Policy
python train.py --config-name=dp3.yaml task=adroit_hammer training.seed=0 training.device=cuda:0
```

### 9.2 主配置关键项

| 配置项 | 所在文件 | 含义 | 影响范围 | 推荐初学者是否修改 |
| --- | --- | --- | --- | --- |
| `defaults: - task: adroit_hammer` | `dp3.yaml`、`simple_dp3.yaml` | 默认任务 | dataset/env/action 维度 | 可以，用命令行 `task=...` |
| `horizon: 16` | 主配置 | 扩散预测轨迹长度 | dataset 采样、U-Net 输入输出 | 谨慎修改 |
| `n_obs_steps: 2` | 主配置 | 输入 observation 时间步数 | dataset padding、encoder condition | 可小范围尝试 |
| `n_action_steps: 8` | 主配置 | 每次执行动作 chunk 长度 | env runner、MultiStepWrapper | 可小范围尝试 |
| `policy._target_` | 主配置 | 策略类 | 模型实现 | 在 `dp3`/`simple_dp3` 间切换 |
| `policy.down_dims` | 主配置 | U-Net 通道宽度 | 模型容量、显存 | 初学者先不要改 |
| `policy.encoder_output_dim` | 主配置 | PointNet 输出维度 | condition 维度 | 谨慎修改 |
| `policy.noise_scheduler` | 主配置 | 扩散 scheduler | 训练目标和采样 | 先不要改 |
| `policy.num_inference_steps` | 主配置 | 推理去噪步数 | 推理速度/质量 | 可以尝试 |
| `policy.use_pc_color` | 主配置 | 是否使用 RGB 点云通道 | PointNet 输入维度 | 改前确认数据是 `(N,6)` |
| `dataloader.batch_size` | 主配置 | batch size | 显存和训练速度 | 可以改 |
| `optimizer.lr` | 主配置 | 学习率 | 训练稳定性 | 可以小范围改 |
| `training.num_epochs` | 主配置 | 训练 epoch | 训练时长 | 可以改 |
| `training.use_ema` | 主配置 | 是否用 EMA 模型 | 评估策略 | 建议保留 |
| `training.rollout_every` | 主配置 | rollout 频率 | 训练耗时/评估频率 | 可改大来省时间 |
| `checkpoint.save_ckpt` | 主配置 | 是否保存 ckpt | 输出文件 | 脚本里默认覆盖为 true |
| `logging.mode` | 主配置 | wandb online/offline | 日志 | 调试建议 offline |
| `hydra.run.dir` | 主配置/脚本 | 输出目录 | checkpoint/log 存放 | 需要明确 |

### 9.3 任务配置关键项

| 配置项 | 示例文件 | 含义 | 影响范围 | 推荐初学者是否修改 |
| --- | --- | --- | --- | --- |
| `shape_meta.obs.point_cloud.shape` | `adroit_hammer.yaml` | 点云点数和通道 | `DP3Encoder` 输入 | 改数据格式时必须同步 |
| `shape_meta.obs.agent_pos.shape` | 各 task yaml | proprioception 维度 | state MLP 输入 | 改机器人状态时必须同步 |
| `shape_meta.obs.imagin_robot.shape` | `dexart_bucket.yaml` | DexArt imagined robot 点云 | `DP3Encoder` 拼接点云 | 仅 DexArt |
| `shape_meta.action.shape` | 各 task yaml | action 维度 | U-Net input_dim、env action | 改 action space 时必须同步 |
| `env_runner._target_` | 各仿真 task yaml | rollout 类 | 评估环境 | 新环境要新增 |
| `dataset._target_` | 各 task yaml | dataset 类 | 训练数据读取 | 新数据集要新增 |
| `dataset.zarr_path` | 各 task yaml | zarr 数据路径 | ReplayBuffer | 最常修改 |
| `dataset.max_train_episodes` | 各 task yaml | 限制训练 episode 数 | 数据量 | 可用于小样本调试 |
| `env_runner.use_point_crop` | Adroit/MetaWorld | 是否裁剪点云 | observation 质量 | 可调试 |

### 9.4 当前配置不清晰点

- `README.md` 的自定义任务步骤写的是 `diffusion_policy_3d/configs/task`，实际目录是 `diffusion_policy_3d/config/task`。
- `config/task/realdex_drill.yaml` 中 `agent_pos.type` 写成 `low_dimx`，其他 RealDex 任务和模型语义应为 `low_dim`。当前 `DP3Encoder` 实际只用 shape，不读 type，所以训练可能不直接报错，但这是配置质量问题。
- `dexart_bucket.yaml` 中配置了 `image`，但 `DexArtDataset._sample_to_data()` 没输出 `image`，`DP3Encoder.forward()` 也不读取 `image`。

## 10. 从零复现流程

### 10.1 环境安装

仓库提供的安装文档在 `INSTALL.md`。文档核心命令是：

```bash
conda create -n dp3 python=3.8
conda activate dp3
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
cd 3D-Diffusion-Policy
pip install -e .
cd ..
```

然后安装 MuJoCo、mujoco-py、`third_party` 下 sim env、简化 PyTorch3D 和必要 Python 包。注意：`INSTALL.md` 开头说明它主要适配 3090/A40/A800/A100、CUDA 11.7、driver 515.65.01 一类环境。现代 GPU 或 CUDA 版本可能需要调整 PyTorch、PyTorch3D、mujoco_py 编译参数。

### 10.2 数据准备

仿真数据可以用脚本生成：

```bash
bash scripts/gen_demonstration_adroit.sh hammer
bash scripts/gen_demonstration_dexart.sh bucket
bash scripts/gen_demonstration_metaworld.sh basketball
```

这些命令要求：

- `third_party/sim/VRL3` 下有 Adroit expert ckpt。
- `third_party/sim/dexart-release/assets` 和 RL checkpoint 已准备。
- `third_party/sim/Metaworld` 安装正常。

数据最终应放在 task yaml 指定的路径，例如：

```text
3D-Diffusion-Policy/data/adroit_hammer_expert.zarr
3D-Diffusion-Policy/data/dexart_bucket_expert.zarr
3D-Diffusion-Policy/data/metaworld_assembly_expert.zarr
3D-Diffusion-Policy/data/realdex_drill.zarr
```

### 10.3 训练命令

推荐先用脚本：

```bash
bash scripts/train_policy.sh dp3 adroit_hammer 0112 0 0
```

等价核心命令：

```bash
cd 3D-Diffusion-Policy
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python train.py --config-name=dp3.yaml \
  task=adroit_hammer \
  hydra.run.dir=data/outputs/adroit_hammer-dp3-0112_seed0 \
  training.debug=False \
  training.seed=0 \
  training.device=cuda:0 \
  exp_name=adroit_hammer-dp3-0112 \
  logging.mode=online \
  checkpoint.save_ckpt=True
```

小范围调试可以显式覆盖：

```bash
cd 3D-Diffusion-Policy
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python train.py --config-name=simple_dp3.yaml \
  task=adroit_hammer \
  training.debug=True \
  training.device=cuda:0 \
  logging.mode=offline \
  checkpoint.save_ckpt=False
```

这不会避免读取数据和构造环境，但会把训练步数、rollout/checkpoint 频率改成 debug 设置。

### 10.4 评估命令

如果训练输出目录中存在：

```text
data/outputs/adroit_hammer-dp3-0112_seed0/checkpoints/latest.ckpt
```

可以使用直接命令：

```bash
cd 3D-Diffusion-Policy
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python eval.py --config-name=dp3.yaml \
  task=adroit_hammer \
  hydra.run.dir=data/outputs/adroit_hammer-dp3-0112_seed0 \
  training.seed=0 \
  training.device=cuda:0 \
  exp_name=adroit_hammer-dp3-0112 \
  logging.mode=offline \
  checkpoint.save_ckpt=False
```

`scripts/eval_policy.sh` 的意图相同，但当前脚本变量不完整，建议先用上面的显式命令。

### 10.5 可视化

点云可视化工具：

```bash
cd visualizer
pip install -e .
```

Python 示例：

```python
import visualizer
visualizer.visualize_pointcloud(pointcloud)
```

真实函数在 `visualizer/visualizer/pointcloud.py`，支持 `(N,3)` 或 `(N,6)` 点云。

### 10.6 常见报错

| 报错/现象 | 文档或源码依据 | 处理思路 |
| --- | --- | --- |
| OpenGL 初始化失败 | `ERROR_CATCH.md` | 尝试 `unset LD_PRELOAD` |
| `distutils` 缺 `version` | `ERROR_CATCH.md` | `pip install setuptools==59.5.0` |
| mujoco-py 编译 Cython 报错 | `ERROR_CATCH.md` | `pip install Cython==0.29.35` |
| 缺 `GL/glew.h` | `ERROR_CATCH.md` | 需要系统 `libglew-dev` |
| PyTorch3D CUDA kernel 不匹配 | `ERROR_CATCH.md` | 重新按当前 CUDA/PyTorch 编译 PyTorch3D |
| `cached_download` import 失败 | `ERROR_CATCH.md` | 降级 `huggingface_hub==0.25.2` |
| `eval_policy.sh` 参数为空 | 当前脚本 | 用显式 `python eval.py ... logging.mode=offline checkpoint.save_ckpt=False` |

### 10.7 如何确认跑通

轻量确认：

```bash
cd 3D-Diffusion-Policy
python train.py --help
python eval.py --help
```

数据确认：

```bash
python - <<'PY'
import zarr
root = zarr.open('data/adroit_hammer_expert.zarr', 'r')
print(root.tree())
print(root['meta/episode_ends'][:5])
PY
```

训练正常的信号：

- `wandb` 或 offline log 中 `train_loss` 能持续记录。
- `bc_loss` 没有 NaN。
- `train_action_mse_error` 在采样 batch 上有合理数值。
- 如果 `checkpoint.save_ckpt=True`，输出目录出现 `checkpoints/latest.ckpt`。

推理可信的信号：

- `eval.py` 能加载 `latest.ckpt`。
- runner 输出 `test_mean_score`。
- 视频中动作不是完全静止或明显发散。
- 对同一 checkpoint 多次 rollout 的成功率波动在可接受范围内。

## 11. 初学者阅读路线

### 第一阶段：只理解整体

1. `README.md`
2. `INSTALL.md`
3. `scripts/train_policy.sh`
4. `3D-Diffusion-Policy/train.py`
5. `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml`
6. `3D-Diffusion-Policy/diffusion_policy_3d/config/task/adroit_hammer.yaml`

### 第二阶段：理解数据流

1. `diffusion_policy_3d/dataset/base_dataset.py`
2. `diffusion_policy_3d/dataset/adroit_dataset.py`
3. `diffusion_policy_3d/common/replay_buffer.py`
4. `diffusion_policy_3d/common/sampler.py`
5. `diffusion_policy_3d/model/common/normalizer.py`

### 第三阶段：理解模型

1. `diffusion_policy_3d/policy/base_policy.py`
2. `diffusion_policy_3d/policy/dp3.py`
3. `diffusion_policy_3d/model/vision/pointnet_extractor.py`
4. `diffusion_policy_3d/model/diffusion/conditional_unet1d.py`
5. `diffusion_policy_3d/model/diffusion/mask_generator.py`

### 第四阶段：理解训练

1. `3D-Diffusion-Policy/train.py:TrainDP3Workspace.run()`
2. `diffusion_policy_3d/model/common/lr_scheduler.py`
3. `diffusion_policy_3d/model/diffusion/ema_model.py`
4. `diffusion_policy_3d/common/checkpoint_util.py`
5. `diffusion_policy_3d/common/logger_util.py`

### 第五阶段：理解推理和部署

1. `3D-Diffusion-Policy/eval.py`
2. `3D-Diffusion-Policy/train.py:TrainDP3Workspace.eval()`
3. `diffusion_policy_3d/env_runner/metaworld_runner.py`
4. `diffusion_policy_3d/gym_util/multistep_wrapper.py`
5. `diffusion_policy_3d/policy/dp3.py:predict_action()`

### 第六阶段：尝试修改

适合初学者的小任务：

| 修改任务 | 修改位置 | 风险 |
| --- | --- | --- |
| 改 batch size | `dp3.yaml` 的 `dataloader.batch_size` | 低 |
| 改 wandb offline | 命令行 `logging.mode=offline` | 低 |
| 改推理步数 | `policy.num_inference_steps` | 中低 |
| 改训练 episode 数 | task yaml 的 `dataset.max_train_episodes` | 中低 |
| 打印 batch shape | `train.py` 中进入 `compute_loss()` 前 | 低 |
| 可视化 point cloud | dataset 取样后调用 `visualizer.visualize_pointcloud()` | 低 |
| 替换 PointNet 容量 | `pointnet_extractor.py` 或 `pointcloud_encoder_cfg` | 中 |
| 添加新 proprioception | task yaml、dataset `_sample_to_data()`、zarr 字段 | 中 |
| 修改 action 维度 | task yaml、dataset 数据、env action space、policy checkpoint | 高 |
| 添加新环境 | `env/`、`env_runner/`、`dataset/`、`config/task/` | 高 |

## 12. 适合后续修改的位置

### 12.1 复现和调参

- `diffusion_policy_3d/config/dp3.yaml`
- `diffusion_policy_3d/config/simple_dp3.yaml`
- `diffusion_policy_3d/config/task/*.yaml`
- `scripts/train_policy.sh`

最常改：

```text
horizon
n_obs_steps
n_action_steps
dataloader.batch_size
optimizer.lr
training.num_epochs
training.rollout_every
policy.num_inference_steps
dataset.zarr_path
```

### 12.2 换数据集

需要改：

- 新增或修改 `diffusion_policy_3d/dataset/*.py`。
- 确保 `__getitem__()` 返回 `{"obs": ..., "action": ...}`。
- 确保 `get_normalizer()` 覆盖所有进入 policy 的 key。
- 新增 `diffusion_policy_3d/config/task/your_task.yaml`。

### 12.3 换 observation

如果添加新的 proprioception 字段：

- zarr 写入新字段。
- dataset `_sample_to_data()` 放入 `obs`。
- `shape_meta.obs` 增加字段。
- `DP3Encoder` 增加处理逻辑。

如果想真正使用 RGB/depth：

- dataset 需要把 `img` 或 `depth` 放入 `obs`。
- `DP3Encoder` 需要新增图像/depth encoder。
- normalizer 要覆盖新字段，或设计图像预处理。
- policy 的 `obs_feature_dim` 会变化，checkpoint 不兼容。

### 12.4 换 action space

需要同时修改：

- zarr `data/action`。
- task yaml 的 `shape_meta.action.shape`。
- 环境 `action_space` 和 `step(action)`。
- 真实机器人部署时的 action 反归一化和控制器映射。

### 12.5 换 backbone

可从这里开始：

- 点云 backbone：`model/vision/pointnet_extractor.py`。
- 动作扩散 backbone：`model/diffusion/conditional_unet1d.py` 或 `simple_conditional_unet1d.py`。
- 策略装配：`policy/dp3.py`。

## 13. 当前仓库可能存在的问题或不清晰点

1. `README.md` 自定义任务步骤写的是 `diffusion_policy_3d/configs/task`，当前真实目录是 `diffusion_policy_3d/config/task`。
2. `scripts/eval_policy.sh` 使用 `wandb_mode` 和 `save_ckpt`，但脚本内没有定义。建议直接显式运行 `python eval.py ... logging.mode=offline checkpoint.save_ckpt=False`。
3. `config/task/realdex_drill.yaml` 的 `agent_pos.type` 是 `low_dimx`，疑似 typo。
4. 多个 task yaml 或 env wrapper 中有 `image`、`depth`，但当前 `DP3Encoder` 没有读取 RGB/depth，核心训练实际是点云加 `agent_pos`。
5. `train.py` 构造了 validation dataset 和 dataloader，但 `RUN_VALIDATION = False` 写死，默认不跑 validation loss。
6. `scripts/convert_real_robot_data.py` 是硬编码路径示例，不是可参数化数据转换工具。
7. `scripts/convert_real_robot_data.py` 覆盖已有 zarr 时用 `os.system('rm -rf ...')`，改路径时必须非常谨慎。
8. `README.md` 说评估脚本用于 deployment/inference，benchmark 以训练期间 wandb rollout 为准；这和代码一致，因为 `train.py` 会周期性调用 `env_runner.run(policy)`。
9. 仓库没有真实机器人部署接口、IK/FK/controller 实现。README 也提示可参考 iDP3。
10. `mujoco_point_cloud.py` 中 `combined_cloud_colors = color_img.reshape(-1, 3)` 只使用最后一个相机循环的 color image。如果多相机启用，颜色和点数可能需要重新检查。

## 14. 总结

这个仓库的主线很清楚：它把机器人模仿学习中的动作轨迹作为扩散模型的生成对象，把点云和机器人状态编码成条件，训练一个条件 1D U-Net 来预测动作轨迹。初学者最应该抓住三条线：

1. 数据线：`ReplayBuffer`、`SequenceSampler`、`Dataset.__getitem__()` 如何把 zarr 变成 `obs/action`。
2. 模型线：`DP3Encoder` 如何把 `point_cloud/agent_pos` 变成条件，`ConditionalUnet1D` 如何对动作轨迹去噪。
3. 执行线：`env_runner` 如何调用 `policy.predict_action()`，再用 `MultiStepWrapper` 执行动作 chunk。

最核心的 5 个文件：

1. `3D-Diffusion-Policy/train.py`
2. `3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py`
3. `3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py`
4. `3D-Diffusion-Policy/diffusion_policy_3d/model/diffusion/conditional_unet1d.py`
5. `3D-Diffusion-Policy/diffusion_policy_3d/common/sampler.py`

下一步最应该读的 5 个文件：

1. `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml`
2. `3D-Diffusion-Policy/diffusion_policy_3d/config/task/adroit_hammer.yaml`
3. `3D-Diffusion-Policy/diffusion_policy_3d/dataset/adroit_dataset.py`
4. `3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py`
5. `3D-Diffusion-Policy/diffusion_policy_3d/env_runner/adroit_runner.py`
