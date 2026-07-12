# DP3 训练与推理参数分类

Flexiv 双臂流程使用两份配置：

- `dp3_train_config.yaml`：训练入口读取。
- `dp3_inference_config.yaml`：机器人推理入口默认读取。

两份配置都通过 `algorithm: simple_dp3 | dp3` 选择官方 `SimpleDP3` 或 `DP3`
实现。训练和推理脚本都不需要位置参数，分别直接使用对应 YAML 中的值。

## 必须训练推理一致

以下字段决定模型结构、输入输出契约或扩散语义。推理入口会在连接机器人前，将推理
YAML 与 checkpoint 内保存的 Hydra 训练配置逐项比较。

| 参数 | 当前 SimpleDP3 | DP3 profile | 作用 |
|---|---:|---:|---|
| `policy._target_` | `SimpleDP3` | `DP3` | 实例化策略类型 |
| `horizon` | 8 | 16 | 扩散预测的完整动作轨迹长度 |
| `n_obs_steps` | 2 | 2 | 参与条件编码的历史观测帧数 |
| `obs_as_global_cond` | `true` | `true` | 点云和状态特征作为全局条件 |
| `condition_type` | `film` | `film` | U-Net 条件注入方式 |
| `down_dims` | `[512,1024,2048]` | `[512,1024,2048]` | U-Net 通道规模；当前 SimpleDP3 对齐旧 DP 容量 |
| `diffusion_step_embed_dim` | 128 | 128 | 扩散 timestep embedding 维度 |
| `kernel_size` / `n_groups` | 5 / 8 | 5 / 8 | U-Net 卷积和 GroupNorm 参数 |
| `encoder_output_dim` | 64 | 64 | DP3 点云/状态编码输出维度 |
| `pointcloud_encoder_cfg` | XYZ, PointNet | XYZ, PointNet | 点云编码器输入通道与归一化 |
| `noise_scheduler` | DDPM | DDPM | checkpoint 保存的正向扩散训练语义 |
| `num_train_timesteps` | 100 | 100 | 训练噪声时间步总数 |
| `prediction_type` | `epsilon` | `epsilon` | 当前配置预测扩散噪声 |
| `shape_meta` | PC `[1024,3]`, state `[28]`, action `[14]` | 同配置决定 | 模型输入输出 shape |

`horizon`、网络通道、点云通道、scheduler 训练语义等字段不能只改推理 YAML。
修改后必须使用按相同参数训练的 checkpoint。

`policy.use_point_crop`、`policy.crop_shape` 和
`policy.pointcloud_encoder_cfg.normal_channel` 也不属于当前模型权重契约：前者在官方
仿真任务中传给环境 wrapper，当前 Flexiv 实数流程的裁剪由 `pointcloud.config` 控制；
后两者由当前点云 encoder 接收但不参与网络构造或 forward。

## 训练专用参数

以下参数只由 `dp3_train_config.yaml` 管理：

| 分类 | 参数 |
|---|---|
| 数据 | `task`、zarr 路径、episode 上限、train/val 划分 |
| DataLoader | batch size、worker、shuffle、pin memory |
| 优化器 | AdamW lr、betas、eps、weight decay |
| 学习率 | scheduler 名称、warmup steps |
| 训练循环 | epoch、seed、debug、resume、梯度累积、采样/验证频率 |
| EMA 更新 | `ema.update_after_step`、`power`、`max_value` 等 |
| 产物 | Hydra run dir、WandB、checkpoint 保存频率和命名 |

训练入口：

```bash
conda activate dp3
bash scripts/train_flexiv_dual_arm_dp3.sh
```

启动前在 `dp3_train_config.yaml` 中设置 task、`task.dataset.zarr_path`、
`launcher.gpu_id`、algorithm、seed、run dir、WandB 和训练超参数。

## 推理专用参数

以下参数只由 `dp3_inference_config.yaml` 管理：

| 参数 | 当前默认 | 作用 |
|---|---:|---|
| `checkpoint.path` | 当前 SimpleDP3 `latest.ckpt` | 部署权重 |
| `inference.gpu_id` | 0 | 物理 GPU，通过 launcher 设置 `CUDA_VISIBLE_DEVICES` |
| `inference.device` | `cuda:0` | 进程内策略设备 |
| `pointcloud.device` | `cuda:0` | PointCloudBuilder 设备 |
| `duration_seconds` | `null` | 默认持续运行到 `Ctrl+C` 或停止文件；正数用于限时测试 |
| `rate_hz` | 30 | 与训练数据一致的动作队列执行频率 |
| `action_mode` | `chunk` | 每次推理后依次执行配置的 action chunk |
| `policy.n_action_steps` | 4 | 从完整预测轨迹中执行的动作数；合法范围为 `1..(horizon-n_obs_steps+1)` |
| `use_ema` | `true` | 从 checkpoint 选择 EMA 或 raw 权重；选择 EMA 时 checkpoint 必须包含 EMA 权重 |
| `scheduler` | `ddim` | 从 checkpoint 的 DDPM beta schedule 构造部署采样器 |
| `policy.noise_scheduler.clip_sample` | `true` | 推理采样输出裁剪；不参与训练 loss，可独立设置 |
| `num_inference_steps` | 10 | DDIM 反向去噪次数，影响延迟与采样质量 |
| `policy_warmup_steps` | 2 | 连接机器人前执行零输入推理，消除 CUDA 冷启动延迟 |
| `robot.*_on_connect` | 逐项配置 | 独立控制使能、清故障、Home、工具、夹爪和笛卡尔模式准备 |
| `robot.use_cartesian_servo_thread` | `true` | 启用 200 Hz Flexiv 目标平滑线程 |
| `low_speed_scale` | 1 | 对末端位姿增量整体缩放 |
| `max_cartesian_delta` | 0.02 | 单步 xyz 增量范数上限 |
| `max_rotation_delta` | 0.04 | 单步旋转向量范数上限 |
| watchdogs | 见 YAML | 限制帧年龄、动作年龄、推理/发送/循环延迟 |
| `visualization.*` | 2 Hz | 独立 Open3D 进程的显示参数 |

`scheduler` 和 `num_inference_steps` 不参与训练 loss，因此只放在推理 YAML。当前
epsilon 模型不能使用 10-step DDPM；部署时显式使用 DDIM 10-step，在 batch=1 上热
启动后约 39--40 ms。双臂与双夹爪动作都直接来自模型，不做任务特定覆盖。

当前训练 checkpoint 的 `n_action_steps=7`，推理配置的 `n_action_steps=4`。这是允许的：
DP3 loss 学习完整 `horizon=8` 轨迹，`predict_action()` 才从 `n_obs_steps-1` 开始切出
执行段。当前最大合法值为 `8-2+1=7`；推理会依次发送配置的 4-step 动作队列，队列
耗尽后再基于最新真机观测重新推理。这与 Le-nero 的 DiffusionPolicy queue 和官方
`MultiStepWrapper` 的 chunk 执行语义一致。

## 一行推理

先在 `dp3_inference_config.yaml` 中设置 checkpoint 和机器人配置；默认持续运行到手动停止：

```bash
conda run -n dp3 bash scripts/run_flexiv_dual_arm_dp3_inference.sh
```

该命令直接进入会调用 `robot.send_action()` 的 `inference` 模式，不需要前置无动作日志
或额外 ACK。可选的 `--check-config` 只用于改配置后的无硬件检查，不是正常部署步骤。
