# DP3 训练与推理参数分类

Flexiv 双臂流程使用两份配置：

- `dp3_train_config.yaml`：训练入口读取。
- `dp3_inference_config.yaml`：机器人推理入口默认读取。

两份配置都通过 `algorithm: simple_dp3 | dp3` 选择官方 `SimpleDP3` 或 `DP3`
实现。训练脚本第三个位置参数会覆盖训练 YAML 的 `algorithm`；推理不需要终端参数，
直接使用推理 YAML 中的值。

## 必须训练推理一致

以下字段决定模型结构、输入输出契约或扩散语义。推理入口会在连接机器人前，将推理
YAML 与 checkpoint 内保存的 Hydra 训练配置逐项比较。

| 参数 | 当前 SimpleDP3 | DP3 profile | 作用 |
|---|---:|---:|---|
| `policy._target_` | `SimpleDP3` | `DP3` | 实例化策略类型 |
| `horizon` | 4 | 16 | 扩散预测的完整动作轨迹长度 |
| `n_obs_steps` | 2 | 2 | 参与条件编码的历史观测帧数 |
| `n_action_steps` | 3 | 8 | 从预测轨迹中切出的可执行动作数 |
| `obs_as_global_cond` | `true` | `true` | 点云和状态特征作为全局条件 |
| `condition_type` | `film` | `film` | U-Net 条件注入方式 |
| `down_dims` | `[128,256,384]` | `[512,1024,2048]` | U-Net 通道规模 |
| `diffusion_step_embed_dim` | 128 | 128 | 扩散 timestep embedding 维度 |
| `kernel_size` / `n_groups` | 5 / 8 | 5 / 8 | U-Net 卷积和 GroupNorm 参数 |
| `encoder_output_dim` | 64 | 64 | DP3 点云/状态编码输出维度 |
| `pointcloud_encoder_cfg` | XYZ, PointNet | XYZ, PointNet | 点云编码器输入通道与归一化 |
| `noise_scheduler` | DDIM | DDIM | 正向扩散训练语义和反向采样器 |
| `num_train_timesteps` | 100 | 100 | 训练噪声时间步总数 |
| `prediction_type` | `sample` | `sample` | 网络预测干净动作样本 |
| `shape_meta` | PC `[1024,3]`, state `[28]`, action `[14]` | 同配置决定 | 模型输入输出 shape |
| `use_ema` | `true` | `true` | 推理使用 EMA 还是原始模型权重 |

`horizon`、网络通道、点云通道、scheduler 训练语义等字段不能只改推理 YAML。
修改后必须使用按相同参数训练的 checkpoint。

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
conda run -n dp3 bash scripts/train_flexiv_dual_arm_dp3.sh \
  xyz \
  /path/to/dataset.zarr \
  simple_dp3 \
  0 \
  42
```

## 推理专用参数

以下参数只由 `dp3_inference_config.yaml` 管理：

| 参数 | 当前默认 | 作用 |
|---|---:|---|
| `checkpoint.path` | 当前 SimpleDP3 `latest.ckpt` | 部署权重 |
| `inference.gpu_id` | 0 | 物理 GPU，通过 launcher 设置 `CUDA_VISIBLE_DEVICES` |
| `inference.device` | `cuda:0` | mask 后进程内策略设备 |
| `pointcloud.device` | `cuda:0` | PointCloudBuilder 设备 |
| `duration_seconds` | `null` | 默认持续运行到 `Ctrl+C` 或停止文件；正数用于限时测试 |
| `rate_hz` | 5 | 外层策略/控制循环目标频率 |
| `action_mode` | `receding` | 每帧重算或执行完整 action chunk |
| `num_inference_steps` | 10 | DDIM 反向去噪次数，影响延迟与采样质量 |
| `low_speed_scale` | 0.1 | 对末端位姿增量整体缩放 |
| `max_cartesian_delta` | 0.01 | 单步 xyz 增量范数上限 |
| `max_rotation_delta` | 0.02 | 单步旋转向量范数上限 |
| watchdogs | 见 YAML | 限制帧年龄、动作年龄、推理/发送/循环延迟 |
| `visualization.*` | 2 Hz | 独立 Open3D 进程的显示参数 |

`num_inference_steps` 不参与训练 loss，因此只放在推理 YAML。减小它通常降低延迟但
可能降低动作采样质量；增大它则相反。

`action_mode=receding` 时，模型仍按 `n_action_steps` 返回动作，但运行时只发送第 0
个动作并在下一帧重新推理。`chunk` 会依次发送完整动作队列。

## 一行推理

先在 `dp3_inference_config.yaml` 中设置 checkpoint 和机器人配置；默认持续运行到手动停止：

```bash
conda run -n dp3 bash scripts/run_flexiv_dual_arm_dp3_inference.sh
```

该命令直接进入会调用 `robot.send_action()` 的 `inference` 模式，不需要前置无动作日志
或额外 ACK。可选的 `--check-config` 只用于改配置后的无硬件检查，不是正常部署步骤。
