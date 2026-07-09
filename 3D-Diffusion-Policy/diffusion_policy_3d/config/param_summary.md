# dp3.yaml 与 simple_dp3.yaml 参数说明

本文基于当前仓库中的 `diffusion_policy_3d/config/dp3.yaml`、`diffusion_policy_3d/config/simple_dp3.yaml` 以及实际代码路径整理。关键代码路径包括：

- `train.py`：Hydra 配置实例化、训练循环、EMA、学习率调度、checkpoint。
- `diffusion_policy_3d/policy/dp3.py` 与 `diffusion_policy_3d/policy/simple_dp3.py`：policy 参数如何进入模型、采样和 loss。
- `diffusion_policy_3d/model/diffusion/conditional_unet1d.py` 与 `simple_conditional_unet1d.py`：DP3 与 SimpleDP3 的 U-Net 结构差异。
- `diffusion_policy_3d/model/vision/pointnet_extractor.py`：点云和低维状态编码。
- `diffusion_policy_3d/config/task/real/flexiv_dual_arm_head_xyz*.yaml`：Flexiv 双臂真实数据的 `shape_meta` 与 dataset 约束。

## 总体区别

| 维度 | `dp3.yaml` | `simple_dp3.yaml` | 作用与影响 |
|---|---:|---:|---|
| policy 类 | `diffusion_policy_3d.policy.dp3.DP3` | `diffusion_policy_3d.policy.simple_dp3.SimpleDP3` | 选择不同 policy 和 U-Net 实现。`DP3` 使用更完整的 `conditional_unet1d.py`，每个 down/mid/up 阶段残差块更多；`SimpleDP3` 使用更轻的 `simple_conditional_unet1d.py`，速度和显存更友好。 |
| 默认时间窗口 `horizon` | `16` | `4` | 每个训练样本预测的动作序列长度。越大越能覆盖长动作，但显存、计算和采样难度都会上升。 |
| 默认执行步数 `n_action_steps` | `8` | `3` | 每次推理从预测序列中取多少步动作执行。`simple_dp3` 的 `horizon=4,n_obs_steps=2,n_action_steps=3` 已经卡在最大可取值。 |
| U-Net 通道 `down_dims` | `[512,1024,2048]` | `[128,256,384]` | 主要容量差异。`dp3` 更大、更慢、更吃显存；`simple_dp3` 适合 smoke test、快速迭代和小数据集。 |
| checkpoint 默认 | `save_ckpt: False` | `save_ckpt: True` | `dp3.yaml` 直接跑默认不保存周期 checkpoint；Flexiv 训练脚本会用环境变量 `SAVE_CKPT=True` 覆盖。 |

## 主配置与时间窗口

| 参数 | `dp3.yaml` 默认值 | `simple_dp3.yaml` 默认值 | 含义/作用 | 调参建议 |
|---|---:|---:|---|---|
| `defaults.task` | `sim/adroit_hammer` | `sim/adroit_hammer` | Hydra 默认任务配置。实际 Flexiv 训练脚本会覆盖为 `real/flexiv_dual_arm_head_xyz` 或 `real/flexiv_dual_arm_head_xyzrgb`。 | 真实数据训练时不要依赖默认值，用脚本或命令行显式指定 task。 |
| `name` | `train_dp3` | `train_simple_dp3` | 实验名的一部分，也进入 Hydra 输出目录。 | 只影响记录和目录命名。 |
| `task_name` | `${task.name}` | `${task.name}` | 从 task YAML 继承任务名。 | 跟 task 配置保持一致即可。 |
| `shape_meta` | `${task.shape_meta}` | `${task.shape_meta}` | 定义 `obs` 和 `action` 的形状，policy 用它推导 action 维度、点云维度、状态维度。 | 必须与 zarr 真实 schema 一致。Flexiv `xyz` 是 `point_cloud=[1024,3]`，`xyzrgb` 是 `[1024,6]`，`agent_pos=[28]`，`action=[14]`。 |
| `exp_name` | `"debug"` | `"debug"` | WandB group 和训练脚本默认输出名可使用它。 | 建议按数据集、相机、点云模式、模型名命名，方便回溯。 |
| `horizon` | `16` | `4` | 数据集切片长度，也是扩散模型一次预测的动作轨迹长度 `T`。 | 必须满足 `horizon >= n_obs_steps`，并建议满足 `n_action_steps <= horizon - n_obs_steps + 1`，否则推理切片会不完整。长任务先增大到 `8/12/16`，显存不够先降 batch。 |
| `n_obs_steps` | `2` | `2` | 每次 policy 使用多少帧历史观测。`obs_as_global_cond=True` 时，观测特征会拼成 `obs_feature_dim * n_obs_steps`。 | 默认 `2` 合理。需要速度/方向信息时可增大，但会增加条件向量维度和显存。 |
| `n_action_steps` | `8` | `3` | 推理时从 `action_pred[:, n_obs_steps-1 : n_obs_steps-1+n_action_steps]` 取出的动作步数。 | 低延迟控制用小值，更平滑/少调用 policy 用大值。不要超过 `horizon - n_obs_steps + 1`。 |
| `n_latency_steps` | `0` | `0` | 当前两份主配置中保留的字段。 | 当前代码搜索未发现它被训练入口直接消费；不要把它当作已生效的机器人延迟补偿。 |
| `dataset_obs_steps` | `${n_obs_steps}` | `${n_obs_steps}` | 保留字段，用于与部分上游数据配置对齐。 | 当前 Flexiv dataset 路径主要使用 `horizon/pad_before/pad_after`，一般保持等于 `n_obs_steps`。 |
| `keypoint_visible_rate` | `1.0` | `1.0` | 上游 mask/keypoint 相关字段。 | 当前 DP3/SimpleDP3 policy 使用 `LowdimMaskGenerator`，该字段未直接传入；保持默认即可。 |
| 顶层 `obs_as_global_cond` | `True` | `True` | 语义上表示观测作为全局条件。真正传给 policy 的是 `policy.obs_as_global_cond`。 | 两处保持一致，默认全局条件路径最常用。 |

## Policy 与模型结构

| 参数 | `dp3.yaml` 默认值 | `simple_dp3.yaml` 默认值 | 含义/作用 | 调参建议 |
|---|---:|---:|---|---|
| `policy._target_` | `DP3` | `SimpleDP3` | Hydra 实例化的 policy 类。 | 快速验证数据和流程用 `simple_dp3`；追求更高容量再切 `dp3`。 |
| `policy.use_point_crop` | `true` | `true` | 主要被仿真 task 的 env runner 读取，用于点云裁剪；policy 构造函数没有显式参数，真实 Flexiv task 的 `env_runner: null` 下基本不影响训练。 | 仿真中可调；真实 zarr 离线训练更应在导出/预处理阶段保证点云质量。 |
| `policy.condition_type` | `film` | `film` | U-Net 残差块如何注入条件特征。`film` 用条件生成 scale/bias 调制特征。 | 建议保持 `film`。`SimpleDP3` 只实现 `film/add/mlp_film`；`DP3` 文件里有 attention 变体，但当前默认路径未验证，改前应单独跑 smoke test。 |
| `policy.use_down_condition` | `true` | `true` | 是否在 down path 注入条件。 | `SimpleDP3` 中不要改成 `false`，当前代码的 false 分支引用未定义的 `resnet2`，可能直接报错。 |
| `policy.use_mid_condition` | `true` | `true` | 是否在 U-Net bottleneck 注入条件。 | 一般保持 `true`。关闭会削弱状态/点云条件对动作生成的影响。 |
| `policy.use_up_condition` | `true` | `true` | 是否在 up path 注入条件。 | `SimpleDP3` 中不要改成 `false`，false 分支同样有未定义 `resnet2` 风险。 |
| `policy.diffusion_step_embed_dim` | `128` | `128` | 扩散 timestep 的位置编码维度，随后进入 MLP。 | 容量不够可增大到 `256`，但会增加所有条件块参数量。 |
| `policy.down_dims` | `[512,1024,2048]` | `[128,256,384]` | U-Net 各层通道数，是模型容量和显存的主控参数。 | OOM 时优先降它或改用 `simple_dp3`。欠拟合时再逐步增大，例如 `[256,512,1024]`。所有通道应能被 `n_groups` 整除。 |
| `policy.crop_shape` | `[80,80]` | `[80,80]` | 传给 `DP3Encoder(img_crop_shape=...)`。当前点云 encoder 代码没有实际使用该参数。 | 真实 Flexiv 点云训练中不要指望它改变输入点云裁剪。 |
| `policy.encoder_output_dim` | `64` | `64` | PointNet 输出维度；最终观测特征还会拼接 state MLP 的 `64` 维，因此默认观测特征约为 `128` 维。 | 点云信息复杂时可增大到 `128`，但会增加条件维度和 U-Net 参数。小数据集不宜盲目增大。 |
| `policy.kernel_size` | `5` | `5` | 1D temporal conv 卷积核大小，作用在动作时间维。 | 较大核扩大局部时间感受野，但计算更大。短 `horizon=4` 时继续增大意义有限。 |
| `policy.n_groups` | `8` | `8` | `Conv1dBlock` 内 `GroupNorm(n_groups, out_channels)`。 | 必须整除对应通道数。改 `down_dims` 后先检查每个通道是否能被 `n_groups` 整除。 |
| `policy.horizon` | `${horizon}` | `${horizon}` | 传入 policy 的预测序列长度。 | 与顶层 `horizon` 保持绑定。 |
| `policy.n_action_steps` | `${n_action_steps}` | `${n_action_steps}` | 传入 policy 的执行动作步数。 | 与顶层 `n_action_steps` 保持绑定。 |
| `policy.n_obs_steps` | `${n_obs_steps}` | `${n_obs_steps}` | 传入 policy 的观测历史步数。 | 与顶层 `n_obs_steps` 保持绑定。 |
| `policy.obs_as_global_cond` | `true` | `true` | `true` 时只把动作作为扩散轨迹，把点云/状态编码作为全局条件；`false` 时把观测特征拼进轨迹并使用 mask。 | 默认 `true` 更直接。改为 `false` 会改变输入维度和训练语义，旧 checkpoint 不兼容。 |
| `policy.shape_meta` | `${shape_meta}` | `${shape_meta}` | policy 解析 action、point cloud、agent_pos 的入口。 | 任何数据维度变化都必须先改 task YAML，再重新训练。 |
| `policy.use_pc_color` | `false` | `false` | `false` 时 policy 会截断 `point_cloud[..., :3]` 只用 xyz；`true` 时使用 xyzrgb。 | 用 `xyzrgb` zarr 时必须设为 `true`，同时 `pointcloud_encoder_cfg.in_channels=6`。脚本 `train_flexiv_dual_arm_dp3.sh xyzrgb ...` 会自动覆盖。 |
| `policy.pointnet_type` | `pointnet` | `pointnet` | 选择点云编码器类型。当前只实现 `pointnet`。 | 不要改成其他字符串，除非新增 encoder 实现。 |
| `policy.pointcloud_encoder_cfg.in_channels` | `3` | `3` | PointNet 输入点特征维度。代码会根据 `use_pc_color` 在 encoder 内重设为 `3` 或 `6`。 | `xyz` 用 `3`，`xyzrgb` 用 `6`。保持与 task 和 zarr 一致。 |
| `policy.pointcloud_encoder_cfg.out_channels` | `${policy.encoder_output_dim}` | `${policy.encoder_output_dim}` | PointNet 输出维度。 | 通常通过 `encoder_output_dim` 调，不单独改这里。 |
| `policy.pointcloud_encoder_cfg.use_layernorm` | `true` | `true` | PointNet MLP 各层是否加 LayerNorm。 | 点云尺度/分布变化大时建议保持开启。 |
| `policy.pointcloud_encoder_cfg.final_norm` | `layernorm` | `layernorm` | PointNet 最终投影后是否归一化。 | 默认稳定；若特征幅度被过度压制，可实验 `none`。 |
| `policy.pointcloud_encoder_cfg.normal_channel` | `false` | `false` | 上游 PointNet 风格字段。当前本仓库 PointNet 实现未直接使用。 | 保持默认。 |

## 扩散噪声与采样

| 参数 | `dp3.yaml` 默认值 | `simple_dp3.yaml` 默认值 | 含义/作用 | 调参建议 |
|---|---:|---:|---|---|
| `policy.noise_scheduler._target_` | `DDIMScheduler` | `DDIMScheduler` | 训练加噪和推理去噪的 scheduler。 | 默认 DDIM 适合少步推理。换 scheduler 会改变训练/推理行为，旧实验不可直接对比。 |
| `num_train_timesteps` | `100` | `100` | 训练时随机采样扩散步 `t in [0,100)`。 | 增大可能提升扩散细粒度，但训练更难更慢；小数据先保持 `100`。 |
| `beta_start` | `0.0001` | `0.0001` | 噪声日程起始 beta。 | 通常不优先调。 |
| `beta_end` | `0.02` | `0.02` | 噪声日程结束 beta。 | 通常不优先调。 |
| `beta_schedule` | `squaredcos_cap_v2` | `squaredcos_cap_v2` | beta 随 timestep 的变化曲线。 | 先保持默认。 |
| `clip_sample` | `True` | `True` | scheduler step 后裁剪样本。 | 动作归一化范围明确时一般稳定；若动作幅值被明显压扁，可单独对比。 |
| `set_alpha_to_one` | `True` | `True` | DDIM 最后一步 alpha 处理。 | 通常不调。 |
| `steps_offset` | `0` | `0` | DDIM timestep 偏移。 | 通常不调。 |
| `prediction_type` | `sample` | `sample` | 模型预测目标。`sample` 表示直接预测干净轨迹，loss target 是 `trajectory`。 | 改为 `epsilon` 或 `v_prediction` 会改变 loss 语义，必须重训。 |
| `policy.num_inference_steps` | `10` | `10` | 推理去噪步数。policy 会 `scheduler.set_timesteps(10)` 并循环采样。 | 直接影响推理速度。质量不足可试 `16/20`；实时控制延迟大时降到 `5/8` 对比。 |

## EMA、优化器与训练循环

| 参数 | `dp3.yaml` 默认值 | `simple_dp3.yaml` 默认值 | 含义/作用 | 调参建议 |
|---|---:|---:|---|---|
| `ema.update_after_step` | `0` | `0` | 从第几步后开始 EMA。 | 默认从头开始。 |
| `ema.inv_gamma` | `1.0` | `1.0` | EMA warmup 速度参数。 | 通常不调。 |
| `ema.power` | `0.75` | `0.75` | EMA decay warmup 曲线。当前实现注释说明 `0.75` 更适合较短训练。 | 默认合理。训练特别长可比较 `2/3`。 |
| `ema.min_value` | `0.0` | `0.0` | EMA decay 下限。 | 通常不调。 |
| `ema.max_value` | `0.9999` | `0.9999` | EMA decay 上限。 | 小数据或快速变化任务可略降，例如 `0.999`。 |
| `optimizer._target_` | `AdamW` | `AdamW` | 优化器。 | 默认合适。 |
| `optimizer.lr` | `1e-4` | `1e-4` | 基础学习率。 | 最常调。loss 抖动/发散降到 `5e-5`；明显欠拟合且稳定可试 `2e-4`。 |
| `optimizer.betas` | `[0.95,0.999]` | `[0.95,0.999]` | AdamW 动量参数。 | 通常不优先调。 |
| `optimizer.eps` | `1e-8` | `1e-8` | AdamW 数值稳定项。 | 通常不调。 |
| `optimizer.weight_decay` | `1e-6` | `1e-6` | 权重衰减。 | 过拟合可增到 `1e-5`；欠拟合可保持很小。 |
| `training.device` | `cuda:0` | `cuda:0` | 训练设备。脚本里还会设置 `CUDA_VISIBLE_DEVICES`。 | 多卡时脚本参数的 GPU id 和这里的 `cuda:0` 配合使用。 |
| `training.seed` | `42` | `42` | PyTorch、NumPy、random 随机种子。 | 做对比实验时只改一个超参并固定 seed；最终可多 seed 验证。 |
| `training.debug` | `False` | `False` | `True` 时 train.py 会强制缩短训练：`num_epochs=100,max_train_steps=10,max_val_steps=3,checkpoint_every=1,sample_every=1` 等。 | smoke test 用 `True`；正式训练用 `False`。 |
| `training.resume` | `True` | `False` | 启动时是否尝试从输出目录下 `latest.ckpt` 恢复。 | `simple_dp3` 当前默认从头训练；训练脚本还会在输出目录已存在时默认报错，避免新旧文件混在一起。 |
| `training.lr_scheduler` | `cosine` | `cosine` | 使用 diffusers 的学习率调度器。 | 默认余弦退火。短训练可试 `constant_with_warmup`，但需保持 warmup 设置。 |
| `training.lr_warmup_steps` | `500` | `50` | 学习率 warmup 步数。 | 小数据/短训练时可降到 `50/100`，否则 500 可能占比过高。 |
| `training.num_epochs` | `3000` | `600` | epoch 数。 | zarr episode 少时 epoch 可以多，但要看 `global_step` 和 loss 曲线，不只看 epoch。 |
| `training.gradient_accumulate_every` | `1` | `1` | 梯度累积步数。train.py 中用它缩放 loss，并按 global step 间隔执行 optimizer step。 | 显存不够时先降 batch；需要保持等效 batch 时再增大该值。注意当前实现从 `global_step=0` 就 step，一般保持 `1` 最少出问题。 |
| `training.use_ema` | `True` | `True` | 训练时维护 EMA policy，评估和 sample 用 EMA。 | 默认开启。 |
| `training.rollout_every` | `200` | `200` | 每多少 epoch 跑 env rollout。 | Flexiv 真实 task 的 `env_runner: null`，不会跑仿真 rollout。 |
| `training.checkpoint_every` | `200` | `200` | 每多少 epoch 检查保存 checkpoint。 | 配合 `checkpoint.save_ckpt` 才生效；如果要按步数保存，需要先换算成 epoch 或改训练循环。 |
| `training.val_every` | `1` | `1` | 每多少 epoch 跑 validation。 | 当前 train.py 里 `RUN_VALIDATION = False`，所以默认不会实际跑 validation。 |
| `training.sample_every` | `5` | `5` | 每多少 epoch 在训练 batch 上采样动作并记录 `train_action_mse_error`。 | 调扩散采样质量时关注该指标，但它不是闭环成功率。 |
| `training.max_train_steps` | `null` | `null` | 每个 epoch 最多训练多少 batch。 | smoke test 可设小，例如 `10/20`；正式训练保持 `null`。 |
| `training.max_val_steps` | `null` | `null` | 每次 validation 最多多少 batch。 | 当前 validation 默认关闭。 |
| `training.tqdm_interval_sec` | `1.0` | `1.0` | tqdm 刷新间隔。 | 只影响显示开销。 |

## DataLoader、日志与保存

| 参数 | `dp3.yaml` 默认值 | `simple_dp3.yaml` 默认值 | 含义/作用 | 调参建议 |
|---|---:|---:|---|---|
| `dataloader.batch_size` | `128` | `128` | 训练 batch。 | OOM 优先降到 `64/32/16`。若 GPU 利用率低且显存充足可增大。 |
| `dataloader.num_workers` | `8` | `8` | 训练数据加载进程数。 | zarr 在本地 SSD 上可用 `8`；CPU/IO 紧张时降到 `4/2`。 |
| `dataloader.shuffle` | `True` | `True` | 训练集随机打乱。 | 保持开启。 |
| `dataloader.pin_memory` | `True` | `True` | DataLoader pinned memory，加速 CPU 到 GPU 拷贝。 | CUDA 训练保持开启。 |
| `dataloader.persistent_workers` | `False` | `False` | worker 是否跨 epoch 常驻。 | 长训练可试 `True`，但若 zarr/worker 退出异常就保持 `False`。 |
| `val_dataloader.*` | batch `128`, workers `8`, no shuffle | 同左 | validation DataLoader。 | 当前 validation 关闭，影响较小。 |
| `logging.group` | `${exp_name}` | `${exp_name}` | WandB group。 | 按实验系列命名。 |
| `logging.id` | `null` | `null` | WandB run id。 | 需要恢复同一个 WandB run 时再设。 |
| `logging.mode` | `online` | `online` | WandB 模式。Flexiv 脚本默认 `WANDB_MODE=offline` 覆盖。 | 服务器/离线环境用 `offline` 或 `disabled`。 |
| `logging.name` | `${training.seed}` | `${training.seed}` | WandB run name。 | 建议加入模型、数据集、关键超参。 |
| `logging.project` | `dp3` | `dp3` | WandB project。 | 可按项目改。 |
| `logging.resume` | `true` | `true` | WandB resume 行为。 | 与 `logging.id` 配套使用更明确。 |
| `checkpoint.save_ckpt` | `False` | `True` | 是否按 `checkpoint_every` 保存 checkpoint。 | 正式训练必须开启。注意 `dp3.yaml` 默认关闭，脚本默认通过 `SAVE_CKPT=True` 覆盖。 |
| `checkpoint.topk.monitor_key` | `test_mean_score` | `test_mean_score` | TopK 保存依据。真实 Flexiv task 无 env_runner 时，train.py 会把它设为 `-train_loss`。 | 真实训练时 TopK 近似按训练 loss 选，不代表真实闭环效果。 |
| `checkpoint.topk.mode` | `max` | `max` | TopK 方向。 | 对 `test_mean_score=-train_loss` 来说 `max` 等价于 loss 越低越好。 |
| `checkpoint.topk.k` | `1` | `1` | 保留 top-k 数量。 | 想保留更多候选可设 `3/5`。 |
| `checkpoint.save_last_ckpt` | `True` | `True` | 保存并覆盖 `checkpoints/latest.ckpt`。仅当 `save_ckpt=True` 生效。 | 正式训练保持开启，方便 resume 或找最近一次权重。 |
| `checkpoint.save_every_ckpt` | `True` | `True` | 每次 `checkpoint_every` 触发时，额外保留一个独立 ckpt 文件，不受 top-k 删除逻辑影响。 | 想保留周期历史 ckpt 时保持开启；磁盘紧张时设为 `False`。 |
| `checkpoint.save_every_ckpt_format_str` | `epoch={epoch:04d}-global_step={global_step}.ckpt` | 同左 | 周期历史 ckpt 的文件名模板。可使用 `epoch`、`global_step`、`train_loss`、`test_mean_score` 等当前日志字段。 | 建议默认即可，避免与 top-k 文件名冲突。 |
| `checkpoint.save_last_snapshot` | `False` | `False` | 保存完整 Python snapshot。 | 一般不需要，代码兼容性要求更高。 |
| `multi_run.run_dir` | `data/outputs/...` | `data/outputs/...` | Hydra sweep 输出目录模板。 | 脚本可用 `RUN_DIR` 覆盖。 |
| `multi_run.wandb_name_base` | 时间戳模板 | 时间戳模板 | 多实验 WandB 名称基础。 | 只影响记录。 |
| `hydra.run.dir` | `data/outputs/...` | `data/outputs/...` | 单次运行输出目录。训练脚本默认覆盖为 `3D-Diffusion-Policy/outputs/<exp_name>_seed<seed>`。 | 复现实验时显式指定；通过训练脚本启动时，已有目录会默认报错，只有加 `--overwrite` 才会先删除整个目录。 |
| `hydra.sweep.dir/subdir` | 时间戳 + job num | 同左 | sweep 输出组织方式。 | 批量扫参时保留默认即可。 |

## Flexiv 双臂训练时的调参方法

1. 先固定数据 schema，再调模型。

   Flexiv `xyz` task 期望 `data/point_cloud` 是 `T x 1024 x 3`，`xyzrgb` task 期望 `T x 1024 x 6`，两者都要求 `data/state` 为 `T x 28`、`data/action` 为 `T x 14`。先用检查工具确认 zarr，再选择：

   ```bash
   bash scripts/train_flexiv_dual_arm_dp3.sh xyz /path/to/data.zarr simple_dp3 0 42
   bash scripts/train_flexiv_dual_arm_dp3.sh xyzrgb /path/to/data.zarr simple_dp3 0 42
   ```

   该脚本会按 `xyz/xyzrgb` 自动覆盖 `task`、`policy.use_pc_color` 和 `policy.pointcloud_encoder_cfg.in_channels`。如果目标输出目录已存在，脚本会默认报错；要重用同名目录并清空旧文件，显式追加 `--overwrite`。

2. 先用 `simple_dp3` 做 smoke test。

   建议先跑短训练，确认 zarr、shape、CUDA、WandB、checkpoint 都正常：

   ```bash
   DEBUG=True WANDB_MODE=offline SAVE_CKPT=True \
   bash scripts/train_flexiv_dual_arm_dp3.sh xyz /path/to/data.zarr simple_dp3 0 42
   ```

   smoke test 通过后再关 `DEBUG`，并显式设置新的输出目录；如果确实要覆盖旧目录，使用 `--overwrite`，它会删除整个 Hydra 输出目录后再训练。

3. 按显存预算调容量。

   调参优先级建议如下：

   - OOM：先降 `dataloader.batch_size`，再降 `policy.down_dims`，最后缩短 `horizon` 或改回 `simple_dp3`。
   - 欠拟合：先延长训练/检查数据质量，再把 `simple_dp3` 切到 `dp3`，或把 `down_dims` 从 `[128,256,384]` 增到 `[256,512,1024]`。
   - 点云信息不足：优先检查点云预处理和可视化；模型侧可尝试 `encoder_output_dim=128` 或使用 `xyzrgb`。

4. 按控制需求调时间窗口。

   经验顺序：

   - 保持 `n_obs_steps=2` 作为基线。
   - 短动作或小数据：`simple_dp3` 的 `horizon=4,n_action_steps=3` 足够快速。
   - 长动作或希望减少 policy 调用频率：尝试 `horizon=8,n_action_steps=4`，再到 `horizon=16,n_action_steps=8`。
   - 始终检查 `n_action_steps <= horizon - n_obs_steps + 1`。例如 `horizon=4,n_obs_steps=2` 时，`n_action_steps` 最大就是 `3`。

5. 采样速度和质量用 `num_inference_steps` 权衡。

   默认 `10` 是速度优先的设置。动作噪声明显或 `train_action_mse_error` 不稳定时，可试 `16/20`；实时推理延迟太高时可试 `5/8`。不要同时大幅改 `num_train_timesteps` 和 `num_inference_steps`，否则难判断问题来源。

6. 学习率先小范围搜索。

   默认 `lr=1e-4`。建议只做小范围：

   - loss 发散或抖动大：`5e-5`。
   - loss 稳定但下降慢：`2e-4`。
   - 短训练或小数据：把 `lr_warmup_steps` 从 `500` 降到 `50/100`。

7. checkpoint 和 resume 要显式管理。

   `dp3.yaml` 默认 `checkpoint.save_ckpt=False`，正式训练应开启。脚本默认 `SAVE_CKPT=True`，但如果直接跑 `python train.py --config-name=dp3.yaml`，要手动加：

   ```bash
   checkpoint.save_ckpt=True checkpoint.save_last_ckpt=True checkpoint.save_every_ckpt=True
   ```

   `training.resume=True` 会从当前输出目录找 `latest.ckpt`。`checkpoint.save_every_ckpt=True` 会额外保留类似 `epoch=0200-global_step=12345.ckpt` 的周期历史文件；这条路径不经过 top-k，不会被 `checkpoint.topk.k=1` 删除。通过训练脚本启动时，想从头跑新实验最好换 `RUN_DIR`/`EXP_NAME`；显式加 `--overwrite` 时会删除整个目标输出目录，避免残留旧 ckpt。

8. 不建议优先调的参数。

   以下参数在当前代码中要么影响不直接，要么风险较高，除非有明确实验目的，否则先保持默认：

   - `n_latency_steps`、`dataset_obs_steps`、`keypoint_visible_rate`。
   - `crop_shape`：当前点云 encoder 没有实际使用。
   - `condition_type`：默认 `film` 最稳。
   - `SimpleDP3` 的 `use_down_condition/use_up_condition`：不要设为 `false`，当前 false 分支有代码风险。
   - `beta_start/beta_end/beta_schedule/set_alpha_to_one/steps_offset`：属于扩散日程底层参数，优先级低于数据、窗口、容量、学习率。

## 推荐起步配置

| 目标 | 建议配置 |
|---|---|
| 检查数据和训练链路 | `simple_dp3`, `DEBUG=True`, `batch_size=32/64`, `WANDB_MODE=offline`, `SAVE_CKPT=True` |
| 小数据快速 baseline | `simple_dp3`, `horizon=4`, `n_action_steps=3`, `lr=1e-4`, `num_inference_steps=10` |
| 中等容量真实训练 | `simple_dp3` + `down_dims=[256,512,1024]` 或直接 `dp3`，视显存决定 |
| 长动作任务 | `horizon=8/16`, `n_action_steps=4/8`, 先降 batch 保证稳定 |
| 推理更稳但更慢 | `num_inference_steps=16/20` |
| 推理更快但可能更粗 | `num_inference_steps=5/8` |
