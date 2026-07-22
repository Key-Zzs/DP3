# 双 Flexiv Rizon4s + GN01 embodiment

这是阶段 2 的操作入口：从固定版本的官方
`flexiv_description` xacro 生成被忽略的 RMBench runtime bundle，并验证
只有一个固定头部相机、没有任务逻辑的 SAPIEN embodiment。

## 生成

```bash
conda activate dp3-rmbench
bash scripts/rmbench/flexiv/bootstrap_description.sh --force
```

helper 会拒绝错误分支或错误环境。Docker 不可用时使用本地 xacro 路径，
并在 `generation_manifest.json` 中记录该事实。

## 有界验证

```bash
conda run -n dp3-rmbench python scripts/rmbench/flexiv/validate_embodiment.py \
  --headless --out-dir outputs/rmbench_flexiv_embodiment
conda run -n dp3-rmbench python scripts/rmbench/flexiv/planner_smoke.py \
  --out-dir outputs/rmbench_flexiv_embodiment
conda run -n dp3-rmbench python scripts/rmbench/flexiv/capture_acceptance_artifacts.py \
  --headless --capture-all --output-dir outputs/rmbench_flexiv_acceptance
```

需要 GUI 时运行 `visualize_embodiment.py --gui --mode home
--view head-camera`；无显示环境时使用 `--headless`。缺少 Vulkan 或 CUDA
设备时必须记录为 `SKIP`，不能伪造为通过。

## 文件与边界

- 源 YAML：`sim_assets/flexiv_rizon4s_dual_gn01/`。
- 生成器和 URDF 审计：`scripts/rmbench/flexiv/`。
- runtime adapters：`3D-Diffusion-Policy/diffusion_policy_3d/sim/flexiv/`。
- 无任务 smoke 环境：`third_party/sim/RMBench/envs/flexiv_embodiment_smoke.py`。
- RMBench 注册：`third_party/sim/RMBench/task_config/`。
- 测试：`tests/test_flexiv_*.py`。

bundle 被忽略但可重复生成，不能手改其中的 URDF/config；不能修改官方
submodule 或 `PointCloudBuilder` gitlink。阶段 3 才单独处理任务、数据和
策略。
