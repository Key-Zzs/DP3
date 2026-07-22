# 双 Flexiv Rizon4s + GN01 embodiment

这是阶段 2 的操作入口：从固定版本的官方
`flexiv_description` xacro 生成被忽略的 RMBench runtime bundle，并验证
只有一个固定头部相机、没有任务逻辑的 SAPIEN embodiment。

当前仿真装配参数为：桌面长边沿 world-y 方向；在负 world-x 长边外设置
一个比桌面高 20 cm 的底座架；双臂基座中心距 30 cm；左/右基座绕 x 轴
分别为 `-45/+45` 度。real 的 home 关节角保持原值，不因仿真装配姿态而修改。

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

可视化验收运行可交互的 SAPIEN 视口：

```bash
conda run -n dp3-rmbench python scripts/rmbench/flexiv/visualize_embodiment.py \
  --gui --mode home --view head-camera
```

GUI 会一直运行到手动关闭 SAPIEN 窗口或按 `Ctrl-C`。鼠标左键拖动旋转视图，
鼠标右键拖动平移，滚轮缩放；键盘相机移动（包括 W/A/S/D）已禁用。需要调试面板时加
`--show-panels`；面板使用每次进程独立的初始布局，不读取用户全局
`~/.sapien/imgui.ini`，因此不会再次堆叠遮挡视口。需要限时运行时加
`--seconds 20`。无显示环境时可运行：

```bash
conda run -n dp3-rmbench python scripts/rmbench/flexiv/visualize_embodiment.py \
  --headless --view front
```

验收不仅检查数组尺寸，还要求 RGB 不是纯平背景且深度存在有效像素。
缺少 Vulkan 或 CUDA 设备时必须记录为 `SKIP`，不能伪造为通过。

## 文件与边界

- 源 YAML：`sim_assets/flexiv_rizon4s_dual_gn01/`。
- 装配架 URDF：`sim_assets/flexiv_rizon4s_dual_gn01/rack.urdf`；采用中心立柱、
  左右外八 45° 斜撑和与左右机械臂底座姿态完全一致的斜安装面。
- 生成器和 URDF 审计：`scripts/rmbench/flexiv/`。
- runtime adapters：`3D-Diffusion-Policy/diffusion_policy_3d/sim/flexiv/`。
- 无任务 smoke 环境：`third_party/sim/RMBench/envs/flexiv_embodiment_smoke.py`。
- RMBench 注册：`third_party/sim/RMBench/task_config/`。
- 测试：`tests/test_flexiv_*.py`。

bundle 被忽略但可重复生成，不能手改其中的 URDF/config；不能修改官方
submodule 或 `PointCloudBuilder` gitlink。阶段 3 才单独处理任务、数据和
策略。
