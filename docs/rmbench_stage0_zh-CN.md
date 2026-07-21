# RMBench Stage 0（阶段 0）

阶段 0 将 RMBench 以 Git subtree 固定集成，并在独立的 SAPIEN 3 环境中
完成最小运行验证；不修改现有 `dp3` 环境，也不修改
`PointCloudBuilder` 子模块。

## 固定输入

- 分支：`develop/RMBench`
- RMBench 上游：`https://github.com/RoboTwin-Platform/RMBench.git`
- subtree 固定提交：`87e0498891073d483d330195c0f160709bd92ff5`
- 运行环境：`dp3-rmbench`，Python 3.10
- 运行时 wheel：PyTorch `2.7.1+cu128`、torchvision `0.22.1+cu128`，用于 RTX
  5080 的 `sm_120`；PyTorch3D 和 CuRobo 均针对该 ABI 重编译
- CuRobo：`v0.7.8`，由 `scripts/rmbench/bootstrap_env.sh` 固定
- Warp：`1.6.2`，保留 CuRobo 0.7.8 使用的 `warp.torch` API
- Hugging Face 数据集：`TianxingChen/RMBench`，不可变资产 revision 为
  `d899d72b53270a89f71d216c08ecbd4d9a7004fd`（官方 `refs/pr/8` 提交；审计
  其他官方 revision 时可用 `RMBENCH_HF_REVISION` 覆盖）

只下载 `embodiments/**` 和 `objects/**`，不下载完整数据集，也不进入
策略训练资产范围。

## 重现命令

在仓库根目录执行，并确保当前没有激活 `dp3`：

```bash
scripts/rmbench/bootstrap_env.sh
scripts/rmbench/fetch_assets.sh --dry-run
scripts/rmbench/fetch_assets.sh
```

资产命令需要已认证的 Hugging Face 会话；下载完成后会更新 embodiment
配置中的本地绝对路径。

默认使用官方 `refs/pr/8` 背后的不可变 commit，因为当前数据集 `main` 列表
缺少 `franka-panda`；该 revision 同时包含所需的 Franka、Aloha 和
`005_button` 资产。只下载限定的资产路径。

helper 会明确检查 `aloha-agilex`、`franka-panda` 和 `005_button`。如果当前
Hugging Face snapshot 缺少其中一个路径，命令会失败并报告缺失项；不会用
Aloha 目录伪造替代缺失的 embodiment。

## 验证等级

```bash
PYTHONNOUSERSITE=1 conda run -n dp3-rmbench \
  python scripts/rmbench/doctor.py --strict --check-assets --check-sim

PYTHONNOUSERSITE=1 conda run -n dp3-rmbench \
  python scripts/rmbench/smoke_test.py --level 0
PYTHONNOUSERSITE=1 conda run -n dp3-rmbench \
  python scripts/rmbench/smoke_test.py --level 1
PYTHONNOUSERSITE=1 conda run -n dp3-rmbench \
  python scripts/rmbench/smoke_test.py --level 2
```

等级 0 验证导入；等级 1 创建最小 SAPIEN 3 场景并导入
`put_back_block`；等级 2 在 scoped 资产存在时初始化任务并检查 observation
契约，未认证或未下载资产时明确报告 `SKIP`。

在已验证的 RTX 5080 主机上，等级 1 和等级 2 通过。等级 2 初始化时
Warp 1.6.2 会输出非致命的 `cuDeviceGetUuid` 驱动 API 警告；诊断主机时应
保留该输出，不能把它静默当成干净的系统日志。

现有 `dp3` 的基线快照位于 `environments/snapshots/dp3_before_rmbench/`。
阶段 0 只允许本地验证，不创建新分支或 worktree，不向远程 push，也不
执行 force checkout/reset。

## 常见错误与恢复

- **分支错误或源码被 shadow：** `doctor.py --strict` 会在仿真前失败。检查
  `git branch --show-current`、`git status` 和 `PYTHONNOUSERSITE=1`；DP3
  必须导入当前 checkout。
- **Hugging Face 认证或限流：** 使用本机 HF 会话认证后重新运行
  `fetch_assets.sh`。不完整 snapshot 必须报告失败，不能用其他机器人目录
  替代，也不能下载完整数据集。
- **资产缺失：** 重新运行 `fetch_assets.sh`；它只下载
  `embodiments/**` 和 `objects/**`，并明确检查 Aloha、Franka 与
  `005_button`。
- **Vulkan/OIDN/驱动警告：** 本机 SAPIEN 可能输出 `svulkan2` 或 OIDN
  诊断信息。报告中保留这些信息；即使 import 通过，任务初始化失败仍算失败。
- **CuRobo/PyTorch3D 扩展不匹配：** 在 `dp3-rmbench` 中重新运行
  `bootstrap_env.sh`；只有 CUDA 头文件、Torch ABI 或 `sm_120` 扩展检查不满足
  时才重编译。

如需重建隔离环境，先保留需要的报告文件，然后只删除指定环境：
`conda env remove -n dp3-rmbench`。随后重新运行 `bootstrap_env.sh`、资产下载
和上述验证命令。被忽略的资产目录可以保留，也可以单独重新下载。

## Subtree 更新

阶段 0 中 subtree 更新仅允许审查。使用
`scripts/rmbench/update_subtree.sh --dry-run` 检查候选上游提交，审查新的
固定提交并重新运行完整验证套件后再应用。当前已完成的 Stage 0 不更新 subtree，
也不向远程 push。

## Stage 1 入口

本报告接受后才进入 Stage 1。第一个有界切片是使用已验证 Aloha
`put_back_block`、单头相机、XYZ 点云和小规模 Zarr 导出的采集，并覆盖
`n_obs_steps` 为 1 和 2 的配置。策略训练、完整数据集采集、多相机、Belief DP3
和在线机器人控制不属于 Stage 0 交付物。

## 边界说明

不要直接运行上游 `_install.sh`：它依赖环境中的未固定 pip 状态、无内容
保护地修改文件，并会下载完整资产树。应使用仓库内 helper 脚本，以便在
执行前检查分支、固定提交、环境隔离和资产范围。
