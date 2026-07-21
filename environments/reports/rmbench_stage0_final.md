# RMBench Stage 0 final audit

Status: **PASS**

The runtime, scoped assets, and bounded simulation smoke path all pass the
Stage 0 gates. The asset helper pins immutable commit
`d899d72b53270a89f71d216c08ecbd4d9a7004fd`, the official `refs/pr/8` snapshot,
because the current HF `main` snapshot omits `franka-panda`.

## Git and integration

| Item | Evidence |
| --- | --- |
| Repository | `/home/deepcybo/workspace/3D-Diffusion-Policy` |
| Branch | `develop/RMBench` |
| Start commit | `1c8400c19cfa46ca4dba227bb321fb495da3271c` |
| End commit | `7b817a8667ae168466506bf95001551f51ebf851` (worktree changes are intentionally uncommitted) |
| RMBench subtree | `third_party/sim/RMBench`, squashed integration commit `7909e1b649bc387955f2fac9a152d2a4f439b510` |
| Upstream pin | `87e0498891073d483d330195c0f160709bd92ff5` |
| PointCloudBuilder | Gitlink `e19d89cb3e88a09db35eb5cdfbaa992ada5618d5`; nested status clean |

The root remote `rmbench-upstream` points to
`https://github.com/RoboTwin-Platform/RMBench.git`. No PointCloudBuilder
content or gitlink was changed.

## Environment

- Environment: `dp3-rmbench`, Python 3.10.20.
- PyTorch `2.7.1+cu128`, torchvision `0.22.1+cu128`; CUDA compiler
  `12.8.61`, runtime development package `12.8.57`.
- GPU: NVIDIA GeForce RTX 5080; CUDA visible and `sm_120` supported.
- SAPIEN `3.0.0b1`, MPLib `0.2.1`, Gymnasium `0.29.1`, NumPy `1.26.4`,
  SciPy `1.10.1`, Open3D `0.18.0`.
- PyTorch3D `0.7.9`, source commit
  `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`, rebuilt against the active
  Torch ABI; `pytorch3d.ops` GPU FPS passed.
- CuRobo `v0.7.8`, commit `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`,
  rebuilt for `sm_120`.
- Warp `1.6.2`, pinned because CuRobo 0.7.8 uses `warp.torch`.
- Environment snapshots: `environments/snapshots/dp3_rmbench_after_install/`.

No Stage 0 command targeted the existing `dp3` environment except for
read-only baseline checks. Its `conda-list.txt` remains identical to
`environments/snapshots/dp3_before_rmbench/conda-list.txt`, and the post-work
baseline report passes the existing DP3, scenes, and dataset import checks.
The live `pip-freeze` is not byte-identical to the old snapshot because of
pre-existing/concurrent package drift (including editable VCS pointers); this
was not introduced by the Stage 0 bootstrap. `pip check` still reports the
baseline `sapien 2.2.1` missing `opencv-python` issue, with no new baseline
import failure.

## Assets

The scoped download fetched 377 files and never downloaded `data/**`:

| Required path | Final local result |
| --- | --- |
| `embodiments/aloha-agilex` | present |
| `embodiments/franka-panda` | present |
| `objects/005_button` | present |

The live HF `main` listing contained 5996 files, zero paths matching `franka`,
84 `aloha-agilex` paths, and 2 `005_button` paths. The immutable official
`refs/pr/8` commit contains 30 `franka-panda` paths, 84 `aloha-agilex` paths,
and 2 `005_button` paths. No substitute embodiment was created.

## Validation

- Repository tests: `10 passed`.
- Level 0: PASS; DP3, Torch, RMBench envs, SAPIEN, Gymnasium, MPLib,
  PyTorch3D ops, and CuRobo extensions import.
- Level 1: PASS; minimal SAPIEN scene and `put_back_block` import.
- Level 2: PASS with available Aloha/button assets; observation keys are
  `endpose`, `joint_action`, `observation`, `pointcloud`, and `third_view_rgb`.
- `doctor --strict --check-sim`: simulation checks PASS.
- `doctor --strict --check-assets --check-sim`: PASS.
- Reports: `environments/reports/rmbench_stage0_level{0,1,2}.json`,
  `dp3_rmbench_environment.json`, and the baseline snapshot.

Level 2 retains non-fatal host warnings from Warp 1.6.2
(`cuDeviceGetUuid`, driver API 36) and SAPIEN's OIDN CUDA backend. The task
still completes and the warnings are not hidden or converted into a clean
system-level result.

## Reproduction and exit condition

```bash
scripts/rmbench/bootstrap_env.sh
scripts/rmbench/fetch_assets.sh
PYTHONNOUSERSITE=1 conda run -n dp3-rmbench python scripts/rmbench/doctor.py --strict --check-assets --check-sim
```

Stage 0 is complete. Stage 1 remains out of scope; its entry point is the
policy/data workflow after this asset gate, not training during Stage 0.
