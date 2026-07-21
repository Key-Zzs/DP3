# RMBench vendor metadata

This directory is a Git subtree of the RMBench repository, kept at the Stage 0
pin below.

- Upstream: `https://github.com/RoboTwin-Platform/RMBench.git`
- Pinned upstream commit: `87e0498891073d483d330195c0f160709bd92ff5`
- Integration date: 2026-07-21
- License: MIT; see [`LICENSE`](LICENSE)
- Asset repository: Hugging Face dataset `TianxingChen/RMBench`, asset revision
  `d899d72b53270a89f71d216c08ecbd4d9a7004fd` (the official `refs/pr/8` commit;
  current `main` omits `franka-panda`)
- Integration: Git subtree at `third_party/sim/RMBench`

## Subtree maintenance

The initial integration was performed with:

```bash
git subtree add \
  --prefix=third_party/sim/RMBench \
  rmbench-upstream \
  87e0498891073d483d330195c0f160709bd92ff5 \
  --squash
```

Updates must be reviewed and pinned explicitly:

```bash
git fetch rmbench-upstream
git subtree pull \
  --prefix=third_party/sim/RMBench \
  rmbench-upstream <reviewed-commit> \
  --squash
```

The repository-local helper `scripts/rmbench/update_subtree.sh` performs the
same operation after validating the remote, target path, and requested commit.

## Local modifications and boundaries

Stage 0 adds repository-owned metadata, scripts, tests, and documentation
outside the upstream source tree. It does not modify RMBench source files,
`policy/DP3`, or the `PointCloudBuilder` submodule. The upstream `policy/DP3`
directory is reference material only; the primary DP3 implementation remains
the editable `diffusion_policy_3d` package in this repository.

The existing `dp3` environment is intentionally not reused: it contains the
current DP3/Flexiv/MetaWorld/Adroit/DexArt workflow and SAPIEN 2, while RMBench
requires SAPIEN 3. Stage 0 uses the separate `dp3-rmbench` environment.

Only the RMBench `embodiments/**` and `objects/**` assets are downloaded in
Stage 0. The complete RMBench dataset is not downloaded. Local assets, data,
CuRobo, caches, videos, and logs are ignored by the outer repository.
