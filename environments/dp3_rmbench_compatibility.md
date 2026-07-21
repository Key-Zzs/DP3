# `dp3-rmbench` compatibility record

This record describes the Stage 0 runtime contract. The generated files under
`environments/snapshots/dp3_rmbench_*` and `dp3_rmbench_lock.txt` are the
machine-state evidence for the actual installation.

| Component | Stage 0 contract |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.7.1, CUDA 12.8 wheel; required for RTX 5080 `sm_120` |
| torchvision | 0.22.1, CUDA 12.8 wheel |
| CUDA compiler | Conda `cuda-nvcc=12.8.61` |
| SAPIEN | 3.0.0b1 |
| MPLib | 0.2.1 |
| Gymnasium | 0.29.1 |
| NumPy | 1.26.4 |
| SciPy | 1.10.1 |
| Open3D | 0.18.0 |
| PyTorch3D | source commit `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba` |
| CuRobo | Git tag `v0.7.8` |
| warp-lang | 1.6.2; retains the `warp.torch` API used by CuRobo 0.7.8 |
| DP3 | editable install from this repository |
| RMBench assets | `TianxingChen/RMBench@d899d72b53270a89f71d216c08ecbd4d9a7004fd`; scoped `embodiments/**` and `objects/**` only |
| setuptools | 75.1.0, retained for SAPIEN's `pkg_resources` import |
| scikit-image | 0.22.0, kept compatible with SciPy 1.10.1 |

The existing `dp3` environment is a separate baseline. It is not an input to
the runtime and is not modified by the bootstrap script. `PYTHONNOUSERSITE=1`
is used for validation so user-site packages cannot silently change imports or
`pip check` results.

The upstream RMBench/CuRobo requirements were audited against this host. The
upstream Torch 2.4.1/cu121 combination cannot execute on the RTX 5080
(`sm_120`), so the isolated environment deliberately uses the compatible
Torch 2.7.1/cu128 wheel set and CUDA 12.8 compiler. PyTorch3D and CuRobo
extensions are rebuilt against that exact Torch ABI. Warp 1.15 is not used:
CuRobo 0.7.8 imports `warp.torch`, which is retained by the pinned 1.6.2
release.

## Compatibility patches

`scripts/rmbench/patch_site_packages.py` applies only content-checked,
idempotent patches to the installed SAPIEN and MPLib packages. Each changed
file receives a `.rmbench-stage0.orig` backup. If an upstream package changes
the expected text, the script fails instead of applying an ambiguous patch.

## Runtime limits

The Stage 0 smoke test validates imports, a minimal SAPIEN 3 scene, and the
`put_back_block` task contract. It does not claim Stage 1 policy training,
full-dataset generation, or long-horizon benchmark performance.

On this host the RTX 5080 and Vulkan device are visible to `dp3-rmbench`, and
Level 1/2 complete. Warp 1.6.2 emits a non-fatal `cuDeviceGetUuid` driver API
warning during task setup; it is retained in the smoke output and must be
treated as a host-driver warning, not silently converted into a clean log.
