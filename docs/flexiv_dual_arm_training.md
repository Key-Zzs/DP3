# Flexiv Dual Arm DP3 Training

This training path starts after the LeRobot dataset has already been exported
to a DP3 replay-buffer zarr by `tools/export_lerobot_to_dp3_zarr.py`.

The supported zarr contract is:

- `data/state`: `(T, 34)` float32 absolute rotation-6D state
- `data/action`: `(T, 14)` float32 delta action
- `data/point_cloud`: `(T, 2048, 3)` for `xyz`, or `(T, 2048, 6)` for `xyzrgb`
- `meta/episode_ends`: cumulative episode end indices, with the last value equal to `T`
- root attrs: `export_status=complete`, matching `expected_total_frames` and
  `converted_frames`, plus SHA-256 entries for all three training arrays

State semantics are the recorded Flexiv observation vector under
`flexiv_abs_rot6d_v2`:

- left 7 joint positions
- left absolute TCP position `x, y, z`
- left absolute TCP rotation-6D `c0x, c0y, c0z, c1x, c1y, c1z`
- left normalized gripper state
- right 7 joint positions
- right absolute TCP position `x, y, z`
- right absolute TCP rotation-6D `c0x, c0y, c0z, c1x, c1y, c1z`
- right normalized gripper state

The six rotation values are the first two columns of the absolute RDK
world/base TCP rotation matrix: `rot6d = [R[:, 0], R[:, 1]]`. They are not
matrix rows and do not use Home-relative orientation. The action remains the
unchanged 14D left/right delta `xyz + rotvec` command plus two gripper values.

Action semantics are the 14-dimensional delta command:

- left delta EE pose `x, y, z, rx, ry, rz`
- right delta EE pose `x, y, z, rx, ry, rz`
- left gripper command
- right gripper command

## Check A Zarr

Use the `dp3` conda environment for local validation:

```bash
conda run -n dp3 python tools/inspect_dp3_zarr.py \
  --zarr-path /path/to/flexiv_head_xyz_state_abs_rot6d_v2.zarr \
  --expected-state-dim 34 \
  --expected-action-dim 14 \
  --expected-pointcloud-dim 3 \
  --expected-num-points 2048
```

For `xyzrgb`, change `--expected-pointcloud-dim 6`.

The exporter writes to a hidden temporary sibling and only commits the final
zarr after every frame and checksum passes. The inspector and the training
dataset both reject incomplete metadata or checksum mismatches.

The exporter also records `state_names`, `action_names`,
`state_schema=flexiv_abs_rot6d_v2`, `rotation6d_convention=matrix_columns_0_1`,
`action_rotation_representation=rotvec`, source FPS, and source/derived state
SHA-256 provenance. A recognized legacy 28D absolute-rotvec Flexiv dataset can
be converted into a new `*_state_abs_rot6d_v2.zarr` with:

```bash
conda run -n dp3 python tools/export_lerobot_to_dp3_zarr.py \
  --lerobot-path /absolute/path/to/legacy_lerobot \
  --allow-legacy-state-conversion \
  --target-state-schema flexiv_abs_rot6d_v2 \
  --output-zarr /absolute/path/to/flexiv_state_abs_rot6d_v2.zarr
```

The converter requires exact legacy Flexiv names/order; unknown 28D data is
rejected. It never overwrites a v1 Zarr. Existing v1 checkpoints are not
compatible with the v2 runtime and must be retrained.

The exporter also accepts the acquisition-side
`flexiv_abs_rot6d_raw_force_v3` LeRobot source (`observation.state` shape
`(48,)`). It validates the source schema metadata and projects to the unchanged
DP3 target `flexiv_abs_rot6d_v2` `(34,)` state by exact field-name indices. It
drops only these 14 source fields:

```text
left_ee_ext_wrench_in_tcp_raw.fx/fy/fz/mx/my/mz, left_gripper_force
right_ee_ext_wrench_in_tcp_raw.fx/fy/fz/mx/my/mz, right_gripper_force
```

The output remains `data/state=(T,34)` and `data/action=(T,14)`. Its provenance
contains the full 48-name source contract, `source_state_dim=48`,
`state_transform=drop_raw_force_fields_v3_to_v2_by_name`, and
`dropped_state_names`. The raw hash covers all 48 source values and the derived
hash covers the 34D projection; raw force/wrench values are excluded from the
DP3 normalizer, model, checkpoint, training samples, and online agent state.

## Configure Training

All launcher and training parameters are configured in
`3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_train_config.yaml`.
For an XYZ SimpleDP3 run, review at least:

```yaml
defaults:
  - task: real/flexiv_dual_arm_head_xyz

launcher:
  gpu_id: 0
  overwrite: false

algorithm: simple_dp3
task:
  dataset:
    zarr_path: /absolute/path/to/flexiv_head_xyz_state_abs_rot6d_v2.zarr
    max_train_episodes: 90

training:
  device: cuda:0
  seed: 42
  resume: false

logging:
  mode: online

checkpoint:
  save_ckpt: true
```

Use `algorithm: dp3` for the full DP3 model. For XYZRGB, change the default
task to `real/flexiv_dual_arm_head_xyzrgb` and point `zarr_path` at the XYZRGB
dataset. `policy.use_pc_color` and the encoder input channels are derived from
the selected task automatically.

`launcher.gpu_id` is the physical GPU exposed through `CUDA_VISIBLE_DEVICES`.
Keep `training.device: cuda:0` so the process uses that selected device.
Prefer an absolute zarr path because `train.py` changes its working directory
during startup.

## Start Training

After activating the environment, the launcher takes no training arguments:

```bash
conda activate dp3
bash scripts/train_flexiv_dual_arm_dp3.sh
```

The YAML `run_dir` controls the Hydra output directory. Checkpoints are stored
under `<run_dir>/checkpoints/`. If that directory already exists, the launcher
fails before training. Change `run_dir` or `exp_name` for a fresh run. Set
`launcher.overwrite: true` only when the entire old run directory should be
deleted first. To continue an interrupted run, set `training.resume: true`
while keeping `launcher.overwrite: false`; the launcher requires an existing
`checkpoints/latest.ckpt`, and training resumes at the next epoch. The last
configured epoch is always checkpointed even when it is not divisible by
`checkpoint_every`.

## Sanity Overfit

For a short one-episode sanity pass, temporarily reduce these YAML sections:

```yaml
task:
  dataset:
    max_train_episodes: 1

dataloader:
  batch_size: 1
  num_workers: 0
val_dataloader:
  batch_size: 1
  num_workers: 0

training:
  num_epochs: 1
  max_train_steps: 1
  use_ema: false

logging:
  mode: disabled
checkpoint:
  save_ckpt: false
```

Then run the same zero-argument command. Restore the full-training values and
choose a new `exp_name` before the real run.

## Current Boundary

This stage only implements offline training from exported zarr. It does not
load a checkpoint for online execution, read live RGB-D, build live point
clouds, send actions to Flexiv hardware, or run robot motion.

The later inference path must reuse the same state, action, and point-cloud
semantics listed above.
