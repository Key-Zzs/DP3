# Flexiv Dual Arm DP3 Training

This training path starts after the LeRobot dataset has already been exported
to a DP3 replay-buffer zarr by `tools/export_lerobot_to_dp3_zarr.py`.

The supported zarr contract is:

- `data/state`: `(T, 28)` float32
- `data/action`: `(T, 14)` float32 delta action
- `data/point_cloud`: `(T, 1024, 3)` for `xyz`, or `(T, 1024, 6)` for `xyzrgb`
- `meta/episode_ends`: cumulative episode end indices, with the last value equal to `T`

State semantics are the recorded Flexiv observation vector:

- left 7 joint positions
- left EE pose `x, y, z, rx, ry, rz`
- left normalized gripper state
- right 7 joint positions
- right EE pose `x, y, z, rx, ry, rz`
- right normalized gripper state

Action semantics are the 14-dimensional delta command:

- left delta EE pose `x, y, z, rx, ry, rz`
- right delta EE pose `x, y, z, rx, ry, rz`
- left gripper command
- right gripper command

## Check A Zarr

Use the `dp3` conda environment for local validation:

```bash
conda run -n dp3 python tools/inspect_dp3_zarr.py \
  --zarr-path /path/to/flexiv_head_xyz.zarr \
  --expected-state-dim 28 \
  --expected-action-dim 14 \
  --expected-pointcloud-dim 3 \
  --expected-num-points 1024
```

For `xyzrgb`, change `--expected-pointcloud-dim 6`.

## Train XYZ

```bash
conda run -n dp3 bash scripts/train_flexiv_dual_arm_dp3.sh \
  xyz \
  /path/to/flexiv_head_xyz.zarr \
  simple_dp3 \
  0 \
  42
```

Equivalent direct Hydra override:

```bash
cd 3D-Diffusion-Policy
HYDRA_FULL_ERROR=1 python train.py --config-name=simple_dp3.yaml \
  task=real/flexiv_dual_arm_head_xyz \
  task.dataset.zarr_path=/path/to/flexiv_head_xyz.zarr \
  hydra.run.dir=3D-Diffusion-Policy/outputs/flexiv_dual_arm_head_xyz-simple_dp3_seed42 \
  training.device=cuda:0 \
  logging.mode=offline
```

## Train XYZRGB

Use the `xyzrgb` task and enable point-cloud color:

```bash
conda run -n dp3 bash scripts/train_flexiv_dual_arm_dp3.sh \
  xyzrgb \
  /path/to/flexiv_head_xyzrgb.zarr \
  simple_dp3 \
  0 \
  42
```

The wrapper passes `policy.use_pc_color=true` and configures the point-cloud
encoder for 6 input channels.

## Sanity Overfit

Run a short one-episode sanity pass before full training:

```bash
DEBUG=True \
SAVE_CKPT=False \
WANDB_MODE=disabled \
MAX_TRAIN_EPISODES=1 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
conda run -n dp3 bash scripts/train_flexiv_dual_arm_dp3.sh \
  xyz \
  /path/to/flexiv_head_xyz.zarr \
  simple_dp3 \
  0 \
  42 \
  training.num_epochs=1 \
  training.max_train_steps=1 \
  training.use_ema=False \
  training.sample_every=999999 \
  policy.num_inference_steps=1 \
  hydra.run.dir=3D-Diffusion-Policy/outputs/dp3_flexiv_sanity_xyz
```

For a smaller non-debug run:

```bash
cd 3D-Diffusion-Policy
HYDRA_FULL_ERROR=1 python train.py --config-name=simple_dp3.yaml \
  task=real/flexiv_dual_arm_head_xyz \
  task.dataset.zarr_path=/path/to/flexiv_head_xyz.zarr \
  task.dataset.max_train_episodes=1 \
  dataloader.batch_size=1 \
  dataloader.num_workers=0 \
  val_dataloader.batch_size=1 \
  val_dataloader.num_workers=0 \
  training.num_epochs=1 \
  training.max_train_steps=1 \
  training.device=cuda:0 \
  logging.mode=disabled \
  checkpoint.save_ckpt=False \
  hydra.run.dir=3D-Diffusion-Policy/outputs/dp3_flexiv_sanity_xyz_direct
```

## Full Training

For full training, keep `DEBUG=False` and `SAVE_CKPT=True`:

```bash
SAVE_CKPT=True WANDB_MODE=offline conda run -n dp3 bash \
  scripts/train_flexiv_dual_arm_dp3.sh \
  xyz \
  /path/to/flexiv_head_xyz.zarr \
  simple_dp3 \
  0 \
  42
```

Checkpoints are written under the Hydra output directory. The wrapper default
is this repository-relative path:

```text
outputs/<exp_name>_seed<seed>/checkpoints/
```

The wrapper refuses to start if the target output directory already exists.
Use a different `RUN_DIR`/`EXP_NAME`, or add `--overwrite` to delete the entire
target output directory before training. This avoids ambiguous leftover
checkpoints from an older run.

`train.py` changes its working directory during startup, so prefer absolute
`zarr_path` values.

## Current Boundary

This stage only implements offline training from exported zarr. It does not
load a checkpoint for online execution, read live RGB-D, build live point
clouds, send actions to Flexiv hardware, or run robot motion.

The later inference path must reuse the same state, action, and point-cloud
semantics listed above.
