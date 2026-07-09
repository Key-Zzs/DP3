# Flexiv Dual Arm Inference TODO

This is a boundary note for the later online inference stage. It is not an
implementation and must not be used to command the robot.

Planned inference work:

- Load a trained DP3 or Simple-DP3 checkpoint.
- Reconstruct the same observation contract used for training:
  `obs["point_cloud"]` plus `obs["agent_pos"]`.
- Read the head RGB-D stream and pass frames through
  `PointCloudBuilder.from_live_frame`.
- Preserve the training point-cloud mode: `xyz` uses `(1024, 3)`, `xyzrgb`
  uses `(1024, 6)`.
- Convert policy output to the same 14-dimensional delta action semantics:
  left delta EE pose, right delta EE pose, left gripper command,
  right gripper command.
- Add safety filtering before any command reaches the Flexiv interface:
  delta clipping, low-speed mode, shadow mode, watchdog timeout, and human
  takeover.
- Only after those checks, map the 14-dimensional action to
  `third_party/real/dual_flexiv_rizon4s/interface/FlexivDualArm.send_action`.

Out of scope for the current training stage:

- Real robot online control scripts.
- Calls that move Flexiv arms or grippers.
- FFS or FoundationStereo.
- Three-view fusion.
- New world-frame transforms.
