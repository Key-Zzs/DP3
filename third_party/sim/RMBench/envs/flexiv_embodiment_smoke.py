"""Pure embodiment smoke environment for the dual Flexiv Stage 2 model.

This environment intentionally contains no task actors, reward, cue, or data
collection path. It loads the generated official URDFs, a table, one fixed head
camera, and exposes deterministic home/joint/gripper/delta-action operations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import sapien.core as sapien
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = REPO_ROOT / "3D-Diffusion-Policy"
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from diffusion_policy_3d.sim.flexiv.action_adapter import FlexivActionAdapter
from diffusion_policy_3d.sim.flexiv.embodiment import DampedLeastSquaresIK, SapienFlexivArm
from diffusion_policy_3d.sim.flexiv.frames import RigidPose, base_to_world_pose, rotation_to_quat_wxyz
from diffusion_policy_3d.sim.flexiv.gripper_adapter import GripperMapping
from diffusion_policy_3d.sim.flexiv.state_adapter import FlexivStateAdapter


ASSET_ROOT = REPO_ROOT / "sim_assets" / "flexiv_rizon4s_dual_gn01"
BUNDLE_ROOT = REPO_ROOT / "third_party" / "sim" / "RMBench" / "assets" / "embodiments" / "flexiv-rizon4s-dual-gn01"


def _pose_from_config(payload: dict[str, Any]) -> RigidPose:
    return RigidPose(np.asarray(payload["position"], dtype=float), Rotation.from_euler("xyz", payload["rpy_rad"]).as_matrix())


def _camera_matrix(position: Any, forward: Any, left: Any, up: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.stack(
        [np.asarray(forward, dtype=float), np.asarray(left, dtype=float), np.asarray(up, dtype=float)], axis=1
    )
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


class FlexivEmbodimentSmoke(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, *, gui: bool = False, seed: int = 0, timestep: float | None = None, headless: bool | None = None):
        super().__init__()
        self.gui = bool(gui)
        self.seed_value = int(seed)
        self.rng = np.random.default_rng(self.seed_value)
        self.headless = not self.gui if headless is None else bool(headless)
        if not BUNDLE_ROOT.exists():
            raise FileNotFoundError(
                f"generated runtime bundle is missing: {BUNDLE_ROOT}; run scripts/rmbench/flexiv/bootstrap_description.sh"
            )
        self.manifest = json.loads((BUNDLE_ROOT / "generation_manifest.json").read_text(encoding="utf-8"))
        self.geometry = yaml.safe_load((ASSET_ROOT / "installation_geometry.yaml").read_text(encoding="utf-8"))
        self.camera_config = yaml.safe_load((ASSET_ROOT / "camera_mount.yaml").read_text(encoding="utf-8"))
        self.overrides = yaml.safe_load((ASSET_ROOT / "simulation_overrides.yaml").read_text(encoding="utf-8"))
        self.home_pose = yaml.safe_load((ASSET_ROOT / "home_pose.yaml").read_text(encoding="utf-8"))
        self.left_base_world = _pose_from_config(self.geometry["left_base_pose_world"])
        self.right_base_world = _pose_from_config(self.geometry["right_base_pose_world"])

        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.scene = self.engine.create_scene()
        physics = self.overrides["physics"]
        self.scene.set_timestep(float(timestep if timestep is not None else physics["timestep_s"]))
        self.scene.default_physical_material = self.scene.create_physical_material(
            float(physics["static_friction"]), float(physics["dynamic_friction"]), float(physics["restitution"])
        )
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0.0, 0.5, -1.0], [0.8, 0.8, 0.8], shadow=True)
        self.scene.add_point_light([1.0, -0.8, 2.0], [1.0, 1.0, 1.0], shadow=True)
        self._build_table()
        self.left_articulation = self._load_articulation("left.urdf", self.left_base_world)
        self.right_articulation = self._load_articulation("right.urdf", self.right_base_world)
        self.left_arm = self._make_arm("left", self.left_articulation)
        self.right_arm = self._make_arm("right", self.right_articulation)
        self._configure_drives()
        self._load_head_camera()
        self.viewer = None
        if self.gui:
            from sapien.utils.viewer import Viewer

            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(x=0.0, y=-1.6, z=1.45)
            self.viewer.set_camera_rpy(r=0.0, p=-0.65, y=1.57)
        self.action_adapter = FlexivActionAdapter()
        self.left_ik = DampedLeastSquaresIK(self.left_arm, **self._ik_options())
        self.right_ik = DampedLeastSquaresIK(self.right_arm, **self._ik_options())
        self.state_adapter = FlexivStateAdapter(
            left_articulation=self.left_articulation,
            right_articulation=self.right_articulation,
            left_arm_joints=self.manifest["sides"]["left"]["arm_joints"],
            right_arm_joints=self.manifest["sides"]["right"]["arm_joints"],
            left_tcp_link=self.manifest["sides"]["left"]["tcp_link"],
            right_tcp_link=self.manifest["sides"]["right"]["tcp_link"],
            left_gripper=self.left_arm.gripper,
            right_gripper=self.right_arm.gripper,
            left_base_world=self.left_base_world,
            right_base_world=self.right_base_world,
        )
        self.reset(seed=self.seed_value)

    def _ik_options(self) -> dict[str, Any]:
        drives = self.overrides["drives"]
        return {
            "damping": float(drives["ik_damping"]),
            "max_iterations": int(drives["ik_max_iterations"]),
            "position_tolerance": float(drives["ik_position_tolerance_m"]),
            "rotation_tolerance": float(drives["ik_rotation_tolerance_rad"]),
        }

    def _build_table(self) -> None:
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.95, 0.70, 0.04])
        builder.add_box_visual(half_size=[0.95, 0.70, 0.04], material=[0.55, 0.55, 0.55, 1.0])
        self.table = builder.build_static(name="stage2_table")
        self.table.set_pose(sapien.Pose([0.0, 0.05, 0.70]))

    def _load_articulation(self, filename: str, base_pose: RigidPose):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        articulation = loader.load(str(BUNDLE_ROOT / filename))
        articulation.set_root_pose(sapien.Pose(base_pose.position, rotation_to_quat_wxyz(base_pose.rotation)))
        return articulation

    def _make_arm(self, side: str, articulation: Any) -> SapienFlexivArm:
        payload = self.manifest["sides"][side]
        arm = SapienFlexivArm(
            articulation,
            arm_joints=list(payload["arm_joints"]),
            tcp_link=payload["tcp_link"],
            gripper=GripperMapping.from_manifest(payload["gripper"]),
        )
        return arm

    def _configure_drives(self) -> None:
        drives = self.overrides["drives"]
        for arm in (self.left_arm, self.right_arm):
            for joint in arm.articulation.get_active_joints():
                if joint.get_name() in arm.arm_joints:
                    joint.set_drive_property(float(drives["joint_stiffness"]), float(drives["joint_damping"]))
                else:
                    joint.set_drive_property(float(drives["gripper_stiffness"]), float(drives["gripper_damping"]))

    def _load_head_camera(self) -> None:
        cfg = self.camera_config
        self.head_camera = self.scene.add_camera(
            name="head_camera",
            width=int(cfg["resolution"][0]),
            height=int(cfg["resolution"][1]),
            fovy=math.radians(float(cfg["fovy_deg"])),
            near=float(cfg["near_m"]),
            far=float(cfg["far_m"]),
        )
        self.head_camera.entity.set_pose(
            sapien.Pose(_camera_matrix(cfg["position_world"], cfg["forward_world"], cfg["left_world"], cfg["up_world"]))
        )
        self.static_camera_names = ["head_camera"]

    def _set_arm_qpos_immediate(self, arm: SapienFlexivArm, q_arm: list[float], gripper_value: float) -> None:
        qpos = arm.qpos()
        qpos[arm.arm_indices] = np.asarray(q_arm, dtype=float)
        targets = arm.gripper.normalized_to_joint_targets(gripper_value)
        for name, value in targets.items():
            qpos[arm._q_index[name]] = value
        arm.articulation.set_qpos(qpos)
        arm.set_arm_drive_targets(np.asarray(q_arm, dtype=float))
        arm.set_gripper(gripper_value)

    def home(self, *, settle_steps: int = 240) -> None:
        self._set_arm_qpos_immediate(self.left_arm, self.home_pose["left"]["qpos"], 1.0)
        self._set_arm_qpos_immediate(self.right_arm, self.home_pose["right"]["qpos"], 1.0)
        self._settle(settle_steps)

    def _settle(self, steps: int) -> None:
        for _ in range(int(steps)):
            for articulation in (self.left_articulation, self.right_articulation):
                articulation.set_qf(
                    articulation.compute_passive_force(
                        gravity=True, coriolis_and_centrifugal=True
                    )
                )
            self.scene.step()
        self.scene.update_render()

    def apply_action(self, action: Any, *, settle_steps: int = 12, clip: bool = True) -> dict[str, Any]:
        left_current = self.state_adapter._tcp_base(self.left_articulation, self.left_arm.tcp_link, self.left_base_world)
        right_current = self.state_adapter._tcp_base(self.right_articulation, self.right_arm.tcp_link, self.right_base_world)
        targets = self.action_adapter.decode(action, left_current_base=left_current, right_current_base=right_current, clip=clip)
        left_world = base_to_world_pose(self.left_base_world, targets.left_base)
        right_world = base_to_world_pose(self.right_base_world, targets.right_base)
        left_result = self.left_ik.solve(left_world)
        right_result = self.right_ik.solve(right_world)
        record = {
            **targets.diagnostics,
            "target_tcp_base": {"left": {"position": targets.left_base.position.tolist()}, "right": {"position": targets.right_base.position.tolist()}},
            "target_tcp_world": {"left": {"position": left_world.position.tolist()}, "right": {"position": right_world.position.tolist()}},
            "ik": {
                "left": {**left_result.__dict__, "qpos": left_result.qpos.tolist()},
                "right": {**right_result.__dict__, "qpos": right_result.qpos.tolist()},
            },
        }
        if not left_result.success or not right_result.success:
            record["status"] = "FAIL"
            record["reason"] = "atomic dual-arm action rejected because one IK solve failed"
            return record
        self.left_articulation.set_qpos(left_result.qpos)
        self.right_articulation.set_qpos(right_result.qpos)
        self.left_arm.set_arm_drive_targets(left_result.qpos[self.left_arm.arm_indices])
        self.right_arm.set_arm_drive_targets(right_result.qpos[self.right_arm.arm_indices])
        self.left_arm.set_gripper(float(targets.clipped_action[12]))
        self.right_arm.set_gripper(float(targets.clipped_action[13]))
        self._settle(settle_steps)
        record["applied_joints"] = {"left": self.left_arm.arm_qpos().tolist(), "right": self.right_arm.arm_qpos().tolist()}
        record["status"] = "PASS"
        return record

    def joint_sweep(self, *, side: str, joint_index: int, amplitude: float = 0.12, frames: int = 20) -> list[np.ndarray]:
        arm = self.left_arm if side == "left" else self.right_arm
        base = arm.arm_qpos()
        frames_out = []
        for scalar in np.linspace(-amplitude, amplitude, int(frames)):
            q = base.copy()
            q[joint_index] += scalar
            arm.set_arm_drive_targets(q)
            self._settle(4)
            frames_out.append(self._head_rgb())
        arm.set_arm_drive_targets(base)
        self._settle(12)
        return frames_out

    def gripper_cycle(self, *, side: str, frames: int = 20) -> list[np.ndarray]:
        arm = self.left_arm if side == "left" else self.right_arm
        frames_out = []
        for value in np.r_[np.linspace(0.0, 1.0, frames), np.linspace(1.0, 0.0, frames)]:
            arm.set_gripper(float(value))
            self._settle(4)
            frames_out.append(self._head_rgb())
        return frames_out

    def _head_rgb(self) -> np.ndarray:
        self.head_camera.take_picture()
        rgba = (self.head_camera.get_picture("Color") * 255.0).clip(0, 255).astype(np.uint8)
        return rgba[..., :3]

    def capture_head(self) -> tuple[np.ndarray, np.ndarray]:
        self.head_camera.take_picture()
        color = (self.head_camera.get_picture("Color") * 255.0).clip(0, 255).astype(np.uint8)[..., :3]
        position = self.head_camera.get_picture("Position")
        depth = (-position[..., 2] * 1000.0).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        return color, depth

    def capture_view(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        views = {
            "front": ([0.0, -1.45, 1.25], [0.0, 0.75, -0.35], [-1.0, 0.0, 0.0], [0.0, 0.35, 0.75]),
            "side": ([1.55, 0.05, 1.25], [-0.95, 0.0, -0.25], [0.0, -1.0, 0.0], [0.25, 0.0, 0.95]),
            "top": ([0.0, 0.1, 2.45], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        }
        if name == "head-camera":
            return self.capture_head()
        if name not in views:
            raise ValueError(f"unknown view {name}")
        position, forward, left, up = views[name]
        camera = self.scene.add_camera(name=f"acceptance_{name}", width=320, height=240, fovy=math.radians(45), near=0.05, far=5.0)
        camera.entity.set_pose(sapien.Pose(_camera_matrix(position, forward, left, up)))
        camera.take_picture()
        color = (camera.get_picture("Color") * 255).clip(0, 255).astype(np.uint8)[..., :3]
        depth = (-camera.get_picture("Position")[..., 2] * 1000.0).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        return color, depth

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)
        self.home(settle_steps=600)
        return self.state_adapter.state(), {"seed": self.seed_value, "camera_names": self.static_camera_names}

    def state(self) -> np.ndarray:
        return self.state_adapter.state()

    def step(self, action: Any):
        result = self.apply_action(action)
        observation = self.state_adapter.state()
        reward = 0.0
        terminated = result["status"] != "PASS"
        return observation, reward, terminated, False, result

    def render(self):
        if self.gui and hasattr(self, "viewer"):
            self.viewer.render()
        return self._head_rgb()

    def close(self) -> None:
        if getattr(self, "viewer", None) is not None:
            self.viewer.close()
            self.viewer = None
        self.scene = None
        self.renderer = None
        self.engine = None
