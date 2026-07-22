"""Pure embodiment smoke environment for the dual Flexiv Stage 2 model.

This environment intentionally contains no task actors, reward, cue, or data
collection path. It loads the generated official URDFs, a table, one fixed head
camera, and exposes deterministic home/joint/gripper/delta-action operations.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
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
    forward = np.asarray(forward, dtype=float)
    forward /= np.linalg.norm(forward)
    left = np.asarray(left, dtype=float)
    left /= np.linalg.norm(left)
    up = np.asarray(up, dtype=float)
    up /= np.linalg.norm(up)
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.stack([forward, left, up], axis=1)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def _look_at_matrix(position: Any, target: Any, up_hint: Any = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Build SAPIEN's camera pose from a world-space look-at target.

    SAPIEN camera poses use columns (forward, left, up).  Computing the frame
    from a target keeps the bounded front/side/top artifacts pointed at the
    workspace instead of relying on approximate hand-written basis vectors.
    """
    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    up_hint = np.asarray(up_hint, dtype=float)
    left = np.cross(up_hint, forward)
    if np.linalg.norm(left) < 1e-8:
        # A top view looks along -z, so the usual world-z up hint is
        # parallel to the optical axis. Use world-y as a stable fallback.
        left = np.cross(np.asarray([0.0, 1.0, 0.0]), forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    up /= np.linalg.norm(up)
    return _camera_matrix(position, forward, left, up)


def _camera_pose_matrix(config: dict[str, Any]) -> np.ndarray:
    if "target_world" in config:
        return _look_at_matrix(
            config["position_world"],
            config["target_world"],
            config.get("up_hint_world", (0.0, 0.0, 1.0)),
        )
    return _camera_matrix(
        config["position_world"],
        config["forward_world"],
        config["left_world"],
        config["up_world"],
    )


class FlexivEmbodimentSmoke(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    _GUI_VIEWER_FOVY_DEG = 50.0

    def __init__(
        self,
        *,
        gui: bool = False,
        seed: int = 0,
        timestep: float | None = None,
        headless: bool | None = None,
        viewer_panels: bool = False,
    ):
        super().__init__()
        self.gui = bool(gui)
        self.viewer_panels = bool(viewer_panels)
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
        self._viewer_imgui_ini_path: Path | None = None

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
        self.rack = self._load_rack()
        self.left_articulation = self._load_articulation("left.urdf", self.left_base_world)
        self.right_articulation = self._load_articulation("right.urdf", self.right_base_world)
        self.left_arm = self._make_arm("left", self.left_articulation)
        self.right_arm = self._make_arm("right", self.right_articulation)
        self._configure_drives()
        self._load_head_camera()
        self.viewer = None
        if self.gui:
            from sapien.utils.viewer import Viewer
            from sapien.utils.viewer.articulation_window import ArticulationWindow
            from sapien.utils.viewer.contact_window import ContactWindow
            from sapien.utils.viewer.control_window import ControlWindow
            from sapien.utils.viewer.entity_window import EntityWindow
            from sapien.utils.viewer.path_window import PathWindow
            from sapien.utils.viewer.render_window import RenderOptionsWindow
            from sapien.utils.viewer.scene_window import SceneWindow
            from sapien.utils.viewer.setting_window import SettingWindow
            from sapien.utils.viewer.transform_window import TransformWindow

            class _FlexivControlWindow(ControlWindow):
                """Use the RMBench viewport bindings for both GUI variants.

                SAPIEN's stock bindings are right-button rotate, middle-button
                pan, and WASD movement.  The embodiment acceptance GUI uses a
                conventional inspection mapping instead: left-button rotate,
                right-button pan, and wheel zoom.  Keyboard movement is
                intentionally disabled so WASD cannot move the camera while a
                user is inspecting the fixed home pose.
                """

                def __init__(self, show_panels: bool):
                    self._show_panels = bool(show_panels)
                    super().__init__()

                def get_ui_windows(self):
                    if not self._show_panels:
                        return []
                    return super().get_ui_windows()

                def _handle_click(self):
                    # The left button is reserved for camera rotation.  Entity
                    # selection is available through the debug windows when
                    # needed, but must not steal the viewport gesture.
                    return None

                def _handle_input_wasd(self):
                    # Explicitly remove the stock keyboard camera movement.
                    return None

                def _handle_input_mouse(self):
                    speed_mod = 0.1 if self.window.shift else 1.0

                    # Left button -> orbit around the fixed workspace center.
                    # Horizontal motion is reversed while vertical motion
                    # follows the original SAPIEN direction.
                    if self.window.mouse_down(0):
                        x, y = self.window.mouse_delta
                        if x != 0 or y != 0:
                            self.arc_camera_controller.rotate_yaw_pitch(
                                -self.rotate_speed * speed_mod * x,
                                self.rotate_speed * speed_mod * y,
                            )
                            self.viewer.set_camera_pose(self.arc_camera_controller.pose)

                    # Right button -> screen-space pan. Move the orbit center
                    # together with the camera so the next orbit uses the
                    # translated workspace point as its pivot.
                    if self.window.mouse_down(1):
                        x, y = self.window.mouse_delta
                        if x != 0 or y != 0:
                            pose = self.arc_camera_controller.pose
                            camera_rotation = Rotation.from_quat(
                                [pose.q[1], pose.q[2], pose.q[3], pose.q[0]]
                            ).as_matrix()
                            pan_delta = (
                                camera_rotation[:, 1] * self.rotate_speed * speed_mod * x
                                + camera_rotation[:, 2] * self.rotate_speed * speed_mod * y
                            )
                            self.arc_camera_controller.set_center(
                                self.arc_camera_controller.center + pan_delta
                            )
                            self.viewer.set_camera_pose(self.arc_camera_controller.pose)

                    # Mouse wheel -> zoom in/out along the optical axis.
                    wx, wy = self.window.mouse_wheel_delta
                    wheel = wx if wx != 0 else wy
                    if wheel != 0:
                        self.arc_camera_controller.zoom(self.scroll_speed * speed_mod * wheel)
                        self.viewer.set_camera_pose(self.arc_camera_controller.pose)

                def _handle_input_f(self):
                    # Keep the orbit center fixed at the workspace center.
                    return None

            # Never reuse the user-global ~/.sapien/imgui.ini.  A stale
            # DPI/monitor-specific dock tree is the source of stacked panels.
            # Viewer writes its canonical default layout to this per-process
            # path, which is removed by close().
            fd, ini_path = tempfile.mkstemp(prefix="rmbench_flexiv_imgui_", suffix=".ini")
            os.close(fd)
            self._viewer_imgui_ini_path = Path(ini_path)
            self._viewer_imgui_ini_path.unlink()
            sapien.render.set_imgui_ini_filename(str(self._viewer_imgui_ini_path))

            plugins = [_FlexivControlWindow(show_panels=self.viewer_panels)]
            if self.viewer_panels:
                plugins = [
                    PathWindow(),
                    ContactWindow(),
                    SettingWindow(),
                    TransformWindow(),
                    RenderOptionsWindow(),
                    plugins[0],
                    SceneWindow(),
                    EntityWindow(),
                    ArticulationWindow(),
                ]
            self.viewer = Viewer(self.renderer, plugins=plugins)
            self.viewer.set_scene(self.scene)
            if self.viewer.control_window is not None:
                self.viewer.control_window.show_camera_linesets = False
            cfg = self.camera_config
            self.viewer.window.set_camera_parameters(
                float(cfg["near_m"]), float(cfg["far_m"]), math.radians(self._GUI_VIEWER_FOVY_DEG)
            )
            self.viewer.set_camera_pose(
                sapien.Pose(_camera_pose_matrix(cfg))
            )
            # Initialize the orbit controller from the configured head-camera
            # pose, using the table/workspace center as its fixed pivot.
            control = self.viewer.control_window
            if control is not None:
                orbit_center = np.asarray(self.geometry["workspace_center_world"], dtype=float)
                camera_position = np.asarray(cfg["position_world"], dtype=float)
                camera_offset = camera_position - orbit_center
                horizontal_radius = float(np.linalg.norm(camera_offset[:2]))
                control.arc_camera_controller.set_center(orbit_center)
                control.arc_camera_controller.set_yaw_pitch(
                    float(np.arctan2(-camera_offset[1], -camera_offset[0])),
                    float(np.arctan2(camera_offset[2], horizontal_radius)),
                )
                control.arc_camera_controller.set_zoom(float(np.linalg.norm(camera_offset)))
                self.viewer.set_camera_pose(control.arc_camera_controller.pose)
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
        table = self.geometry["table"]
        half_size = np.asarray(table["half_size_m"], dtype=float)
        builder.add_box_collision(half_size=half_size.tolist())
        builder.add_box_visual(half_size=half_size.tolist(), material=[0.55, 0.55, 0.55, 1.0])
        self.table = builder.build_static(name="stage2_table")
        self.table.set_pose(sapien.Pose(table["center_world"]))

    def _load_rack(self):
        rack_config = self.geometry["raised_base_rack"]
        rack_path = ASSET_ROOT / rack_config["urdf"]
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        articulations, actors = loader.load_multiple(str(rack_path))
        if articulations:
            rack = articulations[0]
            rack.set_root_pose(sapien.Pose(rack_config["center_world"]))
            self._rack_collision_shapes = [
                shape
                for link in rack.get_links()
                for shape in link.get_collision_shapes()
            ]
        elif actors:
            rack = actors[0]
            rack.set_pose(sapien.Pose(rack_config["center_world"]))
            rigid_body = rack.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            rigid_body.set_kinematic(True)
            self._rack_collision_shapes = rigid_body.get_collision_shapes()
        else:
            raise RuntimeError(f"rack URDF contains no loadable object: {rack_path}")
        return rack

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
        self._configure_arm_collision_groups()

    def _configure_arm_collision_groups(self) -> None:
        """Keep each arm collidable with the static scene but not the other arm.

        The requested 30 cm base spacing plus +/-45 degree roll makes the two
        URDF base collision meshes overlap slightly. Cross-arm collision would
        then inject non-realistic impulses into the fixed-root joint drives.
        """
        static_group = 1
        for side_index, arm in enumerate((self.left_arm, self.right_arm), start=1):
            arm_group = 1 << side_index
            collision_mask = static_group | arm_group
            for link in arm.articulation.get_links():
                for shape in link.get_collision_shapes():
                    _, _, group2, group3 = shape.get_collision_groups()
                    shape.set_collision_groups([arm_group, collision_mask, group2, group3])

        # The rack URDF is geometrically in contact with both bases. The robot
        # roots are fixed to its mounting coordinates, so suppressing the
        # redundant rack-vs-robot contact prevents solver impulses without
        # making either base visually float above the platform.
        for shape in self._rack_collision_shapes:
            shape.set_collision_groups([8, 8, 0, 0])

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
            sapien.Pose(_camera_pose_matrix(cfg))
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
        left_home = self.home_pose["left"]["qpos"]
        right_home = self.home_pose["right"]["qpos"]
        self._set_arm_qpos_immediate(self.left_arm, left_home, 1.0)
        self._set_arm_qpos_immediate(self.right_arm, right_home, 1.0)
        self._settle(settle_steps)
        # Physics settling is useful for contacts and render stability, but
        # the public home contract must remain bit-for-bit aligned with the
        # real runtime home joints. Reassert the commanded home after settling.
        self._set_arm_qpos_immediate(self.left_arm, left_home, 1.0)
        self._set_arm_qpos_immediate(self.right_arm, right_home, 1.0)
        self.scene.update_render()

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
        target = self.geometry["workspace_center_world"]
        views = {
            "front": ([-1.65, target[1], 1.35], target, [0.0, 0.0, 1.0]),
            "side": ([target[0], 1.80, 1.35], target, [0.0, 0.0, 1.0]),
            "top": ([target[0], target[1], 2.65], target, [0.0, 1.0, 0.0]),
        }
        if name == "head-camera":
            return self.capture_head()
        if name not in views:
            raise ValueError(f"unknown view {name}")
        position, target, up_hint = views[name]
        camera = self.scene.add_camera(name=f"acceptance_{name}", width=320, height=240, fovy=math.radians(45), near=0.05, far=5.0)
        camera.entity.set_pose(sapien.Pose(_look_at_matrix(position, target, up_hint)))
        # Cameras added after the last render update otherwise return the
        # initialized gray color buffer and an entirely invalid depth image.
        self.scene.update_render()
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
        if self._viewer_imgui_ini_path is not None:
            self._viewer_imgui_ini_path.unlink(missing_ok=True)
            self._viewer_imgui_ini_path = None
        self.scene = None
        self.renderer = None
        self.engine = None
