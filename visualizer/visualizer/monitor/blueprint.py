"""DP3 Rerun Blueprint construction.

The Rerun SDK is imported only when this child-only helper is called.
"""

from __future__ import annotations

import importlib
from typing import Any, Mapping


def build_dp3_blueprint(
    *,
    rr_module: Any | None = None,
    include_predicted_path: bool = False,
    show_stage_pointclouds: bool = False,
    state_fields: tuple[str, ...] = (),
    action_fields: tuple[str, ...] = (),
) -> Any:
    rr = rr_module or importlib.import_module("rerun")
    blueprint_module = getattr(rr, "blueprint", None)
    if blueprint_module is None:
        return None
    rrb = blueprint_module
    views = []
    spatial = getattr(rrb, "Spatial3DView", None)
    spatial2d = getattr(rrb, "Spatial2DView", None)
    if spatial is not None:
        views.append(spatial(name="Sampled point cloud", origin="/observation/point_cloud/sampled"))
        # Keep expensive diagnostic stages available without making them part
        # of the default visible layout.
        views.append(
            spatial(
                name="Raw point cloud (optional)",
                origin="/observation/point_cloud/raw",
                visible=show_stage_pointclouds,
            )
        )
        views.append(
            spatial(
                name="Cropped point cloud (optional)",
                origin="/observation/point_cloud/cropped",
                visible=show_stage_pointclouds,
            )
        )
        if include_predicted_path:
            views.append(spatial(name="Predicted delta path", origin="/policy/predicted_delta_path"))
    if spatial2d is not None:
        views.extend(
            [
                spatial2d(name="Head RGB", origin="/observation/camera/head/rgb"),
                spatial2d(name="Head depth", origin="/observation/camera/head/depth"),
            ]
        )
    time_view = getattr(rrb, "TimeSeriesView", None)
    if time_view is not None:
        for side in ("left", "right"):
            state_groups = _state_groups(state_fields, side)
            views.extend(
                [
                    _time_view(time_view, f"State | {side} joints [rad]", "/observation/state", state_groups["joints"]),
                    _time_view(time_view, f"State | {side} TCP xyz [m]", "/observation/state", state_groups["position"]),
                    _time_view(
                        time_view,
                        f"State | {side} rotation-6D [unitless]",
                        "/observation/state",
                        state_groups["rotation"],
                    ),
                    _time_view(time_view, f"State | {side} gripper [0..1]", "/observation/state", state_groups["gripper"]),
                ]
            )

        action_groups = _action_groups(action_fields)
        action_prefixes = (
            "/control/action_selected_raw/",
            "/control/action_filtered/",
            "/robot/action_commanded/",
        )
        # These views deliberately combine raw/filtered/commanded curves only
        # when they share a physical unit. Entity names retain the three-stage
        # distinction in the legend.
        views.extend(
            [
                _time_view(
                    time_view,
                    "Actions | delta xyz [m]",
                    "/",
                    _prefixed_paths(action_prefixes, action_groups["position"]),
                ),
                _time_view(
                    time_view,
                    "Actions | delta rotvec [rad]",
                    "/",
                    _prefixed_paths(action_prefixes, action_groups["rotation"]),
                ),
                _time_view(
                    time_view,
                    "Actions | grippers [0..1]",
                    "/",
                    _prefixed_paths(action_prefixes, action_groups["gripper"]),
                ),
                _time_view(
                    time_view,
                    "Policy horizon | delta xyz [m]",
                    "/policy/prediction/horizon",
                    tuple(f"/policy/prediction/horizon/{name}" for name in action_groups["position"]),
                ),
                _time_view(
                    time_view,
                    "Policy horizon | delta rotvec [rad]",
                    "/policy/prediction/horizon",
                    tuple(f"/policy/prediction/horizon/{name}" for name in action_groups["rotation"]),
                ),
                _time_view(
                    time_view,
                    "Policy horizon | grippers [0..1]",
                    "/policy/prediction/horizon",
                    tuple(f"/policy/prediction/horizon/{name}" for name in action_groups["gripper"]),
                ),
                _time_view(
                    time_view,
                    "Policy prediction [index]",
                    "/policy/prediction",
                    ("/policy/prediction/id", "/policy/prediction/selected_index"),
                ),
                _time_view(time_view, "Timing [ms]", "/timing", ()),
                _time_view(time_view, "Telemetry health [count / 0-1]", "/monitor", ()),
            ]
        )
    text_view = getattr(rrb, "TextLogView", None)
    if text_view is not None:
        views.append(text_view(name="Events", origin="/events"))
    blueprint_cls = getattr(rrb, "Blueprint", None)
    if blueprint_cls is None:
        return None
    container = getattr(rrb, "Grid", None)
    root = container(*views, grid_columns=2, name="DP3 monitor") if container is not None else None
    parts = list(tuple([root]) if root is not None else tuple(views))
    time_panel = getattr(rrb, "TimePanel", None)
    if time_panel is not None:
        # The telemetry process starts before robot/camera initialization, so
        # the first image and point-cloud samples can arrive several seconds
        # after the recording begins.  Keep the Viewer cursor on the newest
        # log-time sample instead of leaving it paused in that empty interval.
        try:
            parts.append(time_panel(timeline="log_time", play_state="following"))
        except TypeError:
            # Keep older SDK/fake Blueprint modules usable; 0.34.1 takes both
            # arguments and is the deployment version tested by this project.
            parts.append(time_panel())
    try:
        return blueprint_cls(*parts, collapse_panels=False)
    except TypeError:
        return blueprint_cls(*parts)


def _time_view(view_type: Any, name: str, origin: str, contents: tuple[str, ...]) -> Any:
    if contents:
        try:
            return view_type(name=name, origin=origin, contents=contents)
        except TypeError:
            pass
    return view_type(name=name, origin=origin)


def _state_groups(state_fields: tuple[str, ...], side: str) -> dict[str, tuple[str, ...]]:
    names = tuple(name for name in state_fields if name.startswith(f"{side}_"))
    groups = {
        "joints": tuple(name for name in names if name.startswith(f"{side}_joint_")),
        "position": tuple(name for name in names if name.startswith(f"{side}_ee_pose.")),
        "rotation": tuple(name for name in names if name.startswith(f"{side}_ee_rotation_6d.")),
        "gripper": tuple(name for name in names if name == f"{side}_gripper_state_norm"),
    }
    return {key: tuple(f"/observation/state/{name}" for name in values) for key, values in groups.items()}


def _action_groups(action_fields: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        "position": tuple(name for name in action_fields if name.rsplit(".", 1)[-1] in {"x", "y", "z"}),
        "rotation": tuple(name for name in action_fields if name.rsplit(".", 1)[-1] in {"rx", "ry", "rz"}),
        "gripper": tuple(name for name in action_fields if name.endswith("_gripper_cmd")),
    }


def _prefixed_paths(prefixes: tuple[str, ...], names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(prefix + name for prefix in prefixes for name in names)


def static_metadata(*, state_fields: tuple[str, ...], action_fields: tuple[str, ...], shapes: Mapping[str, Any], camera_contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state_field_names": list(state_fields),
        "action_field_names": list(action_fields),
        "pointcloud_shapes": dict(shapes),
        "camera_contract": dict(camera_contract),
    }


__all__ = ["build_dp3_blueprint", "static_metadata"]
