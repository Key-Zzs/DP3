from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_flexiv_dp3_perception_only import _camera_module  # noqa: E402


class _FakeFrame:
    def __init__(self, data: np.ndarray, *, timestamp: float = 12.5, number: int = 7) -> None:
        self.data = data
        self.timestamp = timestamp
        self.number = number

    def __bool__(self) -> bool:
        return True

    def get_data(self) -> np.ndarray:
        return self.data

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_frame_number(self) -> int:
        return self.number


class _FakeFrameset:
    def __init__(self, color: _FakeFrame, depth: _FakeFrame) -> None:
        self.color = color
        self.depth = depth

    def get_color_frame(self) -> _FakeFrame:
        return self.color

    def get_depth_frame(self) -> _FakeFrame:
        return self.depth


class _FakeIRFrameset(_FakeFrameset):
    def __init__(
        self,
        color: _FakeFrame,
        depth: _FakeFrame,
        left: _FakeFrame,
        right: _FakeFrame,
    ) -> None:
        super().__init__(color, depth)
        self.left = left
        self.right = right

    def get_infrared_frame(self, stream_index: int) -> _FakeFrame:
        return self.left if stream_index == 1 else self.right


def test_copy_frame_data_owns_contiguous_memory() -> None:
    camera_mod = _camera_module()
    source = np.arange(12, dtype=np.uint16).reshape(3, 4)

    copied = camera_mod._copy_frame_data(_FakeFrame(source))
    source[:] = 0

    assert copied.flags.owndata
    assert copied.flags.c_contiguous
    assert copied.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


def test_rgbd_frame_survives_sdk_buffer_reuse() -> None:
    camera_mod = _camera_module()
    camera = camera_mod.RealSenseCamera(
        camera_mod.RealSenseCameraConfig(
            serial_number_or_name="1234",
            fps=30,
            width=2,
            height=2,
            color_mode=camera_mod.ColorMode.RGB,
            use_depth=True,
            use_ir=False,
            rotation=camera_mod.Cv2Rotation.NO_ROTATION,
        )
    )
    color = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.array([[100, 200], [300, 400]], dtype=np.uint16)

    result = camera._frameset_to_rgbd_ir(
        _FakeFrameset(_FakeFrame(color), _FakeFrame(depth))
    )
    color[:] = 0
    depth[:] = 0

    assert result["rgb"].flags.owndata
    assert result["depth"].flags.owndata
    assert result["rgb"].flags.c_contiguous
    assert result["depth"].flags.c_contiguous
    assert int(result["rgb"].sum()) == 66
    assert result["depth"].tolist() == [[100, 200], [300, 400]]


def test_ffs_rgbd_frame_contains_same_frameset_ir_identity() -> None:
    camera_mod = _camera_module()
    camera = camera_mod.RealSenseCamera(
        camera_mod.RealSenseCameraConfig(
            serial_number_or_name="1234",
            fps=30,
            width=2,
            height=2,
            color_mode=camera_mod.ColorMode.RGB,
            use_depth=True,
            use_ir=True,
            rotation=camera_mod.Cv2Rotation.NO_ROTATION,
        )
    )
    result = camera._frameset_to_rgbd_ir(
        _FakeIRFrameset(
            _FakeFrame(np.zeros((2, 2, 3), dtype=np.uint8), timestamp=12.5, number=7),
            _FakeFrame(np.ones((2, 2), dtype=np.uint16), timestamp=12.5, number=7),
            _FakeFrame(np.zeros((2, 2), dtype=np.uint8), timestamp=12.5, number=7),
            _FakeFrame(np.ones((2, 2), dtype=np.uint8), timestamp=12.5, number=7),
        )
    )

    assert result["left_ir"].shape == (2, 2)
    assert result["right_ir"].shape == (2, 2)
    assert result["left_ir_timestamp"] == result["right_ir_timestamp"] == 12.5
    assert result["left_ir_frame_index"] == result["right_ir_frame_index"] == 7


class _WarmupCamera:
    def __init__(self, depths: list[np.ndarray]) -> None:
        self.depths = iter(depths)

    def read_rgbd_ir(self, timeout_ms: int):
        del timeout_ms
        return {"depth": next(self.depths), "rgb": np.zeros((2, 2, 3), dtype=np.uint8)}


def _flexiv_runtime(*, minimum_ratio: float):
    camera_mod = _camera_module()
    config_mod = importlib.import_module(f"{camera_mod.__package__}.config_flexiv")
    flexiv_mod = importlib.import_module(f"{camera_mod.__package__}.flexiv_dual_arm")
    config = config_mod.FlexivDualArmConfig(
        save_depth_sidecar=True,
        camera_warmup_attempts=2,
        camera_warmup_frames=3,
        camera_warmup_stability_window=2,
        camera_min_valid_depth_ratio=minimum_ratio,
        camera_max_valid_depth_ratio_range=0.1,
    )
    return flexiv_mod.FlexivDualArm(config)


def test_runtime_warmup_accepts_stable_depth() -> None:
    robot = _flexiv_runtime(minimum_ratio=0.75)
    good = np.array([[100, 200], [300, 0]], dtype=np.uint16)

    robot._warmup_camera("head_rgb", _WarmupCamera([good.copy() for _ in range(3)]))

    assert robot._latest_frames["head_rgb"]["depth"].tolist() == good.tolist()


def test_runtime_warmup_rejects_low_valid_depth_ratio() -> None:
    robot = _flexiv_runtime(minimum_ratio=0.75)
    poor = np.array([[100, 0], [0, 0]], dtype=np.uint16)

    with pytest.raises(RuntimeError, match="depth valid ratio median"):
        robot._warmup_camera("head_rgb", _WarmupCamera([poor.copy() for _ in range(3)]))
