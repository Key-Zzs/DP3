from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import debug_lerobot_pointcloud_stages as debug_lerobot  # noqa: E402
import debug_zarr_pointcloud_stages as debug_zarr  # noqa: E402
import run_flexiv_dp3_perception_only as perception_only  # noqa: E402


class _FakeSource:
    storage = "zarr_v2"

    def __init__(self, tmp_path: Path):
        calibration_path = tmp_path / "meta/realsense_calibration.json"
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text("{}", encoding="utf-8")
        calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        row = {
            "global_frame_index": 7,
            "head_rgbd_timestamp": 10.5,
            "head_rgbd_reused": False,
            "episode_index": 0,
            "frame_index": 7,
            "index": 7,
        }
        pair = debug_lerobot.exporter.rgbd_source.IRPair(
            left_ir=np.zeros((480, 640), dtype=np.uint8),
            right_ir=np.ones((480, 640), dtype=np.uint8),
            timestamp=row["head_rgbd_timestamp"],
            reused=False,
            calibration_path=calibration_path,
            calibration_sha256=calibration_sha,
        )
        self.frame = SimpleNamespace(
            row_index=0,
            row=row,
            source_path=tmp_path / "data/chunk-000/file-000.parquet",
            depth=np.full((480, 640), 1000, dtype=np.uint16),
            ir_pair=pair,
        )
        self.calibration = {}
        self.include_ir_calls: list[bool] = []

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "source_sidecar_path": "sidecars/realsense.zarr",
            "source_sidecar_manifest_path": "meta/rgbd_sidecar.json",
        }

    def validate_join(self, data_paths, *, camera: str, batch_size: int = 128) -> None:
        del data_paths, camera, batch_size

    def read_frame_at(self, data_paths, *, camera, row_index, columns, include_ir=False):
        del data_paths, camera, row_index, columns
        self.include_ir_calls.append(bool(include_ir))
        return self.frame


class _FakeBuilder:
    frames: list[dict[str, object]] = []

    @classmethod
    def from_yaml(cls, path: Path):
        del path
        return cls()

    def build_stages(self, frame):
        type(self).frames.append(frame)
        sampled = torch.zeros((4, 3), dtype=torch.float32)
        stages = {name: sampled for name in ("raw", "cropped", "sampled")}
        return stages, {"depth_source": "ffs_stereo"}


def _debug_args(tmp_path: Path) -> debug_lerobot.DebugInputs:
    args = debug_lerobot.DebugInputs()
    args.lerobot_path = tmp_path
    args.frame_index = 0
    args.camera = "head"
    args.rgbd_sidecar_source = "auto"
    args.pointcloud_mode = "xyz"
    args.num_points = 4
    args.builder_config = str(tmp_path / "ffs.yaml")
    args.depth_source = "ffs_stereo"
    args.ffs_backend = "pytorch"
    args.ffs_artifact_id = "fake"
    args.ffs_precision = "fp16"
    args.ffs_builder_optimization_level = 3
    args.ffs_workspace_gib = 8.0
    args.dp3_zarr = None
    args.temp_config_path = None
    args.debug_output_path = tmp_path / "debug.zarr"
    return args


def test_lerobot_debug_ffs_requests_ir_and_never_passes_native_depth(monkeypatch, tmp_path):
    source = _FakeSource(tmp_path)
    _FakeBuilder.frames = []
    resolution = debug_lerobot.exporter.BuilderConfigResolution(
        config_path=tmp_path / "ffs.yaml",
        config={"depth_source": {"mode": "ffs_stereo"}},
        depth_source="ffs_stereo",
        ffs_provenance={
            "ffs_backend": "pytorch",
            "artifact_id": "fake",
            "precision": "fp16",
            "max_disp": 192,
            "valid_iters": 8,
            "builder_optimization_level": 3,
            "workspace_gib": 8.0,
            "calibration_sha256": "a" * 64,
            "builder_config": {"resolved_config_sha256": "b" * 64},
            "artifacts": {"manifest_sha256": "c" * 64, "files": {}},
        },
    )
    monkeypatch.setattr(debug_lerobot.exporter, "_data_parquet_paths", lambda _root: [tmp_path / "data.parquet"])
    monkeypatch.setattr(debug_lerobot.exporter, "_count_parquet_rows", lambda _paths: 1)
    monkeypatch.setattr(debug_lerobot.exporter, "_read_json", lambda _path: {})
    monkeypatch.setattr(debug_lerobot.exporter, "_read_episode_rows", lambda _root: [])
    monkeypatch.setattr(
        debug_lerobot.exporter.rgbd_source,
        "open_rgbd_sidecar_source",
        lambda *args, **kwargs: source,
    )
    monkeypatch.setattr(
        debug_lerobot,
        "_resolve_builder_resolution_for_debug",
        lambda **kwargs: resolution,
    )
    monkeypatch.setattr(debug_lerobot.exporter, "_import_pointcloud_builder", lambda: _FakeBuilder)

    stages, _meta, row, _config_path = debug_lerobot._build_frame_stages(_debug_args(tmp_path))

    assert source.include_ir_calls == [True]
    assert set(_FakeBuilder.frames[0]) == {"left_ir", "right_ir", "timestamp", "global_frame_index"}
    assert row["_depth_source"] == "ffs_stereo"
    assert tuple(stages["sampled"].shape) == (4, 3)


def test_zarr_debug_infers_ffs_contract_from_attrs(tmp_path):
    args = argparse.Namespace(
        dp3_zarr=str(tmp_path / "ffs.zarr"),
        frame_index=0,
        lerobot_path=None,
        camera=None,
        rgbd_sidecar_source=None,
        pointcloud_mode=None,
        num_points=None,
        builder_config=None,
        depth_source=None,
        ffs_backend=None,
        ffs_artifact_id=None,
        ffs_precision=None,
        ffs_builder_optimization_level=None,
        ffs_workspace_gib=None,
        window_width=1800,
        window_height=760,
        point_size=2.0,
        no_show=True,
    )
    attrs = {
        "source_lerobot_path": str(tmp_path / "dataset"),
        "camera": "head",
        "pointcloud_mode": "xyz",
        "num_points": 1024,
        "source_sidecar_storage": "zarr_v2",
        "depth_source": "ffs_stereo",
        "ffs_backend": "tensorrt_single",
        "artifact_id": "fp16_o3",
        "precision": "fp16",
        "builder_optimization_level": 3,
        "workspace_gib": 8.0,
        "pointcloud_builder_config": {"depth_source": {"mode": "ffs_stereo"}},
    }

    resolved = debug_zarr._lerobot_args_from_zarr_attrs(args, attrs, tmp_path / "tmp")

    assert resolved.depth_source == "ffs_stereo"
    assert resolved.ffs_backend == "tensorrt_single"
    assert resolved.ffs_artifact_id == "fp16_o3"
    assert resolved.ffs_precision == "fp16"
    assert resolved.ffs_builder_optimization_level == 3
    assert resolved.ffs_workspace_gib == 8.0
    assert Path(resolved.builder_config).is_file()


def test_live_perception_ffs_preflights_declared_route(monkeypatch, tmp_path):
    config_path = tmp_path / "live_ffs.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "camera": {
                    "name": "head",
                    "depth_scale": 1.0,
                    "aligned_depth_to_color": False,
                    "color_intrinsics": {},
                    "depth_intrinsics": {},
                },
                "pointcloud": {"use_rgb": False, "output_format": "xyz"},
                "depth_source": {
                    "mode": "ffs_stereo",
                    "ffs": {
                        "backend": "pytorch",
                        "artifact_id": "fp16_o3",
                        "checkpoint_path": "model.pth",
                        "model_config_path": "cfg.yaml",
                        "manifest_path": "manifest.json",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        perception_only.exporter,
        "_preflight_ffs_artifacts",
        lambda *args, **kwargs: {},
    )
    args = argparse.Namespace(
        builder_config=config_path,
        depth_source="ffs_stereo",
        ffs_backend="pytorch",
        ffs_artifact_id="fp16_o3",
        ffs_precision="fp16",
        ffs_builder_optimization_level=3,
        ffs_workspace_gib=8.0,
    )

    assert perception_only._resolve_live_depth_source(args, config_path) == "ffs_stereo"


def test_live_perception_rejects_native_mode_with_ffs_yaml(tmp_path):
    config_path = tmp_path / "ffs.yaml"
    config_path.write_text(
        yaml.safe_dump({"depth_source": {"mode": "ffs_stereo", "ffs": {}}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        builder_config=config_path,
        depth_source="native_depth",
        ffs_backend=None,
        ffs_artifact_id=None,
        ffs_precision=None,
        ffs_builder_optimization_level=None,
        ffs_workspace_gib=None,
    )

    try:
        perception_only._resolve_live_depth_source(args, config_path)
    except SystemExit as exc:
        assert "conflicts with Builder YAML" in str(exc)
    else:
        raise AssertionError("native mode unexpectedly accepted an FFS Builder YAML")
