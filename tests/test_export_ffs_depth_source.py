from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "export_lerobot_to_dp3_zarr.py"
SPEC = importlib.util.spec_from_file_location("exporter_ffs_depth_source", MODULE_PATH)
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
assert SPEC.loader is not None
SPEC.loader.exec_module(exporter)


def _state() -> np.ndarray:
    value = np.zeros(exporter.STATE_DIM, dtype=np.float32)
    value[10:16] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    value[27:33] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    value[16] = 0.5
    value[33] = 0.5
    return value


def _info() -> dict:
    return {
        "total_frames": 1,
        "total_episodes": 1,
        "features": {
            exporter.STATE_COLUMN: {
                "shape": [exporter.STATE_DIM],
                "names": list(exporter.STATE_FIELD_NAMES),
            },
            exporter.ACTION_COLUMN: {
                "shape": [exporter.ACTION_DIM],
                "names": list(exporter.ACTION_FIELD_NAMES),
            },
        },
    }


def _pair(tmp_path: Path, *, left: np.ndarray | None = None, right: np.ndarray | None = None, timestamp: float = 10.5):
    calibration_path = tmp_path / "meta/realsense_calibration.json"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    return exporter.rgbd_source.IRPair(
        left_ir=np.zeros((480, 640), dtype=np.uint8) if left is None else left,
        right_ir=np.ones((480, 640), dtype=np.uint8) if right is None else right,
        timestamp=timestamp,
        reused=False,
        calibration_path=calibration_path,
        calibration_sha256=digest,
    )


class FakeSource:
    storage = "zarr_v2"
    schema_name = "lerobot_realsense_raw_sidecar"
    schema_version = 2

    def __init__(self, tmp_path: Path, *, frame_count: int = 1, pair=None, calibration_sha: str | None = None):
        self.include_ir_calls: list[bool] = []
        calibration_path = tmp_path / "meta/realsense_calibration.json"
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text("{}", encoding="utf-8")
        self.calibration_path = calibration_path
        self.calibration_sha256 = calibration_sha or hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        self.calibration = {}
        self.frames = []
        for index in range(frame_count):
            row = {
                exporter.STATE_COLUMN: _state(),
                exporter.ACTION_COLUMN: np.zeros(exporter.ACTION_DIM, dtype=np.float32),
                "global_frame_index": index,
                "head_rgbd_timestamp": 10.5 + index,
                "head_rgbd_reused": False,
                "episode_index": 0,
                "frame_index": index,
                "index": index,
            }
            frame_pair = pair if pair is not None else _pair(tmp_path, timestamp=row["head_rgbd_timestamp"])
            if pair is not None and index:
                frame_pair = exporter.rgbd_source.IRPair(
                    left_ir=frame_pair.left_ir,
                    right_ir=frame_pair.right_ir,
                    timestamp=row["head_rgbd_timestamp"],
                    reused=frame_pair.reused,
                    calibration_path=frame_pair.calibration_path,
                    calibration_sha256=frame_pair.calibration_sha256,
                )
            self.frames.append(
                exporter.rgbd_source.RGBDSourceFrame(
                    row_index=index,
                    row=row,
                    source_path=tmp_path / "data/chunk-000/file-000.parquet",
                    depth=np.full((2, 3), 1000, dtype=np.uint16),
                    ir_pair=frame_pair,
                )
            )

    @property
    def provenance(self) -> dict:
        return {
            "source_sidecar_storage": self.storage,
            "source_sidecar_schema_name": self.schema_name,
            "source_sidecar_schema_version": self.schema_version,
            "source_sidecar_manifest_relative_path": "meta/rgbd_sidecar.json",
            "source_sidecar_manifest_path": "meta/rgbd_sidecar.json",
            "source_sidecar_manifest_sha256": "a" * 64,
            "source_sidecar_calibration_relative_path": "meta/realsense_calibration.json",
            "source_sidecar_calibration_path": str(self.calibration_path),
            "source_sidecar_calibration_sha256": self.calibration_sha256,
            "source_sidecar_committed_frames": len(self.frames),
            "source_sidecar_committed_episodes": 1,
            "source_sidecar_relative_path": "sidecars/realsense.zarr",
            "source_sidecar_path": "sidecars/realsense.zarr",
            "source_sidecar_depth_units": "native_realsense_uint16_units",
        }

    def validate_join(self, data_paths, *, camera: str, batch_size: int = 128) -> None:
        del data_paths, camera, batch_size

    def depth_scale_m_per_unit(self, camera: str) -> float:
        del camera
        return 0.001

    def iter_frames(self, data_paths, *, camera, columns, max_frames, include_ir=False, batch_size=32):
        del data_paths, camera, columns, batch_size
        self.include_ir_calls.append(bool(include_ir))
        yield from self.frames[:max_frames]


class FakeBuilder:
    init_count = 0
    frame_count = 0
    frames: list[dict] = []
    fail = False

    @classmethod
    def from_yaml(cls, path: Path):
        cls.init_count += 1
        if cls.fail:
            raise RuntimeError("fake FFS backend construction failed")
        return cls()

    def from_recorded_frame(self, frame):
        type(self).frame_count += 1
        type(self).frames.append(frame)
        return torch.zeros((4, 3), dtype=torch.float32), {
            "stage": "sampled",
            "mode": "recorded",
            "num_raw_points": 6,
            "num_cropped_points": 5,
            "num_sampled_points": 4,
            "sampling_enabled": True,
            "sampling_mode": "stride",
            "target_num_points": 4,
            "padded": False,
            "camera_name": "head",
            "timestamp": frame["timestamp"],
            "global_frame_index": frame["global_frame_index"],
        }


def _resolution(mode: str, tmp_path: Path) -> exporter.BuilderConfigResolution:
    config = {
        "camera": {},
        "pointcloud": {"use_rgb": False, "output_format": "xyz"},
        "sampling": {"enabled": True, "num_points": 4},
    }
    if mode == "ffs_stereo":
        config["depth_source"] = {"mode": "ffs_stereo", "ffs": {"left_key": "left_ir", "right_key": "right_ir"}}
        return exporter.BuilderConfigResolution(
            config_path=tmp_path / "ffs.yaml",
            config=config,
            depth_source=mode,
            ffs_provenance={
                "ffs_backend": "pytorch",
                "artifact_id": "fake",
                "precision": "fp16",
                "max_disp": 192,
                "valid_iters": 8,
                "builder_optimization_level": 3,
                "workspace_gib": 8.0,
                "normalization_contract": "internal_imagenet_0_255",
                "calibration_sha256": "b" * 64,
                "rectification_mode": "identity/no-op",
                "builder_config": {
                    "resolved_config": {"depth_source": {"mode": "ffs_stereo"}},
                    "resolved_config_sha256": "c" * 64,
                    "runtime_config_sha256": "d" * 64,
                },
                "artifacts": {
                    "manifest_sha256": "e" * 64,
                    "manifest_relative_path": "fake.manifest.json",
                    "files": {
                        "checkpoint_path": {"file_name": "fake.pth", "relative_path": "fake.pth", "sha256": "f" * 64}
                    },
                },
            },
            generated_config_path=False,
        )
    return exporter.BuilderConfigResolution(
        config_path=tmp_path / "native.yaml",
        config=config,
        depth_source=mode,
        generated_config_path=False,
    )


def _args(tmp_path: Path, output: Path, *, mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        lerobot_path=str(tmp_path),
        output_zarr=str(output),
        camera="head",
        rgbd_sidecar_source="auto",
        pointcloud_mode="xyz",
        num_points=4,
        builder_config=str(tmp_path / "builder.yaml"),
        depth_source=mode,
        ffs_backend=None,
        ffs_artifact_id=None,
        ffs_precision=None,
        ffs_builder_optimization_level=None,
        ffs_workspace_gib=None,
        target_state_schema=exporter.TARGET_STATE_SCHEMA,
        allow_legacy_state_conversion=False,
        overwrite=False,
        max_frames=None,
        save_img=False,
        verbose=False,
    )


def _run_export(
    monkeypatch,
    tmp_path: Path,
    *,
    mode: str,
    source: FakeSource,
    output: Path,
    builder_fail: bool = False,
):
    FakeBuilder.init_count = 0
    FakeBuilder.frame_count = 0
    FakeBuilder.frames = []
    FakeBuilder.fail = builder_fail
    args = _args(tmp_path, output, mode=mode)
    monkeypatch.setattr(exporter, "_read_json", lambda _path: _info())
    monkeypatch.setattr(exporter, "_data_parquet_paths", lambda _root: [tmp_path / "data.parquet"])
    monkeypatch.setattr(exporter, "_count_parquet_rows", lambda _paths: len(source.frames))
    monkeypatch.setattr(exporter, "_read_episode_rows", lambda _root: [{"episode_index": 0, "length": len(source.frames), "dataset_to_index": len(source.frames)}])
    monkeypatch.setattr(exporter.rgbd_source, "open_rgbd_sidecar_source", lambda *a, **k: source)
    monkeypatch.setattr(exporter, "_resolve_builder_config_for_export", lambda **_kwargs: _resolution(mode, tmp_path))
    monkeypatch.setattr(exporter, "_import_pointcloud_builder", lambda: FakeBuilder)
    return exporter.export_lerobot_to_dp3_zarr(args)


def test_parse_args_defaults_depth_source_to_native(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["export_lerobot_to_dp3_zarr.py", "--lerobot-path", "/tmp/dataset"])
    args = exporter.parse_args()
    assert args.depth_source == "native_depth"


def test_native_routes_depth_without_ir_and_preserves_schema(monkeypatch, tmp_path):
    source = FakeSource(tmp_path)
    output = tmp_path / "native.zarr"
    summary = _run_export(monkeypatch, tmp_path, mode="native_depth", source=source, output=output)
    assert source.include_ir_calls == [False]
    assert set(FakeBuilder.frames[0]) == {"depth", "timestamp", "global_frame_index"}
    assert FakeBuilder.init_count == 1
    assert summary["point_cloud_shape"] == (1, 4, 3)
    root = __import__("zarr").open(str(output), mode="r")
    assert root["data/state"].shape == (1, exporter.STATE_DIM)
    assert root["data/action"].shape == (1, exporter.ACTION_DIM)
    assert root["meta/episode_ends"][:].tolist() == [1]
    assert root.attrs["depth_source"] == "native_depth"
    assert root.attrs["native_depth_used_for_builder"] is True
    assert exporter.verify_dp3_zarr(output)["state"] == root.attrs["integrity"]["state"]


def test_ffs_routes_ir_without_native_depth_and_records_provenance(monkeypatch, tmp_path):
    source = FakeSource(tmp_path)
    output = tmp_path / "ffs.zarr"
    _run_export(monkeypatch, tmp_path, mode="ffs_stereo", source=source, output=output)
    assert source.include_ir_calls == [True]
    frame = FakeBuilder.frames[0]
    assert set(frame) == {"left_ir", "right_ir", "timestamp", "global_frame_index"}
    assert "depth" not in frame
    assert frame["left_ir"].shape == (480, 640)
    root = __import__("zarr").open(str(output), mode="r")
    assert root.attrs["depth_source"] == "ffs_stereo"
    assert root.attrs["ffs_backend"] == "pytorch"
    assert root.attrs["artifact_id"] == "fake"
    assert root.attrs["native_depth_used_for_builder"] is False
    assert root.attrs["pointcloud_builder_metadata"]["count_fields"]["num_sampled_points"]["last"] == 4
    assert exporter.verify_dp3_zarr(output)["point_cloud"] == root.attrs["integrity"]["point_cloud"]


def test_ffs_builder_is_initialized_once_for_multiple_frames(monkeypatch, tmp_path):
    source = FakeSource(tmp_path, frame_count=3)
    output = tmp_path / "ffs_many.zarr"
    _run_export(monkeypatch, tmp_path, mode="ffs_stereo", source=source, output=output)
    assert source.include_ir_calls == [True]
    assert FakeBuilder.init_count == 1
    assert FakeBuilder.frame_count == 3


def test_missing_ir_pair_fails_fast_and_cleans_atomic_output(monkeypatch, tmp_path):
    source = FakeSource(tmp_path)
    source.frames[0] = exporter.rgbd_source.RGBDSourceFrame(
        row_index=0,
        row=source.frames[0].row,
        source_path=source.frames[0].source_path,
        depth=source.frames[0].depth,
        ir_pair=None,
    )
    output = tmp_path / "missing_ir.zarr"
    with pytest.raises(ValueError, match="camera=head.*ir_pair"):
        _run_export(monkeypatch, tmp_path, mode="ffs_stereo", source=source, output=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".missing_ir.zarr.incomplete-*"))


@pytest.mark.parametrize("kind", ["left_shape", "right_shape", "dtype_range", "timestamp"])
def test_ffs_ir_validation_rejects_shape_dtype_or_timestamp(monkeypatch, tmp_path, kind):
    valid = _pair(tmp_path)
    left = valid.left_ir
    right = valid.right_ir
    timestamp = valid.timestamp
    if kind == "left_shape":
        left = np.zeros((480, 639), dtype=np.uint8)
    elif kind == "right_shape":
        right = np.zeros((479, 640), dtype=np.uint8)
    elif kind == "dtype_range":
        left = np.full((480, 640), 256, dtype=np.uint16)
    elif kind == "timestamp":
        timestamp = 99.0
    pair = exporter.rgbd_source.IRPair(
        left_ir=left,
        right_ir=right,
        timestamp=timestamp,
        reused=False,
        calibration_path=valid.calibration_path,
        calibration_sha256=valid.calibration_sha256,
    )
    source = FakeSource(tmp_path, pair=pair)
    with pytest.raises(ValueError):
        _run_export(monkeypatch, tmp_path, mode="ffs_stereo", source=source, output=tmp_path / "bad.zarr")


def test_ffs_backend_construction_failure_has_no_native_fallback(monkeypatch, tmp_path):
    source = FakeSource(tmp_path)
    output = tmp_path / "backend_failure.zarr"
    with pytest.raises(RuntimeError, match="fake FFS backend construction failed"):
        _run_export(
            monkeypatch,
            tmp_path,
            mode="ffs_stereo",
            source=source,
            output=output,
            builder_fail=True,
        )
    assert source.include_ir_calls == []
    assert not output.exists()


def test_cli_yaml_backend_conflict_fails_fast(tmp_path):
    config_path = tmp_path / "builder.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "camera": {},
                "pointcloud": {"use_rgb": False, "output_format": "xyz"},
                "depth_source": {"mode": "ffs_stereo", "ffs": {"backend": "pytorch", "artifact_id": "fp16_o3"}},
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        builder_config=str(config_path),
        lerobot_path=str(tmp_path),
        camera="head",
        pointcloud_mode="xyz",
        num_points=4,
        depth_source="ffs_stereo",
        ffs_backend="tensorrt_single",
        ffs_artifact_id=None,
        ffs_precision=None,
        ffs_builder_optimization_level=None,
        ffs_workspace_gib=None,
    )
    with pytest.raises(ValueError, match="CLI/YAML conflict for backend"):
        exporter._resolve_ffs_builder_config(
            args=args,
            output_zarr=tmp_path / "out.zarr",
            realsense_calibration={},
        )


@pytest.mark.parametrize(
    ("key", "yaml_value", "cli_value"),
    [
        ("artifact_id", "fp16_o3", "fp32_o0"),
        ("precision", "fp16", "fp32"),
        ("builder_optimization_level", 3, 0),
        ("workspace_gib", 8.0, 4.0),
    ],
)
def test_cli_yaml_ffs_scalar_conflicts_fail_fast(key, yaml_value, cli_value):
    fields = {key: yaml_value}
    with pytest.raises(ValueError, match=f"CLI/YAML conflict for {key}"):
        exporter._resolve_ffs_option(
            fields,
            key=key,
            cli_value=cli_value,
            default=None,
            normalizer=(lambda value: str(value)) if key in {"artifact_id", "precision"} else (lambda value: float(value)),
        )


def test_ffs_requires_explicit_builder_yaml(tmp_path):
    args = argparse.Namespace(
        lerobot_path=str(tmp_path),
        output_zarr=str(tmp_path / "out.zarr"),
        camera="head",
        pointcloud_mode="xyz",
        num_points=4,
        depth_source="ffs_stereo",
        builder_config=None,
        ffs_backend=None,
        ffs_artifact_id=None,
        ffs_precision=None,
        ffs_builder_optimization_level=None,
        ffs_workspace_gib=None,
    )
    with pytest.raises(ValueError, match="requires an explicit --builder-config"):
        exporter._resolve_builder_config_for_export(
            args=args,
            output_zarr=tmp_path / "out.zarr",
            realsense_calibration={},
        )


def test_native_mode_rejects_ffs_builder_yaml_before_runtime_import(tmp_path):
    config_path = tmp_path / "ffs_builder.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "camera": {},
                "pointcloud": {"use_rgb": False, "output_format": "xyz"},
                "depth_source": {"mode": "ffs_stereo", "ffs": {}},
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        builder_config=str(config_path),
        lerobot_path=str(tmp_path),
        camera="head",
        pointcloud_mode="xyz",
        num_points=4,
        depth_source="native_depth",
        ffs_backend=None,
        ffs_artifact_id=None,
        ffs_precision=None,
        ffs_builder_optimization_level=None,
        ffs_workspace_gib=None,
    )
    with pytest.raises(ValueError, match="conflicts with Builder YAML"):
        exporter._resolve_builder_config_for_export(
            args=args,
            output_zarr=tmp_path / "out.zarr",
            realsense_calibration={},
        )


def test_native_mode_does_not_import_optional_ffs_runtime_modules(monkeypatch, tmp_path):
    source = FakeSource(tmp_path)
    _run_export(monkeypatch, tmp_path, mode="native_depth", source=source, output=tmp_path / "native_imports.zarr")
    forbidden = {
        "tensorrt",
        "onnx",
        "open3d",
        "pointcloud_builder.ffs.vendor_loader",
        "pointcloud_builder.ffs.pytorch_backend",
        "pointcloud_builder.ffs.tensorrt_common",
    }
    assert forbidden.isdisjoint(sys.modules)
