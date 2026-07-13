"""Validated RGB-D/IR sidecar readers for local LeRobot datasets.

The raw acquisition sidecar and the derived DP3 replay buffer are different
schemas.  This module only reads the former.  It deliberately keeps storage
selection, manifest validation, join validation, and chunked sensor reads out
of the exporter point-cloud loop.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RAW_MANIFEST_RELATIVE_PATH = Path("meta/rgbd_sidecar.json")
RAW_SCHEMA_NAME = "lerobot_realsense_raw_sidecar"
RAW_SCHEMA_VERSION = 1
RAW_STORAGE = "zarr_v2"
RAW_ZARR_RELATIVE_PATH = Path("sidecars/realsense.zarr")
LEGACY_SCHEMA_NAME = "lerobot_rgbd_parquet_sidecar"

CAMERA_SPECS: dict[str, dict[str, str]] = {
    "head": {
        "calibration_key": "head_rgb",
        "depth_column": "sidecar.head_depth",
        "left_ir_column": "sidecar.head_left_ir",
        "right_ir_column": "sidecar.head_right_ir",
        "video_key": "observation.images.head_rgb",
        "timestamp_column": "head_rgbd_timestamp",
        "reused_column": "head_rgbd_reused",
    },
    "left_wrist": {
        "calibration_key": "left_wrist_rgb",
        "depth_column": "sidecar.left_wrist_depth",
        "left_ir_column": "sidecar.left_wrist_left_ir",
        "right_ir_column": "sidecar.left_wrist_right_ir",
        "video_key": "observation.images.left_wrist_rgb",
        "timestamp_column": "left_wrist_rgbd_timestamp",
        "reused_column": "left_wrist_rgbd_reused",
    },
    "right_wrist": {
        "calibration_key": "right_wrist_rgb",
        "depth_column": "sidecar.right_wrist_depth",
        "left_ir_column": "sidecar.right_wrist_left_ir",
        "right_ir_column": "sidecar.right_wrist_right_ir",
        "video_key": "observation.images.right_wrist_rgb",
        "timestamp_column": "right_wrist_rgbd_timestamp",
        "reused_column": "right_wrist_rgbd_reused",
    },
}

META_JOIN_COLUMNS = ("index", "episode_index", "frame_index", "global_frame_index")
ZARR_META_ARRAYS = (*META_JOIN_COLUMNS, "robot_timestamp")


@dataclass(frozen=True)
class IRPair:
    """Lossless same-row raw IR data and its synchronization metadata."""

    left_ir: np.ndarray
    right_ir: np.ndarray
    timestamp: float
    reused: bool
    calibration_path: Path
    calibration_sha256: str

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class RGBDSourceFrame:
    """One scalar Parquet row joined to one native-depth sidecar frame."""

    row_index: int
    row: dict[str, Any]
    source_path: Path
    depth: np.ndarray
    ir_pair: IRPair | None = None


class RGBDSidecarSource(ABC):
    """Unified interface implemented by legacy Parquet and raw Zarr sources."""

    storage: str
    schema_name: str
    schema_version: int

    def __init__(
        self,
        dataset_root: Path,
        *,
        total_frames: int,
        total_episodes: int,
        calibration_path: Path,
        calibration: dict[str, Any],
        calibration_sha256: str,
    ) -> None:
        self.dataset_root = dataset_root
        self.total_frames = int(total_frames)
        self.total_episodes = int(total_episodes)
        self.calibration_path = calibration_path
        self.calibration = calibration
        self.calibration_sha256 = calibration_sha256

    @property
    @abstractmethod
    def provenance(self) -> dict[str, Any]:
        """JSON-compatible source metadata for the derived DP3 attrs."""

    @abstractmethod
    def validate_join(
        self,
        data_paths: list[Path],
        *,
        camera: str,
        batch_size: int = 128,
    ) -> None:
        """Validate the complete scalar/sensor join before point generation."""

    @abstractmethod
    def get_depth(
        self,
        camera: str,
        row_index: int,
        row: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Read one native uint16 depth frame without reopening the source."""

    @abstractmethod
    def get_ir_pair(
        self,
        camera: str,
        row_index: int,
        row: Mapping[str, Any] | None = None,
    ) -> IRPair:
        """Read one lossless left/right IR pair without deriving depth."""

    @abstractmethod
    def _parquet_columns(
        self,
        camera: str,
        requested: list[str],
        *,
        include_ir: bool,
    ) -> list[str]:
        pass

    @abstractmethod
    def _depth_batch(
        self,
        camera: str,
        start: int,
        stop: int,
        rows: dict[str, list[Any]],
        source_path: Path,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _ir_batch(
        self,
        camera: str,
        start: int,
        stop: int,
        rows: dict[str, list[Any]],
        source_path: Path,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        pass

    def depth_scale_m_per_unit(self, camera: str) -> float:
        calibration = _camera_calibration(self.calibration, camera)
        scale = float(calibration["depth_scale_m_per_unit"])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid depth scale for camera {camera!r}: {scale}")
        return scale

    def iter_frames(
        self,
        data_paths: list[Path],
        *,
        camera: str,
        columns: list[str],
        max_frames: int,
        include_ir: bool = False,
        batch_size: int = 32,
    ) -> Iterator[RGBDSourceFrame]:
        """Yield joined frames while reading Parquet and sensor arrays in batches."""

        _require_camera(camera)
        if max_frames <= 0 or max_frames > self.total_frames:
            raise ValueError(
                f"max_frames must be within [1, {self.total_frames}], got {max_frames}"
            )
        parquet_columns = self._parquet_columns(camera, columns, include_ir=include_ir)
        row_index = 0
        for rows, source_path in iter_parquet_batches(
            data_paths,
            columns=parquet_columns,
            max_frames=max_frames,
            batch_size=batch_size,
        ):
            count = len(rows[parquet_columns[0]])
            stop = row_index + count
            depth_batch = self._depth_batch(camera, row_index, stop, rows, source_path)
            ir_batch = self._ir_batch(camera, row_index, stop, rows, source_path) if include_ir else None
            for offset in range(count):
                row = {column: rows[column][offset] for column in parquet_columns}
                ir_pair = None
                if ir_batch is not None:
                    ir_pair = IRPair(
                        left_ir=np.asarray(ir_batch[0][offset], dtype=np.uint8),
                        right_ir=np.asarray(ir_batch[1][offset], dtype=np.uint8),
                        timestamp=float(row[CAMERA_SPECS[camera]["timestamp_column"]]),
                        reused=bool(row[CAMERA_SPECS[camera]["reused_column"]]),
                        calibration_path=self.calibration_path,
                        calibration_sha256=self.calibration_sha256,
                    )
                yield RGBDSourceFrame(
                    row_index=row_index + offset,
                    row=row,
                    source_path=source_path,
                    depth=np.asarray(depth_batch[offset], dtype=np.uint16),
                    ir_pair=ir_pair,
                )
            row_index = stop
        if row_index != max_frames:
            raise RuntimeError(f"Read {row_index} scalar rows but expected {max_frames}")

    def read_frame_at(
        self,
        data_paths: list[Path],
        *,
        camera: str,
        row_index: int,
        columns: list[str],
        include_ir: bool = False,
    ) -> RGBDSourceFrame:
        """Read one joined frame while skipping unrelated Parquet row groups."""

        _require_camera(camera)
        _require_row_index(row_index, self.total_frames)
        parquet_columns = self._parquet_columns(camera, columns, include_ir=include_ir)
        row, source_path = read_parquet_row_at_index(
            data_paths,
            columns=parquet_columns,
            row_index=row_index,
        )
        depth = self.get_depth(camera, row_index, row)
        ir_pair = self.get_ir_pair(camera, row_index, row) if include_ir else None
        return RGBDSourceFrame(
            row_index=row_index,
            row=row,
            source_path=source_path,
            depth=depth,
            ir_pair=ir_pair,
        )


class LegacyParquetRGBDSource(RGBDSidecarSource):
    """Reader for the original ``sidecar.<camera>_*`` Parquet columns."""

    storage = "parquet"
    schema_name = LEGACY_SCHEMA_NAME
    schema_version = 1

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "source_sidecar_storage": self.storage,
            "source_sidecar_schema_name": self.schema_name,
            "source_sidecar_schema_version": self.schema_version,
            "source_sidecar_manifest_relative_path": None,
            "source_sidecar_manifest_path": None,
            "source_sidecar_manifest_sha256": None,
            "source_sidecar_calibration_relative_path": self.calibration_path.relative_to(
                self.dataset_root
            ).as_posix(),
            "source_sidecar_calibration_path": str(self.calibration_path),
            "source_sidecar_calibration_sha256": self.calibration_sha256,
            "source_sidecar_committed_frames": self.total_frames,
            "source_sidecar_committed_episodes": self.total_episodes,
            "source_sidecar_relative_path": "data/chunk-*/file-*.parquet",
            "source_sidecar_path": str(self.dataset_root / "data"),
            "source_sidecar_depth_units": "native_realsense_uint16_units",
        }

    def validate_join(
        self,
        data_paths: list[Path],
        *,
        camera: str,
        batch_size: int = 128,
    ) -> None:
        spec = CAMERA_SPECS[_require_camera(camera)]
        columns = [
            *META_JOIN_COLUMNS,
            spec["timestamp_column"],
            spec["reused_column"],
            spec["depth_column"],
        ]
        seen = 0
        order = _ParquetOrderValidator(self.total_frames)
        for rows, source_path in iter_parquet_batches(
            data_paths,
            columns=columns,
            max_frames=self.total_frames,
            batch_size=batch_size,
        ):
            count = len(rows[columns[0]])
            order.validate(rows, start=seen, source_path=source_path)
            for offset, value in enumerate(rows[spec["depth_column"]]):
                _as_image(value, np.uint16, f"{source_path}: {spec['depth_column']} row {seen + offset}")
            _validate_sync_values(rows, camera=camera, source_path=source_path)
            seen += count
        _require_complete_row_count(seen, self.total_frames)

    def get_depth(
        self,
        camera: str,
        row_index: int,
        row: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        spec = CAMERA_SPECS[_require_camera(camera)]
        if row is None or spec["depth_column"] not in row:
            raise ValueError("Legacy Parquet get_depth requires the corresponding scalar row")
        _require_row_index(row_index, self.total_frames)
        return _as_image(row[spec["depth_column"]], np.uint16, spec["depth_column"])

    def get_ir_pair(
        self,
        camera: str,
        row_index: int,
        row: Mapping[str, Any] | None = None,
    ) -> IRPair:
        spec = CAMERA_SPECS[_require_camera(camera)]
        if row is None:
            raise ValueError("Legacy Parquet get_ir_pair requires the corresponding scalar row")
        _require_row_index(row_index, self.total_frames)
        required = [
            spec["left_ir_column"],
            spec["right_ir_column"],
            spec["timestamp_column"],
            spec["reused_column"],
        ]
        missing = [column for column in required if column not in row]
        if missing:
            raise KeyError(f"Legacy Parquet row is missing IR fields: {missing}")
        return IRPair(
            left_ir=_as_image(row[spec["left_ir_column"]], np.uint8, spec["left_ir_column"]),
            right_ir=_as_image(row[spec["right_ir_column"]], np.uint8, spec["right_ir_column"]),
            timestamp=float(row[spec["timestamp_column"]]),
            reused=bool(row[spec["reused_column"]]),
            calibration_path=self.calibration_path,
            calibration_sha256=self.calibration_sha256,
        )

    def _parquet_columns(
        self,
        camera: str,
        requested: list[str],
        *,
        include_ir: bool,
    ) -> list[str]:
        spec = CAMERA_SPECS[_require_camera(camera)]
        extra = [spec["depth_column"]]
        if include_ir:
            extra.extend([spec["left_ir_column"], spec["right_ir_column"]])
        return _unique([*requested, *extra])

    def _depth_batch(
        self,
        camera: str,
        start: int,
        stop: int,
        rows: dict[str, list[Any]],
        source_path: Path,
    ) -> np.ndarray:
        column = CAMERA_SPECS[camera]["depth_column"]
        return np.stack(
            [
                _as_image(value, np.uint16, f"{source_path}: {column} row {start + offset}")
                for offset, value in enumerate(rows[column])
            ],
            axis=0,
        )

    def _ir_batch(
        self,
        camera: str,
        start: int,
        stop: int,
        rows: dict[str, list[Any]],
        source_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        spec = CAMERA_SPECS[camera]
        values = []
        for column in (spec["left_ir_column"], spec["right_ir_column"]):
            values.append(
                np.stack(
                    [
                        _as_image(
                            value,
                            np.uint8,
                            f"{source_path}: {column} row {start + offset}",
                        )
                        for offset, value in enumerate(rows[column])
                    ],
                    axis=0,
                )
            )
        return values[0], values[1]


class ZarrRGBDSource(RGBDSidecarSource):
    """Reader for a complete ``lerobot_realsense_raw_sidecar`` Zarr v2 store."""

    storage = RAW_STORAGE
    schema_name = RAW_SCHEMA_NAME
    schema_version = RAW_SCHEMA_VERSION

    def __init__(
        self,
        dataset_root: Path,
        *,
        total_frames: int,
        total_episodes: int,
        manifest_path: Path,
        manifest: dict[str, Any],
        manifest_sha256: str,
        zarr_path: Path,
        root: Any,
        calibration_path: Path,
        calibration: dict[str, Any],
        calibration_sha256: str,
    ) -> None:
        super().__init__(
            dataset_root,
            total_frames=total_frames,
            total_episodes=total_episodes,
            calibration_path=calibration_path,
            calibration=calibration,
            calibration_sha256=calibration_sha256,
        )
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self.zarr_path = zarr_path
        self.root = root

    @property
    def provenance(self) -> dict[str, Any]:
        depth_units = _manifest_depth_units(self.manifest)
        return {
            "source_sidecar_storage": self.storage,
            "source_sidecar_schema_name": self.schema_name,
            "source_sidecar_schema_version": self.schema_version,
            "source_sidecar_manifest_relative_path": self.manifest_path.relative_to(
                self.dataset_root
            ).as_posix(),
            "source_sidecar_manifest_path": str(self.manifest_path),
            "source_sidecar_manifest_sha256": self.manifest_sha256,
            "source_sidecar_calibration_relative_path": self.calibration_path.relative_to(
                self.dataset_root
            ).as_posix(),
            "source_sidecar_calibration_path": str(self.calibration_path),
            "source_sidecar_calibration_sha256": self.calibration_sha256,
            "source_sidecar_committed_frames": self.total_frames,
            "source_sidecar_committed_episodes": self.total_episodes,
            "source_sidecar_relative_path": self.zarr_path.relative_to(
                self.dataset_root
            ).as_posix(),
            "source_sidecar_path": str(self.zarr_path),
            "source_sidecar_depth_units": depth_units,
        }

    def validate_join(
        self,
        data_paths: list[Path],
        *,
        camera: str,
        batch_size: int = 128,
    ) -> None:
        spec = CAMERA_SPECS[_require_camera(camera)]
        columns = [
            *META_JOIN_COLUMNS,
            "robot_timestamp",
            spec["timestamp_column"],
            spec["reused_column"],
        ]
        seen = 0
        order = _ParquetOrderValidator(self.total_frames)
        for rows, source_path in iter_parquet_batches(
            data_paths,
            columns=columns,
            max_frames=self.total_frames,
            batch_size=batch_size,
        ):
            count = len(rows[columns[0]])
            stop = seen + count
            order.validate(rows, start=seen, source_path=source_path)
            _validate_sync_values(rows, camera=camera, source_path=source_path)
            for column in META_JOIN_COLUMNS:
                _require_array_equal(
                    np.asarray(rows[column], dtype=np.int64),
                    np.asarray(self.root[f"meta/{column}"][seen:stop], dtype=np.int64),
                    name=column,
                    start=seen,
                )
            _require_array_equal(
                np.asarray(rows["robot_timestamp"], dtype=np.float64),
                np.asarray(self.root["meta/robot_timestamp"][seen:stop], dtype=np.float64),
                name="robot_timestamp",
                start=seen,
            )
            _require_array_equal(
                np.asarray(rows[spec["timestamp_column"]], dtype=np.float64),
                np.asarray(
                    self.root[f"data/{camera}/rgbd_timestamp"][seen:stop],
                    dtype=np.float64,
                ),
                name=spec["timestamp_column"],
                start=seen,
            )
            _require_array_equal(
                np.asarray(rows[spec["reused_column"]], dtype=np.bool_),
                np.asarray(
                    self.root[f"data/{camera}/rgbd_reused"][seen:stop],
                    dtype=np.bool_,
                ),
                name=spec["reused_column"],
                start=seen,
            )
            seen = stop
        _require_complete_row_count(seen, self.total_frames)

    def get_depth(
        self,
        camera: str,
        row_index: int,
        row: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        camera = _require_camera(camera)
        _require_row_index(row_index, self.total_frames)
        if row is not None:
            self._validate_row_join(camera, row_index, row)
        return np.asarray(self.root[f"data/{camera}/depth"][row_index], dtype=np.uint16)

    def get_ir_pair(
        self,
        camera: str,
        row_index: int,
        row: Mapping[str, Any] | None = None,
    ) -> IRPair:
        camera = _require_camera(camera)
        _require_row_index(row_index, self.total_frames)
        if row is not None:
            self._validate_row_join(camera, row_index, row)
        return IRPair(
            left_ir=np.asarray(self.root[f"data/{camera}/left_ir"][row_index], dtype=np.uint8),
            right_ir=np.asarray(self.root[f"data/{camera}/right_ir"][row_index], dtype=np.uint8),
            timestamp=float(self.root[f"data/{camera}/rgbd_timestamp"][row_index]),
            reused=bool(self.root[f"data/{camera}/rgbd_reused"][row_index]),
            calibration_path=self.calibration_path,
            calibration_sha256=self.calibration_sha256,
        )

    def _parquet_columns(
        self,
        camera: str,
        requested: list[str],
        *,
        include_ir: bool,
    ) -> list[str]:
        spec = CAMERA_SPECS[_require_camera(camera)]
        return _unique(
            [
                *requested,
                *META_JOIN_COLUMNS,
                "robot_timestamp",
                spec["timestamp_column"],
                spec["reused_column"],
            ]
        )

    def _depth_batch(
        self,
        camera: str,
        start: int,
        stop: int,
        rows: dict[str, list[Any]],
        source_path: Path,
    ) -> np.ndarray:
        self._validate_batch_join(camera, start, stop, rows)
        return np.asarray(self.root[f"data/{camera}/depth"][start:stop], dtype=np.uint16)

    def _ir_batch(
        self,
        camera: str,
        start: int,
        stop: int,
        rows: dict[str, list[Any]],
        source_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.root[f"data/{camera}/left_ir"][start:stop], dtype=np.uint8),
            np.asarray(self.root[f"data/{camera}/right_ir"][start:stop], dtype=np.uint8),
        )

    def _validate_row_join(
        self,
        camera: str,
        row_index: int,
        row: Mapping[str, Any],
    ) -> None:
        spec = CAMERA_SPECS[camera]
        pairs = [
            ("index", "meta/index", np.int64),
            ("episode_index", "meta/episode_index", np.int64),
            ("frame_index", "meta/frame_index", np.int64),
            ("global_frame_index", "meta/global_frame_index", np.int64),
            ("robot_timestamp", "meta/robot_timestamp", np.float64),
            (spec["timestamp_column"], f"data/{camera}/rgbd_timestamp", np.float64),
            (spec["reused_column"], f"data/{camera}/rgbd_reused", np.bool_),
        ]
        missing = [column for column, _, _ in pairs if column not in row]
        if missing:
            raise KeyError(f"Scalar Parquet row is missing Zarr join fields: {missing}")
        for column, zarr_key, dtype in pairs:
            left = np.asarray([row[column]], dtype=dtype)
            right = np.asarray([self.root[zarr_key][row_index]], dtype=dtype)
            _require_array_equal(left, right, name=column, start=row_index)

    def _validate_batch_join(
        self,
        camera: str,
        start: int,
        stop: int,
        rows: dict[str, list[Any]],
    ) -> None:
        spec = CAMERA_SPECS[camera]
        available = set(rows)
        pairs = [
            ("index", "meta/index", np.int64),
            ("episode_index", "meta/episode_index", np.int64),
            ("frame_index", "meta/frame_index", np.int64),
            ("global_frame_index", "meta/global_frame_index", np.int64),
            ("robot_timestamp", "meta/robot_timestamp", np.float64),
            (spec["timestamp_column"], f"data/{camera}/rgbd_timestamp", np.float64),
            (spec["reused_column"], f"data/{camera}/rgbd_reused", np.bool_),
        ]
        for column, zarr_key, dtype in pairs:
            if column not in available:
                continue
            _require_array_equal(
                np.asarray(rows[column], dtype=dtype),
                np.asarray(self.root[zarr_key][start:stop], dtype=dtype),
                name=column,
                start=start,
            )


def open_rgbd_sidecar_source(
    dataset_root: str | Path,
    *,
    source: str,
    info: dict[str, Any],
    parquet_row_count: int,
    total_episodes: int | None = None,
) -> RGBDSidecarSource:
    """Select and fully validate a raw Zarr or legacy Parquet sidecar."""

    root = Path(dataset_root).expanduser().resolve()
    if source not in {"auto", "zarr", "parquet"}:
        raise ValueError("source must be one of: auto, zarr, parquet")
    manifest_path = root / RAW_MANIFEST_RELATIVE_PATH
    manifest_exists = manifest_path.is_file()
    if source == "zarr" and not manifest_exists:
        raise FileNotFoundError(
            f"--rgbd-sidecar-source=zarr requires manifest: {manifest_path}"
        )
    if source == "parquet" and manifest_exists:
        raise ValueError(
            "--rgbd-sidecar-source=parquet conflicts with the authoritative raw Zarr "
            f"manifest at {manifest_path}; use zarr or auto"
        )
    selected = "zarr" if manifest_exists and source in {"auto", "zarr"} else "parquet"

    info_frames = _required_nonnegative_int(info, "total_frames", "meta/info.json")
    if info_frames <= 0:
        raise ValueError("meta/info.json total_frames must be positive")
    if int(parquet_row_count) != info_frames:
        raise ValueError(
            f"Scalar Parquet row count {parquet_row_count} does not match "
            f"meta/info.json total_frames {info_frames}"
        )
    info_episodes_value = info.get("total_episodes", total_episodes)
    if info_episodes_value is None:
        raise KeyError("meta/info.json is missing total_episodes")
    info_episodes = int(info_episodes_value)
    if info_episodes < 0:
        raise ValueError("meta/info.json total_episodes must be non-negative")
    if total_episodes is not None and int(total_episodes) != info_episodes:
        raise ValueError(
            f"Episode metadata count {total_episodes} does not match "
            f"meta/info.json total_episodes {info_episodes}"
        )

    if selected == "zarr":
        return _open_zarr_source(
            root,
            manifest_path=manifest_path,
            info_frames=info_frames,
            info_episodes=info_episodes,
        )

    calibration_path = (root / "meta/realsense_calibration.json").resolve()
    calibration = _read_json_mapping(calibration_path)
    return LegacyParquetRGBDSource(
        root,
        total_frames=info_frames,
        total_episodes=info_episodes,
        calibration_path=calibration_path,
        calibration=calibration,
        calibration_sha256=_file_sha256(calibration_path),
    )


def iter_parquet_batches(
    data_paths: list[Path],
    *,
    columns: list[str],
    max_frames: int,
    batch_size: int,
) -> Iterator[tuple[dict[str, list[Any]], Path]]:
    """Yield bounded PyArrow batches without materializing the full dataset."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to read LeRobot parquet files") from exc
    if not data_paths:
        raise FileNotFoundError("No scalar LeRobot Parquet files were provided")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    remaining = int(max_frames)
    for path in data_paths:
        parquet_file = pq.ParquetFile(path)
        missing = sorted(set(columns) - set(parquet_file.schema_arrow.names))
        if missing:
            raise KeyError(f"Missing parquet columns in {path}: {missing}")
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            if remaining <= 0:
                return
            data = batch.to_pydict()
            count = min(batch.num_rows, remaining)
            if count != batch.num_rows:
                data = {column: values[:count] for column, values in data.items()}
            yield data, path
            remaining -= count
            if remaining <= 0:
                return


def read_parquet_row_at_index(
    data_paths: list[Path],
    *,
    columns: list[str],
    row_index: int,
) -> tuple[dict[str, Any], Path]:
    """Read one global Parquet row without scanning every preceding row."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to read LeRobot parquet files") from exc
    if row_index < 0:
        raise IndexError("row_index must be non-negative")
    remaining = int(row_index)
    for path in data_paths:
        parquet_file = pq.ParquetFile(path)
        missing = sorted(set(columns) - set(parquet_file.schema_arrow.names))
        if missing:
            raise KeyError(f"Missing parquet columns in {path}: {missing}")
        file_rows = int(parquet_file.metadata.num_rows)
        if remaining >= file_rows:
            remaining -= file_rows
            continue
        for row_group in range(parquet_file.num_row_groups):
            group_rows = int(parquet_file.metadata.row_group(row_group).num_rows)
            if remaining >= group_rows:
                remaining -= group_rows
                continue
            table = parquet_file.read_row_group(row_group, columns=columns).slice(remaining, 1)
            values = table.to_pydict()
            return {column: values[column][0] for column in columns}, path
    raise IndexError(f"Parquet dataset ended before row {row_index}")


def _open_zarr_source(
    root: Path,
    *,
    manifest_path: Path,
    info_frames: int,
    info_episodes: int,
) -> ZarrRGBDSource:
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid raw sidecar manifest JSON: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Raw sidecar manifest must be a JSON mapping: {manifest_path}")
    _validate_manifest_identity(manifest, manifest_path)
    committed_frames = _required_nonnegative_int(manifest, "committed_frames", str(manifest_path))
    committed_episodes = _required_nonnegative_int(
        manifest, "committed_episodes", str(manifest_path)
    )
    if committed_frames != info_frames:
        raise ValueError(
            f"Manifest committed_frames {committed_frames} does not match "
            f"meta/info.json total_frames {info_frames}"
        )
    if committed_episodes != info_episodes:
        raise ValueError(
            f"Manifest committed_episodes {committed_episodes} does not match "
            f"meta/info.json total_episodes {info_episodes}"
        )
    relative_path = _safe_manifest_relative_path(
        manifest.get("relative_path"),
        expected=RAW_ZARR_RELATIVE_PATH,
        field="relative_path",
    )
    zarr_path = (root / relative_path).resolve()
    if not zarr_path.is_dir():
        raise FileNotFoundError(f"Raw sidecar Zarr store does not exist: {zarr_path}")
    _validate_zarr_v2(zarr_path)

    calibration_relative, expected_calibration_hash = _manifest_calibration_reference(manifest)
    calibration_relative = _safe_manifest_relative_path(
        calibration_relative,
        expected=None,
        field="calibration.relative_path",
    )
    calibration_path = (root / calibration_relative).resolve()
    calibration = _read_json_mapping(calibration_path)
    actual_calibration_hash = _file_sha256(calibration_path)
    if actual_calibration_hash != expected_calibration_hash:
        raise ValueError(
            f"Calibration SHA-256 mismatch for {calibration_path}: "
            f"actual={actual_calibration_hash}, manifest={expected_calibration_hash}"
        )

    try:
        import zarr
    except ImportError as exc:
        raise ImportError("zarr>=2,<3 is required to read raw RGB-D sidecars") from exc
    zarr_root = zarr.open(str(zarr_path), mode="r")
    _validate_mirrored_attrs(
        dict(zarr_root.attrs),
        manifest,
        calibration_sha256=actual_calibration_hash,
    )
    _validate_manifest_join_semantics(manifest)
    _validate_manifest_depth_units(manifest)
    _validate_manifest_layout_declarations(manifest)
    _validate_zarr_arrays(
        zarr_root,
        manifest=manifest,
        calibration=calibration,
        total_frames=committed_frames,
        total_episodes=committed_episodes,
    )
    return ZarrRGBDSource(
        root,
        total_frames=committed_frames,
        total_episodes=committed_episodes,
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        zarr_path=zarr_path,
        root=zarr_root,
        calibration_path=calibration_path,
        calibration=calibration,
        calibration_sha256=actual_calibration_hash,
    )


def _validate_manifest_identity(manifest: dict[str, Any], path: Path) -> None:
    expected = {
        "schema_name": RAW_SCHEMA_NAME,
        "schema_version": RAW_SCHEMA_VERSION,
        "storage": RAW_STORAGE,
        "status": "complete",
    }
    for field, value in expected.items():
        actual = manifest.get(field)
        if actual != value:
            raise ValueError(
                f"Unsupported raw sidecar manifest {field}={actual!r} in {path}; "
                f"expected {value!r}"
            )


def _validate_zarr_v2(path: Path) -> None:
    zgroup_path = path / ".zgroup"
    if not zgroup_path.is_file():
        raise ValueError(f"Raw sidecar is not a Zarr v2 group (missing {zgroup_path})")
    zgroup = _read_json_mapping(zgroup_path)
    if zgroup.get("zarr_format") != 2:
        raise ValueError(
            f"Raw sidecar must use Zarr v2, got zarr_format={zgroup.get('zarr_format')!r}"
        )


def _validate_zarr_arrays(
    root: Any,
    *,
    manifest: dict[str, Any],
    calibration: dict[str, Any],
    total_frames: int,
    total_episodes: int,
) -> None:
    entries = _manifest_array_entries(manifest)
    manifest_frame_shape = (
        int(manifest["frame_shape"]["height"]),
        int(manifest["frame_shape"]["width"]),
    )
    expected: dict[str, tuple[np.dtype[Any], int]] = {
        "meta/index": (np.dtype("int64"), 1),
        "meta/episode_index": (np.dtype("int64"), 1),
        "meta/frame_index": (np.dtype("int64"), 1),
        "meta/global_frame_index": (np.dtype("int64"), 1),
        "meta/robot_timestamp": (np.dtype("float64"), 1),
        "meta/episode_ends": (np.dtype("int64"), 1),
    }
    for camera in CAMERA_SPECS:
        expected.update(
            {
                f"data/{camera}/depth": (np.dtype("uint16"), 3),
                f"data/{camera}/left_ir": (np.dtype("uint8"), 3),
                f"data/{camera}/right_ir": (np.dtype("uint8"), 3),
                f"data/{camera}/rgbd_timestamp": (np.dtype("float64"), 1),
                f"data/{camera}/rgbd_reused": (np.dtype("bool"), 1),
            }
        )

    for path, (dtype, ndim) in expected.items():
        if path not in root:
            raise KeyError(f"Missing raw sidecar Zarr array: /{path}")
        if path not in entries:
            raise KeyError(f"Raw sidecar manifest arrays is missing /{path}")
        array = root[path]
        if np.dtype(array.dtype) != dtype:
            raise ValueError(f"/{path} dtype {array.dtype} != {dtype}")
        if int(array.ndim) != ndim:
            raise ValueError(f"/{path} rank {array.ndim} != {ndim}")
        expected_length = total_episodes if path == "meta/episode_ends" else total_frames
        if int(array.shape[0]) != expected_length:
            raise ValueError(
                f"/{path} axis-0 length {array.shape[0]} != {expected_length}"
            )
        _validate_manifest_array_entry(path, entries[path], array)

    for camera in CAMERA_SPECS:
        camera_calibration = _camera_calibration(calibration, camera)
        expected_shapes = {
            "depth": _calibration_stream_shape(camera_calibration, "depth"),
            "left_ir": _calibration_stream_shape(camera_calibration, "infrared1"),
            "right_ir": _calibration_stream_shape(camera_calibration, "infrared2"),
        }
        for modality, shape in expected_shapes.items():
            if shape != manifest_frame_shape:
                raise ValueError(
                    f"Manifest frame_shape {manifest_frame_shape} does not match "
                    f"calibration {camera}/{modality} {shape}"
                )
            actual = tuple(int(x) for x in root[f"data/{camera}/{modality}"].shape[1:])
            if actual != shape:
                raise ValueError(
                    f"/data/{camera}/{modality} H/W {actual} does not match "
                    f"calibration {shape}"
                )
        scale = float(camera_calibration["depth_scale_m_per_unit"])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid calibration depth scale for camera {camera!r}: {scale}")

    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if episode_ends.ndim != 1 or episode_ends.size != total_episodes:
        raise ValueError(
            f"/meta/episode_ends must have {total_episodes} values, got {episode_ends.shape}"
        )
    if total_frames > 0:
        if episode_ends.size == 0:
            raise ValueError("Non-empty raw sidecar must contain episode_ends")
        if np.any(np.diff(episode_ends) <= 0):
            raise ValueError("/meta/episode_ends must be strictly increasing")
        if int(episode_ends[-1]) != total_frames:
            raise ValueError(
                f"/meta/episode_ends[-1]={episode_ends[-1]} != committed_frames {total_frames}"
            )
    _validate_zarr_index_order(root, episode_ends, total_frames)


def _validate_zarr_index_order(root: Any, episode_ends: np.ndarray, total_frames: int) -> None:
    chunk_size = max(1, min(4096, total_frames))
    episode_start = 0
    episode = 0
    previous_global: int | None = None
    previous_camera_timestamp: dict[str, float] = {}
    for start in range(0, total_frames, chunk_size):
        stop = min(total_frames, start + chunk_size)
        index = np.asarray(root["meta/index"][start:stop], dtype=np.int64)
        expected_index = np.arange(start, stop, dtype=np.int64)
        _require_array_equal(index, expected_index, name="meta/index ordinal", start=start)
        episode_index = np.asarray(root["meta/episode_index"][start:stop], dtype=np.int64)
        frame_index = np.asarray(root["meta/frame_index"][start:stop], dtype=np.int64)
        for offset, row_index in enumerate(range(start, stop)):
            while episode < len(episode_ends) and row_index >= int(episode_ends[episode]):
                episode_start = int(episode_ends[episode])
                episode += 1
            if episode >= len(episode_ends):
                raise ValueError(f"Row {row_index} lies outside /meta/episode_ends")
            if int(episode_index[offset]) != episode:
                raise ValueError(
                    f"/meta/episode_index row {row_index}={episode_index[offset]} != {episode}"
                )
            expected_frame = row_index - episode_start
            if int(frame_index[offset]) != expected_frame:
                raise ValueError(
                    f"/meta/frame_index row {row_index}={frame_index[offset]} != {expected_frame}"
                )
        global_index = np.asarray(root["meta/global_frame_index"][start:stop], dtype=np.int64)
        if global_index.size:
            if previous_global is not None and int(global_index[0]) <= previous_global:
                raise ValueError("/meta/global_frame_index must be strictly increasing")
            if np.any(np.diff(global_index) <= 0):
                raise ValueError("/meta/global_frame_index must be strictly increasing")
            previous_global = int(global_index[-1])
        robot_timestamp = np.asarray(root["meta/robot_timestamp"][start:stop], dtype=np.float64)
        if not np.isfinite(robot_timestamp).all():
            raise ValueError("/meta/robot_timestamp contains NaN or Inf")
        for camera in CAMERA_SPECS:
            timestamp = np.asarray(
                root[f"data/{camera}/rgbd_timestamp"][start:stop], dtype=np.float64
            )
            if not np.isfinite(timestamp).all():
                raise ValueError(f"/data/{camera}/rgbd_timestamp contains NaN or Inf")
            if timestamp.size:
                previous = previous_camera_timestamp.get(camera)
                if previous is not None and float(timestamp[0]) < previous:
                    raise ValueError(f"/data/{camera}/rgbd_timestamp must be nondecreasing")
                if np.any(np.diff(timestamp) < 0):
                    raise ValueError(f"/data/{camera}/rgbd_timestamp must be nondecreasing")
                previous_camera_timestamp[camera] = float(timestamp[-1])


def _manifest_array_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("arrays")
    if isinstance(raw, list):
        items = []
        for entry in raw:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("Manifest arrays list entries must be mappings with path")
            items.append((entry["path"], entry))
    elif isinstance(raw, dict):
        items = []
        for key, entry in raw.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Manifest array entry {key!r} must be a mapping")
            path = entry.get("path", key)
            if not isinstance(path, str):
                raise ValueError(f"Manifest array entry {key!r} has invalid path")
            items.append((path, entry))
    else:
        raise KeyError("Raw sidecar manifest must contain an arrays mapping or list")
    entries: dict[str, dict[str, Any]] = {}
    for path, entry in items:
        normalized = path.strip("/")
        if not normalized or normalized in entries:
            raise ValueError(f"Duplicate or empty manifest array path: {path!r}")
        entries[normalized] = entry
    return entries


def _validate_manifest_array_entry(path: str, entry: dict[str, Any], array: Any) -> None:
    for field in ("dtype", "shape", "compressor"):
        if field not in entry:
            raise KeyError(f"Manifest array /{path} is missing {field}")
    chunks = entry.get("chunks", entry.get("chunk"))
    if chunks is None:
        raise KeyError(f"Manifest array /{path} is missing chunks")
    try:
        declared_dtype = np.dtype(entry["dtype"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Manifest array /{path} has invalid dtype {entry['dtype']!r}") from exc
    if declared_dtype != np.dtype(array.dtype):
        raise ValueError(
            f"Manifest array /{path} dtype {declared_dtype} != Zarr dtype {array.dtype}"
        )
    declared_shape = tuple(int(x) for x in entry["shape"])
    if declared_shape != tuple(int(x) for x in array.shape):
        raise ValueError(
            f"Manifest array /{path} shape {declared_shape} != Zarr shape {array.shape}"
        )
    declared_chunks = tuple(int(x) for x in chunks)
    if declared_chunks != tuple(int(x) for x in array.chunks):
        raise ValueError(
            f"Manifest array /{path} chunks {declared_chunks} != Zarr chunks {array.chunks}"
        )
    actual_compressor = array.compressor.get_config() if array.compressor is not None else None
    declared_compressor = entry["compressor"]
    if isinstance(declared_compressor, dict):
        if not isinstance(actual_compressor, dict):
            raise ValueError(f"Manifest array /{path} declares a compressor but Zarr has none")
        for key, value in declared_compressor.items():
            if actual_compressor.get(key) != value:
                raise ValueError(
                    f"Manifest array /{path} compressor {key}={value!r} != "
                    f"Zarr {actual_compressor.get(key)!r}"
                )
    elif declared_compressor is None:
        if actual_compressor is not None:
            raise ValueError(f"Manifest array /{path} declares no compressor but Zarr has one")
    else:
        actual_text = json.dumps(actual_compressor, sort_keys=True).lower()
        if str(declared_compressor).lower() not in actual_text:
            raise ValueError(
                f"Manifest array /{path} compressor {declared_compressor!r} "
                f"does not match {actual_compressor!r}"
            )


def _validate_mirrored_attrs(
    attrs: dict[str, Any],
    manifest: dict[str, Any],
    *,
    calibration_sha256: str,
) -> None:
    comparisons = {
        "schema_name": manifest["schema_name"],
        "schema_version": manifest["schema_version"],
        "storage": manifest["storage"],
        "relative_path": manifest["relative_path"],
        "status": manifest["status"],
        "committed_frames": manifest["committed_frames"],
        "committed_episodes": manifest["committed_episodes"],
    }
    for key, expected in comparisons.items():
        if key in attrs and attrs[key] != expected:
            raise ValueError(
                f"Raw sidecar Zarr attr {key}={attrs[key]!r} conflicts with manifest {expected!r}"
            )
    for key in ("calibration_sha256", "calibration_sha_256"):
        if key in attrs and attrs[key] != calibration_sha256:
            raise ValueError(
                f"Raw sidecar Zarr attr {key} conflicts with calibration SHA-256"
            )


def _manifest_calibration_reference(manifest: dict[str, Any]) -> tuple[Any, str]:
    value = manifest.get("calibration", manifest.get("calibration_reference"))
    if isinstance(value, dict):
        relative = value.get("relative_path", value.get("path"))
        digest = value.get("sha256", value.get("sha_256"))
    else:
        relative = manifest.get("calibration_relative_path")
        digest = manifest.get("calibration_sha256", manifest.get("calibration_sha_256"))
    if not isinstance(relative, str) or not relative:
        raise KeyError("Raw sidecar manifest is missing calibration relative_path")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Raw sidecar manifest calibration SHA-256 must be 64 hex characters")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError("Raw sidecar manifest calibration SHA-256 is not hexadecimal") from exc
    return relative, digest.lower()


def _manifest_depth_units(manifest: dict[str, Any]) -> Any:
    if "depth_units" in manifest:
        return manifest["depth_units"]
    if isinstance(manifest.get("depth"), dict) and "units" in manifest["depth"]:
        return manifest["depth"]
    entries = _manifest_array_entries(manifest)
    units = {
        json.dumps(entry.get("units"), sort_keys=True)
        for path, entry in entries.items()
        if path.endswith("/depth") and "units" in entry
    }
    if len(units) == 1:
        return json.loads(next(iter(units)))
    raise KeyError("Raw sidecar manifest is missing unambiguous native depth units")


def _validate_manifest_depth_units(manifest: dict[str, Any]) -> None:
    units = _manifest_depth_units(manifest)
    text = json.dumps(units, sort_keys=True).lower()
    if "uint16" not in text or not any(token in text for token in ("native", "raw", "device")):
        raise ValueError(
            "Raw sidecar depth units must declare native/raw/device uint16 RealSense units"
        )


def _validate_manifest_join_semantics(manifest: dict[str, Any]) -> None:
    join = manifest.get("join_semantics", manifest.get("join"))
    if isinstance(join, dict):
        text = json.dumps(join, sort_keys=True).lower()
        required_groups = [
            ("index",),
            ("episode_index",),
            ("frame_index",),
            ("global_frame_index",),
            ("robot_timestamp", "robot timestamp"),
            ("rgbd_timestamp", "camera timestamp", "per-camera timestamp"),
            ("rgbd_reused", "reused"),
        ]
    else:
        row_semantics = manifest.get("row_semantics")
        if not isinstance(row_semantics, str) or not row_semantics.strip():
            raise KeyError(
                "Raw sidecar manifest must contain join_semantics or row_semantics"
            )
        text = row_semantics.lower()
        required_groups = [
            ("index",),
            ("episode_index",),
            ("frame_index",),
            ("global_frame_index",),
            ("robot_timestamp", "robot timestamp", "robot/"),
            ("rgbd_timestamp", "camera timestamp", "per-camera timestamp", "per-camera timestamps"),
            ("rgbd_reused", "reused"),
        ]
    missing = [group[0] for group in required_groups if not any(token in text for token in group)]
    if missing:
        raise ValueError(f"Raw sidecar manifest row/join semantics is missing keys: {missing}")


def _validate_manifest_layout_declarations(manifest: dict[str, Any]) -> None:
    cameras = tuple(str(value) for value in manifest.get("cameras", []))
    if cameras != tuple(CAMERA_SPECS):
        raise ValueError(
            f"Raw sidecar manifest cameras {cameras} != {tuple(CAMERA_SPECS)}"
        )
    modalities = tuple(str(value) for value in manifest.get("modalities", []))
    expected_modalities = ("depth", "left_ir", "right_ir")
    if modalities != expected_modalities:
        raise ValueError(
            f"Raw sidecar manifest modalities {modalities} != {expected_modalities}"
        )
    frame_shape = manifest.get("frame_shape")
    if not isinstance(frame_shape, dict):
        raise KeyError("Raw sidecar manifest is missing frame_shape")
    height = int(frame_shape.get("height", -1))
    width = int(frame_shape.get("width", -1))
    if height <= 0 or width <= 0:
        raise ValueError(f"Raw sidecar manifest has invalid frame_shape: {frame_shape}")


def _camera_calibration(calibration: dict[str, Any], camera: str) -> dict[str, Any]:
    spec = CAMERA_SPECS[_require_camera(camera)]
    cameras = calibration.get("cameras")
    if not isinstance(cameras, dict):
        raise KeyError("Calibration is missing cameras mapping")
    for key in (spec["calibration_key"], camera):
        value = cameras.get(key)
        if isinstance(value, dict):
            return value
    for value in cameras.values():
        if isinstance(value, dict) and value.get("logical_camera") in {
            camera,
            spec["calibration_key"],
        }:
            return value
    raise KeyError(
        f"Calibration is missing camera {camera!r} ({spec['calibration_key']!r})"
    )


def _calibration_stream_shape(camera_calibration: dict[str, Any], stream_name: str) -> tuple[int, int]:
    streams = camera_calibration.get("streams")
    if not isinstance(streams, dict) or not isinstance(streams.get(stream_name), dict):
        raise KeyError(f"Calibration is missing stream profile {stream_name!r}")
    stream = streams[stream_name]
    intrinsics = stream.get("intrinsics") if isinstance(stream.get("intrinsics"), dict) else {}
    width = intrinsics.get("width", stream.get("width"))
    height = intrinsics.get("height", stream.get("height"))
    if width is None or height is None:
        raise KeyError(f"Calibration stream {stream_name!r} is missing width/height")
    return int(height), int(width)


class _ParquetOrderValidator:
    def __init__(self, total_frames: int) -> None:
        self.total_frames = total_frames
        self.previous_episode: int | None = None
        self.previous_frame: int | None = None
        self.previous_global: int | None = None

    def validate(self, rows: dict[str, list[Any]], *, start: int, source_path: Path) -> None:
        count = len(rows["index"])
        index = np.asarray(rows["index"], dtype=np.int64)
        expected = np.arange(start, start + count, dtype=np.int64)
        _require_array_equal(index, expected, name=f"{source_path}: index ordinal", start=start)
        episodes = np.asarray(rows["episode_index"], dtype=np.int64)
        frames = np.asarray(rows["frame_index"], dtype=np.int64)
        globals_ = np.asarray(rows["global_frame_index"], dtype=np.int64)
        for offset in range(count):
            episode = int(episodes[offset])
            frame = int(frames[offset])
            global_index = int(globals_[offset])
            if episode < 0 or frame < 0:
                raise ValueError(f"{source_path}: negative episode/frame index at row {start + offset}")
            if self.previous_episode is None:
                if episode != 0 or frame != 0:
                    raise ValueError(f"{source_path}: first row must start episode_index=0, frame_index=0")
            elif episode == self.previous_episode:
                if frame != int(self.previous_frame) + 1:
                    raise ValueError(f"{source_path}: frame_index is not contiguous at row {start + offset}")
            elif episode == self.previous_episode + 1:
                if frame != 0:
                    raise ValueError(f"{source_path}: new episode must reset frame_index at row {start + offset}")
            else:
                raise ValueError(f"{source_path}: episode_index is missing/duplicated/out of order at row {start + offset}")
            if self.previous_global is not None and global_index <= self.previous_global:
                raise ValueError(f"{source_path}: global_frame_index must be strictly increasing")
            self.previous_episode = episode
            self.previous_frame = frame
            self.previous_global = global_index


def _validate_sync_values(
    rows: dict[str, list[Any]],
    *,
    camera: str,
    source_path: Path,
) -> None:
    timestamp_column = CAMERA_SPECS[camera]["timestamp_column"]
    timestamps = np.asarray(rows[timestamp_column], dtype=np.float64)
    if not np.isfinite(timestamps).all():
        raise ValueError(f"{source_path}: {timestamp_column} contains NaN or Inf")
    if "robot_timestamp" in rows:
        robot_timestamp = np.asarray(rows["robot_timestamp"], dtype=np.float64)
        if not np.isfinite(robot_timestamp).all():
            raise ValueError(f"{source_path}: robot_timestamp contains NaN or Inf")
    reused_column = CAMERA_SPECS[camera]["reused_column"]
    reused = np.asarray(rows[reused_column])
    if reused.dtype.kind not in {"b"} and any(not isinstance(value, bool) for value in rows[reused_column]):
        raise ValueError(f"{source_path}: {reused_column} must contain bool values")


def _require_array_equal(left: np.ndarray, right: np.ndarray, *, name: str, start: int) -> None:
    if left.shape != right.shape:
        raise ValueError(f"Join mismatch for {name}: shapes {left.shape} != {right.shape}")
    unequal = np.flatnonzero(left != right)
    if unequal.size:
        offset = int(unequal[0])
        raise ValueError(
            f"Join mismatch for {name} at row {start + offset}: "
            f"Parquet={left[offset]!r}, sidecar={right[offset]!r}"
        )


def _safe_manifest_relative_path(value: Any, *, expected: Path | None, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest {field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Manifest {field} must remain inside the dataset root: {value!r}")
    if expected is not None and path.as_posix() != expected.as_posix():
        raise ValueError(
            f"Manifest {field}={path.as_posix()!r} != required {expected.as_posix()!r}"
        )
    return path


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain a mapping: {path}")
    return value


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _required_nonnegative_int(mapping: Mapping[str, Any], key: str, source: str) -> int:
    if key not in mapping or isinstance(mapping[key], bool):
        raise KeyError(f"{source} is missing integer {key}")
    try:
        value = int(mapping[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} {key} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{source} {key} must be non-negative")
    return value


def _as_image(value: Any, dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype:
        try:
            converted = array.astype(expected_dtype, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} cannot be represented as {expected_dtype}") from exc
        if not np.array_equal(array, converted):
            raise ValueError(f"{name} values are not losslessly representable as {expected_dtype}")
        array = converted
    if array.ndim != 2:
        raise ValueError(f"{name} must be HxW, got {array.shape}")
    return array


def _require_camera(camera: str) -> str:
    if camera not in CAMERA_SPECS:
        raise ValueError(f"Unsupported camera: {camera}")
    return camera


def _require_row_index(row_index: int, total_frames: int) -> None:
    if row_index < 0 or row_index >= total_frames:
        raise IndexError(f"row_index {row_index} is outside [0, {total_frames})")


def _require_complete_row_count(actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"Read {actual} scalar Parquet rows but expected {expected}")


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
