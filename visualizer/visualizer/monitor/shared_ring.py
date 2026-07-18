"""Bounded, latest-only shared-memory SPSC channels.

Only fixed-size NumPy arrays cross the process boundary.  The small descriptor
and multiprocessing locks are control metadata; payload arrays never enter a
Queue, Pipe, pickle, or Manager.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Mapping

import numpy as np

from .schema import META_DTYPE, ChannelSnapshot, ChannelSpec

LOGGER = logging.getLogger(__name__)


def _shm_open(*, name: str | None = None, size: int | None = None, create: bool) -> SharedMemory:
    kwargs: dict[str, Any] = {"create": create}
    if create:
        kwargs["size"] = int(size or 0)
    else:
        kwargs["name"] = name
    try:
        # Python 3.13 supports track=False, preventing an attached child from
        # racing the parent's unlink with its resource tracker.
        if not create:
            kwargs["track"] = False
        return SharedMemory(**kwargs)
    except TypeError:
        kwargs.pop("track", None)
        return SharedMemory(**kwargs)


@dataclass(frozen=True)
class PublishResult:
    committed: bool
    sequence: int
    overwritten: bool = False
    lock_contention: bool = False


class _RingChannel:
    def __init__(
        self,
        spec: ChannelSpec,
        capacity: int,
        *,
        context: Any,
        descriptor: Mapping[str, Any] | None = None,
        owner: bool,
    ) -> None:
        self.spec = spec
        self.capacity = int(capacity)
        self.owner = bool(owner)
        self._locks = list(descriptor["locks"]) if descriptor is not None else [context.Lock() for _ in range(self.capacity)]
        self._shm: list[SharedMemory] = []
        self._arrays: dict[str, np.ndarray] = {}
        self._meta_shm: SharedMemory
        self._meta: np.ndarray
        self._next_sequence = 0
        self.published = 0
        self.dropped = 0
        self.overwritten = 0
        if descriptor is None:
            self._meta_shm = _shm_open(create=True, size=META_DTYPE.itemsize * self.capacity)
            self._meta = np.ndarray((self.capacity,), dtype=META_DTYPE, buffer=self._meta_shm.buf)
            self._meta[...] = 0
            self._meta["sequence"] = -1
            self._meta["consumed_sequence"] = -1
            for field in spec.fields:
                dtype = np.dtype(field.dtype)
                size = self.capacity * int(np.prod(field.shape, dtype=np.int64)) * dtype.itemsize
                shm = _shm_open(create=True, size=size)
                self._shm.append(shm)
                self._arrays[field.name] = np.ndarray(
                    (self.capacity, *field.shape), dtype=dtype, buffer=shm.buf
                )
                self._arrays[field.name][...] = 0
        else:
            self._meta_shm = _shm_open(name=str(descriptor["meta_name"]), create=False)
            self._meta = np.ndarray((self.capacity,), dtype=META_DTYPE, buffer=self._meta_shm.buf)
            for field, name in zip(spec.fields, descriptor["field_names"]):
                shm = _shm_open(name=str(name), create=False)
                self._shm.append(shm)
                self._arrays[field.name] = np.ndarray(
                    (self.capacity, *field.shape), dtype=np.dtype(field.dtype), buffer=shm.buf
                )

    def descriptor(self) -> dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "capacity": self.capacity,
            "meta_name": self._meta_shm.name,
            "field_names": [shm.name for shm in self._shm],
            "locks": self._locks,
        }

    def publish(
        self,
        arrays: Mapping[str, Any] | None = None,
        *,
        cycle_id: int = 0,
        observation_timestamp: float = 0.0,
        monotonic_timestamp: float = 0.0,
        wall_timestamp: float = 0.0,
        valid_flags: int = 0,
        status: int = 0,
        point_count: int = 0,
        point_count_2: int = 0,
        frame_index: int = -1,
        prediction_id: int = -1,
        horizon_length: int = 0,
        selected_index: int = -1,
        commanded_valid: bool = False,
        depth_scale: float = 0.0,
        event_message: str = "",
    ) -> PublishResult:
        sequence = self._next_sequence
        self._next_sequence += 1
        arrays = arrays or {}
        start = sequence % self.capacity
        for offset in range(self.capacity):
            index = (start + offset) % self.capacity
            lock = self._locks[index]
            if not lock.acquire(block=False):
                continue
            try:
                old_sequence = int(self._meta[index]["sequence"])
                old_consumed_sequence = int(self._meta[index]["consumed_sequence"])
                overwrites_unconsumed = old_sequence >= 0 and old_consumed_sequence != old_sequence
                for field in self.spec.fields:
                    if field.name not in arrays or arrays[field.name] is None:
                        continue
                    source = np.asarray(arrays[field.name])
                    target = self._arrays[field.name][index]
                    if field.shape == ():
                        # Indexing a 1-D scalar field returns a NumPy scalar;
                        # retain a writable 0-D ndarray view for np.copyto.
                        target = self._arrays[field.name][index : index + 1].reshape(())
                    if source.shape != target.shape:
                        raise ValueError(
                            f"{self.spec.name}.{field.name} shape {source.shape} != {target.shape}"
                        )
                    np.copyto(target, source, casting="unsafe")
                metadata = self._meta[index]
                metadata["cycle_id"] = int(cycle_id)
                metadata["observation_timestamp"] = float(observation_timestamp)
                metadata["monotonic_timestamp"] = float(monotonic_timestamp)
                metadata["wall_timestamp"] = float(wall_timestamp)
                metadata["valid_flags"] = int(valid_flags)
                metadata["status"] = int(status)
                metadata["event_text"] = str(event_message).encode("utf-8")[:255]
                metadata["point_count"] = int(point_count)
                metadata["point_count_2"] = int(point_count_2)
                metadata["frame_index"] = int(frame_index)
                metadata["prediction_id"] = int(prediction_id)
                metadata["horizon_length"] = int(horizon_length)
                metadata["selected_index"] = int(selected_index)
                metadata["commanded_valid"] = int(bool(commanded_valid))
                metadata["depth_scale"] = float(depth_scale)
                if overwrites_unconsumed:
                    self.overwritten += 1
                metadata["dropped_count"] = int(self.dropped)
                metadata["overwritten_count"] = int(self.overwritten)
                metadata["consumed_sequence"] = -1
                # Commit marker is written last while the slot lock is held.
                metadata["sequence"] = int(sequence)
                self.published += 1
                return PublishResult(True, sequence, overwrites_unconsumed)
            finally:
                lock.release()
        self.dropped += 1
        return PublishResult(False, sequence, lock_contention=True)

    def consume_latest(self, *, after_sequence: int = -1) -> ChannelSnapshot | None:
        # First identify the newest committed slot without copying any payload.
        # Then reacquire only that slot and copy it once. This keeps the lock
        # hold bounded and avoids copying every increasingly-new slot observed
        # during a scan.
        best_sequence = int(after_sequence)
        best_index: int | None = None
        for index, lock in enumerate(self._locks):
            if not lock.acquire(block=False):
                continue
            try:
                sequence = int(self._meta[index]["sequence"])
                if sequence > best_sequence:
                    best_sequence = sequence
                    best_index = index
            finally:
                lock.release()
        if best_index is None:
            return None
        lock = self._locks[best_index]
        if not lock.acquire(block=False):
            return None
        try:
            sequence = int(self._meta[best_index]["sequence"])
            if sequence <= int(after_sequence):
                return None
            metadata = {
                name: self._meta[best_index][name].item()
                for name in META_DTYPE.names or ()
                if name not in {"sequence", "consumed_sequence"}
            }
            arrays = {
                field.name: np.array(self._arrays[field.name][best_index], copy=True)
                for field in self.spec.fields
            }
            self._meta[best_index]["consumed_sequence"] = sequence
            return ChannelSnapshot(self.spec.name, sequence, metadata, arrays)
        finally:
            lock.release()

    def close(self) -> None:
        for shm in (*self._shm, self._meta_shm):
            try:
                shm.close()
            except (OSError, ValueError):
                pass

    def unlink(self) -> None:
        if not self.owner:
            return
        for shm in (*self._shm, self._meta_shm):
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except (OSError, ValueError):
                LOGGER.debug("shared memory unlink failed", exc_info=True)

    def stats(self) -> dict[str, int]:
        return {
            "published": int(self.published),
            "dropped": int(self.dropped),
            "overwritten": int(self.overwritten),
        }


class SharedMemoryTelemetryBus:
    """Owner-side creator and child-side attacher for telemetry channels."""

    def __init__(
        self,
        schema: Any,
        *,
        capacity: int = 3,
        context: Any | None = None,
        descriptor: Mapping[str, Any] | None = None,
        owner: bool = True,
    ) -> None:
        self.schema = schema
        self.capacity = int(capacity if descriptor is None else descriptor["capacity"])
        self.owner = bool(owner)
        self._closed = False
        context = context or __import__("multiprocessing").get_context("spawn")
        if descriptor is None:
            self._channels = {
                spec.name: _RingChannel(spec, self.capacity, context=context, owner=True)
                for spec in schema.channels
            }
        else:
            self._channels = {
                name: _RingChannel(
                    _spec_from_dict(channel["spec"]),
                    int(channel["capacity"]),
                    context=context,
                    descriptor=channel,
                    owner=False,
                )
                for name, channel in descriptor["channels"].items()
            }

    @classmethod
    def create(cls, schema: Any, *, capacity: int = 3, context: Any | None = None) -> "SharedMemoryTelemetryBus":
        return cls(schema, capacity=capacity, context=context, owner=True)

    @classmethod
    def attach(cls, descriptor: Mapping[str, Any], *, context: Any | None = None) -> "SharedMemoryTelemetryBus":
        schema = type("Schema", (), {"channels": tuple(_spec_from_dict(item["spec"]) for item in descriptor["channels"].values())})()
        return cls(schema, context=context, descriptor=descriptor, owner=False)

    def descriptor(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "channels": {name: channel.descriptor() for name, channel in self._channels.items()},
        }

    def publish(self, channel: str, arrays: Mapping[str, Any] | None = None, **metadata: Any) -> PublishResult:
        if self._closed:
            return PublishResult(False, -1)
        return self._channels[channel].publish(arrays, **metadata)

    def consume_latest(self, channel: str, *, after_sequence: int = -1) -> ChannelSnapshot | None:
        if self._closed:
            return None
        return self._channels[channel].consume_latest(after_sequence=after_sequence)

    def channel_names(self) -> tuple[str, ...]:
        return tuple(self._channels)

    def stats(self) -> dict[str, dict[str, int]]:
        return {name: channel.stats() for name, channel in self._channels.items()}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for channel in self._channels.values():
            channel.close()

    def unlink(self) -> None:
        if not self.owner:
            return
        for channel in self._channels.values():
            channel.unlink()


def _spec_from_dict(raw: Mapping[str, Any]) -> ChannelSpec:
    from .schema import FieldSpec

    return ChannelSpec(
        str(raw["name"]),
        tuple(
            FieldSpec(str(field["name"]), tuple(int(x) for x in field["shape"]), str(field["dtype"]))
            for field in raw["fields"]
        ),
    )


__all__ = ["PublishResult", "SharedMemoryTelemetryBus"]
