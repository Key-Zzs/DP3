"""Spawned telemetry consumer process and small control protocol."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Mapping
from typing import Any, Callable

from .config import MonitorConfig
from .shared_ring import SharedMemoryTelemetryBus

LOGGER = logging.getLogger(__name__)


class NullSink:
    """Synthetic sink used by tests and transport benchmarks."""

    def __init__(self, **_: Any) -> None:
        self.processed = 0

    def start(self) -> None:
        return None

    def consume(self, snapshot: Any) -> None:
        self.processed += 1

    def close(self) -> None:
        return None


def _make_sink(kind: str, config: MonitorConfig, static_metadata: Mapping[str, Any], sink_factory: Callable[..., Any] | None) -> Any:
    if sink_factory is not None:
        return sink_factory(config=config, static_metadata=static_metadata)
    if kind in {"null", "in_memory", "fake"}:
        return NullSink(config=config, static_metadata=static_metadata)
    if kind != "rerun":
        raise ValueError(f"Unsupported telemetry sink: {kind!r}")
    # The import is deliberately inside the child entry path.  Importing the
    # producer-side monitor package never imports rerun.
    from .rerun_sink import RerunSink

    return RerunSink(config=config, static_metadata=static_metadata)


def _send_control(connection: Any, message: Mapping[str, Any]) -> None:
    try:
        connection.send(dict(message))
    except (BrokenPipeError, EOFError, OSError):
        pass


def _telemetry_child_entry(
    descriptor: Mapping[str, Any],
    config: MonitorConfig,
    static_metadata: Mapping[str, Any],
    sink_kind: str,
    connection: Any,
    stop_event: Any,
    sink_factory: Callable[..., Any] | None,
) -> None:
    bus: SharedMemoryTelemetryBus | None = None
    sink: Any | None = None
    try:
        bus = SharedMemoryTelemetryBus.attach(descriptor)
        sink = _make_sink(sink_kind, config, static_metadata, sink_factory)
        sink.start()
        _send_control(connection, {"type": "ready", "pid": mp.current_process().pid})
        last_sequences = {name: -1 for name in bus.channel_names()}
        processed_counts = {name: 0 for name in bus.channel_names()}
        next_heartbeat = time.monotonic() + float(config.heartbeat_interval_sec)
        while not stop_event.is_set():
            if time.monotonic() >= next_heartbeat:
                _send_control(
                    connection,
                    {
                        "type": "heartbeat",
                        "pid": mp.current_process().pid,
                        "latest_sequences": dict(last_sequences),
                        "processed_counts": dict(processed_counts),
                    },
                )
                next_heartbeat = time.monotonic() + float(config.heartbeat_interval_sec)
            processed = False
            for name in bus.channel_names():
                snapshot = bus.consume_latest(name, after_sequence=last_sequences[name])
                if snapshot is None:
                    continue
                # This is the number of stale records skipped to reach the
                # latest slot. It remains telemetry-local and never feeds back
                # into the producer/control process.
                snapshot.metadata["consumer_lag_cycles"] = max(
                    0,
                    int(snapshot.sequence) - int(last_sequences[name]) - 1,
                )
                last_sequences[name] = snapshot.sequence
                processed_counts[name] += 1
                processed = True
                try:
                    sink.consume(snapshot)
                except Exception as exc:  # noqa: BLE001
                    # A viewer/sink failure is telemetry-local.  Disable the
                    # sink and continue consuming latest-only records so the
                    # producer remains independent of the failure.
                    _send_control(connection, {"type": "sink_error", "error": repr(exc)})
                    try:
                        sink.close()
                    except Exception:  # noqa: BLE001
                        pass
                    reconnect = getattr(sink, "try_reconnect", None)
                    if callable(reconnect) and reconnect():
                        _send_control(connection, {"type": "viewer_reconnected"})
                    else:
                        sink = NullSink()
                        sink.start()
            if not processed:
                stop_event.wait(0.005)
        _send_control(connection, {"type": "stopped"})
    except BaseException as exc:  # noqa: BLE001
        _send_control(connection, {"type": "failed", "error": repr(exc)})
    finally:
        if sink is not None:
            try:
                sink.close()
            except Exception:  # noqa: BLE001
                pass
        if bus is not None:
            # Child closes attachments but never unlinks parent-owned segments.
            bus.close()
        try:
            connection.close()
        except (OSError, ValueError):
            pass


class TelemetryProcess:
    """Parent-side lifecycle wrapper around a non-daemon spawn child."""

    def __init__(
        self,
        bus_descriptor: Mapping[str, Any],
        *,
        config: MonitorConfig,
        static_metadata: Mapping[str, Any] | None = None,
        sink_kind: str = "rerun",
        sink_factory: Callable[..., Any] | None = None,
        context: Any | None = None,
    ) -> None:
        self._descriptor = bus_descriptor
        self.config = config
        self.static_metadata = dict(static_metadata or {})
        self.sink_kind = sink_kind
        self.sink_factory = sink_factory
        self._context = context or mp.get_context("spawn")
        self._parent_connection, child_connection = self._context.Pipe(duplex=False)
        self._stop_event = self._context.Event()
        self._process = self._context.Process(
            target=_telemetry_child_entry,
            args=(
                self._descriptor,
                self.config,
                self.static_metadata,
                self.sink_kind,
                child_connection,
                self._stop_event,
                self.sink_factory,
            ),
            name="dp3_telemetry",
            daemon=False,
        )
        self._started = False
        self._closed = False
        self._ready = False
        self._messages: list[dict[str, Any]] = []
        self._last_heartbeat: dict[str, Any] | None = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._started else None

    @property
    def process(self) -> Any:
        return self._process

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def last_heartbeat(self) -> dict[str, Any] | None:
        return None if self._last_heartbeat is None else dict(self._last_heartbeat)

    def is_alive(self) -> bool:
        return bool(self._started and self._process.is_alive())

    def start(self, *, timeout: float | None = None) -> None:
        if self._started:
            return
        self._process.start()
        self._started = True
        timeout = self.config.startup_timeout_sec if timeout is None else float(timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._parent_connection.poll(min(0.05, max(0.0, deadline - time.monotonic()))):
                message = dict(self._parent_connection.recv())
                self._messages.append(message)
                kind = message.get("type")
                if kind == "ready":
                    self._ready = True
                    return
                if kind == "failed":
                    raise RuntimeError(f"Telemetry child startup failed: {message.get('error')}")
            if not self._process.is_alive():
                raise RuntimeError(f"Telemetry child exited during startup with code {self._process.exitcode}")
        raise TimeoutError(f"Telemetry child did not become ready within {timeout:.2f}s")

    def poll_messages(self) -> list[dict[str, Any]]:
        messages = list(self._messages)
        self._messages.clear()
        while self._parent_connection.poll():
            try:
                messages.append(dict(self._parent_connection.recv()))
                if messages[-1].get("type") == "heartbeat":
                    self._last_heartbeat = messages[-1]
            except (EOFError, OSError):
                break
        return messages

    def close(self, *, timeout: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        timeout = self.config.shutdown_timeout_sec if timeout is None else float(timeout)
        if self._started and self._process.is_alive():
            self._stop_event.set()
            self._process.join(timeout=max(0.0, timeout))
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=max(0.0, min(1.0, timeout)))
        try:
            self._parent_connection.close()
        except (OSError, ValueError):
            pass


__all__ = ["NullSink", "TelemetryProcess"]
