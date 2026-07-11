"""Connection errors shared by the standalone Flexiv and camera adapters."""


class DeviceNotConnectedError(ConnectionError):
    """Raised when an operation requires a connected device."""


class DeviceAlreadyConnectedError(ConnectionError):
    """Raised when connect is called for an already-connected device."""
