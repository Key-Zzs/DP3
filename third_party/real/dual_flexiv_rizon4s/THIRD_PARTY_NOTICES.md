# Third-Party Notices

## Standalone RealSense runtime

`interface/realsense_camera.py` is adapted from the Intel RealSense camera
implementation in Hugging Face LeRobot and the `Key-Zzs/Le-nero` fork.

- Original project: Hugging Face LeRobot
- Fork: `https://github.com/Key-Zzs/Le-nero`
- Fork package version: `0.3.4`
- Source commit: `0045cb8feb09757be617b00c1322f1714c6921ce`
- License: Apache License 2.0; see `LICENSE.lerobot`

Source mapping:

| Upstream source | Local source |
| --- | --- |
| `src/lerobot/cameras/realsense/configuration_realsense.py` | `interface/realsense_camera.py` |
| `src/lerobot/cameras/realsense/camera_realsense.py` | `interface/realsense_camera.py` |

The fork commit above adds the coherent RGB-D/IR frameset behavior required by
the live DP3 observation contract. The local adaptation preserves that frame
contract while removing the upstream camera base class, configuration registry,
device factory, package imports, and runtime source-path injection. It also
defers name-to-serial discovery until `connect()`, adds startup cleanup, and uses
local connection-error types.

The rest of this repository remains under its existing licenses. The adapted
camera source retains its Apache-2.0 header and is not relicensed as MIT-only.
