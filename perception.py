import cv2
import numpy as np
from viam.components.camera import Camera

DEPTH_HEADER_BYTES = 24  # "DEPTHMAP" magic + uint64 width + uint64 height, big-endian


def decode_depth(data: bytes) -> np.ndarray:
    """Decode Viam's image/vnd.viam.dep payload into an (H, W) uint16 array in millimeters."""
    width = int.from_bytes(data[8:16], "big")
    height = int.from_bytes(data[16:24], "big")
    return np.frombuffer(data[DEPTH_HEADER_BYTES:], dtype=">u2").reshape(height, width).astype(np.uint16)


def decode_color(data: bytes) -> np.ndarray:
    """Decode a JPEG/PNG payload into a BGR array."""
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


async def capture(cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Grab a simultaneous (color BGR, depth mm) pair from the camera."""
    images, _ = await cam.get_images()
    by_name = {img.name: img for img in images}
    return decode_color(by_name["color"].data), decode_depth(by_name["depth"].data)


def deproject(depth: np.ndarray, intr, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Back-project valid depth pixels into camera-frame points (mm).

    Returns (points (N, 3), pixels (N, 2) as u, v). `intr` is the camera's IntrinsicParameters.
    """
    d = depth[::step, ::step].astype(np.float32)
    v, u = np.nonzero(d)
    z = d[v, u]
    u, v = u * step, v * step
    x = (u - intr.center_x_px) * z / intr.focal_x_px
    y = (v - intr.center_y_px) * z / intr.focal_y_px
    return np.stack([x, y, z], axis=1), np.stack([u, v], axis=1)


def pixel_to_cam(u: float, v: float, depth: np.ndarray, intr, win: int = 3) -> np.ndarray | None:
    """Camera-frame point (mm) for one pixel, using the median valid depth in a (2*win+1)^2 window."""
    patch = depth[max(int(v) - win, 0):int(v) + win + 1, max(int(u) - win, 0):int(u) + win + 1]
    patch = patch[patch > 0]
    if patch.size == 0:
        return None
    z = float(np.median(patch))
    return np.array([(u - intr.center_x_px) * z / intr.focal_x_px,
                     (v - intr.center_y_px) * z / intr.focal_y_px, z])
