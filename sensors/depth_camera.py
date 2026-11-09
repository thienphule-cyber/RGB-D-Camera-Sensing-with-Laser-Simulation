"""Depth camera sensor (offscreen MuJoCo depth rendering) and depth deprojection."""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from simulation.environment import Environment

from .rgb_camera import CameraIntrinsics


def depth_to_points(depth: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Back-project a depth image to an organized point cloud.

    Returns an (H, W, 3) array in the OpenCV camera frame; invalid pixels are NaN.
    """
    h, w = depth.shape
    if (w, h) != (intrinsics.width, intrinsics.height):
        raise ValueError("depth image size does not match the intrinsics")
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = depth.astype(np.float64)
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=-1)


class DepthCamera:
    """Fixed depth camera sharing the pose and intrinsics of the RGB camera.

    Depth is the distance along the optical axis [m]. Pixels outside
    [min_depth, max_depth] (including the background) are NaN.
    Depth noise: sigma = noise_std + noise_std_per_m2 * depth^2.
    """

    def __init__(
        self,
        env: Environment,
        min_depth: float = 0.1,
        max_depth: float = 8.0,
        noise_std: float = 0.0,
        noise_std_per_m2: float = 0.0,
        dropout_prob: float = 0.0,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        if not 0.0 <= min_depth < max_depth:
            raise ValueError("need 0 <= min_depth < max_depth")
        if noise_std < 0.0 or noise_std_per_m2 < 0.0:
            raise ValueError("noise parameters must be >= 0")
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be in [0, 1]")

        cfg = env.config
        w = int(width) if width is not None else cfg.camera_resolution[0]
        h = int(height) if height is not None else cfg.camera_resolution[1]

        self._env = env
        self._camera_name = cfg.camera_name
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.noise_std = float(noise_std)
        self.noise_std_per_m2 = float(noise_std_per_m2)
        self.dropout_prob = float(dropout_prob)
        self._rng = np.random.default_rng(seed)
        self.intrinsics = CameraIntrinsics.from_fovy(w, h, cfg.camera_fovy_deg)
        self._renderer = mujoco.Renderer(env.model, height=h, width=w)
        self._renderer.enable_depth_rendering()
        self._closed = False

    def capture(self) -> np.ndarray:
        """Render a depth image of shape (H, W), float32 meters, NaN = no measurement."""
        self._renderer.update_scene(self._env.data, camera=self._camera_name)
        depth = np.array(self._renderer.render(), dtype=np.float32)

        with np.errstate(invalid="ignore"):
            invalid = ~np.isfinite(depth) | (depth < self.min_depth) | (depth > self.max_depth)
        depth[invalid] = np.nan

        valid = np.isfinite(depth)
        if self.noise_std > 0.0 or self.noise_std_per_m2 > 0.0:
            sigma = self.noise_std + self.noise_std_per_m2 * depth.astype(np.float64) ** 2
            noise = self._rng.normal(0.0, 1.0, depth.shape)
            depth[valid] += (sigma * noise)[valid]
        if self.dropout_prob > 0.0:
            depth[self._rng.random(depth.shape) < self.dropout_prob] = np.nan
        return depth

    def capture_point_cloud(self) -> np.ndarray:
        """Capture and back-project. Returns valid points (N, 3) in the OpenCV camera frame."""
        points = depth_to_points(self.capture(), self.intrinsics).reshape(-1, 3)
        return points[np.isfinite(points).all(axis=1)]

    def close(self) -> None:
        """Release the OpenGL context."""
        if not self._closed:
            self._renderer.close()
            self._closed = True

    def __enter__(self) -> "DepthCamera":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()