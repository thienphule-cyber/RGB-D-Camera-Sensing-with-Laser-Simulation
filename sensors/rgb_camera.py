"""RGB camera sensor (offscreen MuJoCo rendering) and pinhole camera intrinsics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco
import numpy as np

from simulation.environment import Environment

from .common import MUJOCO_TO_OPENCV


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics. Pixel (u, v) = (column, row), pixel centers at integers.

    Camera frame is the OpenCV optical frame: x right, y down, z forward.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fovy(cls, width: int, height: int, fovy_deg: float) -> "CameraIntrinsics":
        """Build intrinsics from MuJoCo's vertical field of view (square pixels)."""
        if width <= 0 or height <= 0:
            raise ValueError("image size must be positive")
        if not 0.0 < fovy_deg < 180.0:
            raise ValueError("fovy_deg must be in (0, 180)")
        f = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
        return cls(f, f, (width - 1) / 2.0, (height - 1) / 2.0, int(width), int(height))

    @property
    def matrix(self) -> np.ndarray:
        """3x3 camera matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    def project(self, points_cam) -> np.ndarray:
        """Project (N, 3) camera-frame points to (N, 2) pixels [u, v]."""
        p = np.asarray(points_cam, dtype=float).reshape(-1, 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * p[:, 0] / p[:, 2] + self.cx
            v = self.fy * p[:, 1] / p[:, 2] + self.cy
        return np.stack([u, v], axis=1)

    def deproject(self, pixels, depth) -> np.ndarray:
        """Back-project (N, 2) pixels with (N,) depth [m] to (N, 3) camera-frame points."""
        uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
        z = np.asarray(depth, dtype=float).reshape(-1)
        x = (uv[:, 0] - self.cx) * z / self.fx
        y = (uv[:, 1] - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=1)


def ground_truth_camera_pose_opencv(env: Environment) -> Tuple[np.ndarray, np.ndarray]:
    """Ground-truth camera pose in world: (position (3,), rotation (3, 3)).

    The rotation columns are the OpenCV camera axes (x right, y down, z forward)
    expressed in the world frame, so p_world = R @ p_cam + position.
    Use for verification and calibration evaluation only, never in the estimator.
    """
    position, rot_mujoco = env.get_camera_pose()
    return position, rot_mujoco @ MUJOCO_TO_OPENCV


class RGBCamera:
    """Fixed RGB camera rendering offscreen images from the Environment."""

    def __init__(
        self,
        env: Environment,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        if noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")
        cfg = env.config
        w = int(width) if width is not None else cfg.camera_resolution[0]
        h = int(height) if height is not None else cfg.camera_resolution[1]

        self._env = env
        self._camera_name = cfg.camera_name
        self.noise_std = float(noise_std)  # additive Gaussian noise [intensity levels, 0-255]
        self._rng = np.random.default_rng(seed)
        self.intrinsics = CameraIntrinsics.from_fovy(w, h, cfg.camera_fovy_deg)
        self._renderer = mujoco.Renderer(env.model, height=h, width=w)
        self._closed = False

    def capture(self) -> np.ndarray:
        """Render the current scene. Returns a uint8 image of shape (H, W, 3), row 0 = top."""
        self._renderer.update_scene(self._env.data, camera=self._camera_name)
        image = np.array(self._renderer.render(), dtype=np.uint8)
        if self.noise_std > 0.0:
            noisy = image.astype(np.float32) + self._rng.normal(0.0, self.noise_std, image.shape)
            image = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
        return image

    def close(self) -> None:
        """Release the OpenGL context."""
        if not self._closed:
            self._renderer.close()
            self._closed = True

    def __enter__(self) -> "RGBCamera":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()