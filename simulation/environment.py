"""MuJoCo environment: arena, fixed sensor assembly and moving obstacles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np

from .obstacle_manager import ObstacleManager, ObstacleSpec, default_obstacles


@dataclass
class EnvironmentConfig:
    """Initial design assumptions (can be tuned later)."""

    arena_size: Tuple[float, float, float] = (5.0, 5.0, 3.0)  # x, y, z [m]
    timestep: float = 0.002  # physics timestep [s]
    duration: float = 60.0  # simulation duration [s]
    sensor_position: Tuple[float, float, float] = (0.0, -3.0, 1.5)  # world position [m]
    camera_name: str = "rgbd_camera"
    lidar_site_name: str = "lidar_site"
    laser_site_name: str = "laser_site"
    laser_offset: Tuple[float, float, float] = (0.0, 0.0, -0.12)
    camera_resolution: Tuple[int, int] = (640, 480)  # (width, height)
    camera_fovy_deg: float = 60.0

    def __post_init__(self) -> None:
        if self.timestep <= 0:
            raise ValueError("timestep must be positive")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if any(s <= 0 for s in self.arena_size):
            raise ValueError("arena_size values must be positive")


class Environment:
    """Fixed sensor assembly observing obstacles on analytic trajectories.

    Obstacles are mocap bodies, so their motion is fully scripted and the
    ground-truth velocity is exact.

    Camera convention: the MuJoCo camera looks along its local -Z axis with +Y
    up. The sensor assembly is oriented so the camera looks along world +Y.
    """

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        obstacles: Optional[Sequence[ObstacleSpec]] = None,
    ) -> None:
        self.config = config or EnvironmentConfig()
        specs = list(obstacles) if obstacles is not None else default_obstacles()
        self.obstacles = ObstacleManager(specs)

        self.model = mujoco.MjModel.from_xml_string(self._build_xml())
        self.data = mujoco.MjData(self.model)
        self.obstacles.bind(self.model)

        self._camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera_name
        )
        if self._camera_id < 0:
            raise RuntimeError(f"camera {self.config.camera_name!r} not found")

        self.reset()

    # ------------------------------------------------------------------ model
    def _build_xml(self) -> str:
        cfg = self.config
        ax, ay, _ = cfg.arena_size
        sx, sy, sz = cfg.sensor_position
        width, height = cfg.camera_resolution
        lx, ly, lz = cfg.laser_offset
        return f"""
<mujoco model="multi_object_tracking">
    <option timestep="{cfg.timestep}" gravity="0 0 -9.81"/>
    <visual>
    <global offwidth="{width}" offheight="{height}"/>
    </visual>
    <worldbody>
    <light pos="0 0 2.9" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="{ax / 2} {ay / 2} 0.1" rgba="0.75 0.75 0.75 1"/>

    <!-- Fixed sensor assembly. xyaxes makes the camera look along world +Y, Z up. -->
    <body name="sensor_assembly" pos="{sx} {sy} {sz}">
        <camera name="{cfg.camera_name}" pos="0 0 0" xyaxes="1 0 0 0 0 1" fovy="{cfg.camera_fovy_deg}"/>
        <site name="{cfg.lidar_site_name}" pos="0 0 0.12" size="0.02" rgba="0 0 1 1"/>
        <site name="{cfg.laser_site_name}" pos="{lx} {ly} {lz}" size="0.02" rgba="1 0 0 1"/>
        <!-- Housing sits behind the camera so it never blocks the view. -->
        <geom name="sensor_housing" type="box" pos="0 -0.08 0" size="0.1 0.03 0.08"
            rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
    </body>

    <!-- Moving obstacles (mocap bodies) -->
{self.obstacles.build_mjcf_bodies()}
    </worldbody>
</mujoco>
"""

    # ------------------------------------------------------------- simulation
    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(self) -> None:
        """Reset to t = 0 and place obstacles at their initial poses."""
        mujoco.mj_resetData(self.model, self.data)
        self.obstacles.update(self.data, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        """Advance one physics step and refresh obstacle poses for the new time."""
        mujoco.mj_step(self.model, self.data)
        self.obstacles.update(self.data, self.data.time)
        # Refresh kinematics so xpos / camera poses match the new mocap positions.
        mujoco.mj_forward(self.model, self.data)

    def steps_per_period(self, rate_hz: float) -> int:
        """Number of physics steps between two events at rate_hz (e.g. 30 Hz camera)."""
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        return max(1, int(round(1.0 / (rate_hz * self.config.timestep))))

    # ------------------------------------------------------------ sensor poses
    def get_camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ground-truth camera pose in world: (position (3,), rotation (3, 3)).

        Rotation columns are the camera axes in MuJoCo convention (looks along -Z).
        Use for evaluating calibration only, never inside the estimation pipeline.
        """
        pos = self.data.cam_xpos[self._camera_id].copy()
        rot = self.data.cam_xmat[self._camera_id].reshape(3, 3).copy()
        return pos, rot

    def get_site_pose(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Ground-truth world pose of a site (e.g. lidar_site, laser_site)."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"site {site_name!r} not found")
        pos = self.data.site_xpos[site_id].copy()
        rot = self.data.site_xmat[site_id].reshape(3, 3).copy()
        return pos, rot