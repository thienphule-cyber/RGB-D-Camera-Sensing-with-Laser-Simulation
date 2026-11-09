"""Obstacle definitions and mocap-based pose updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import mujoco
import numpy as np

from .motion_generator import (
    CircularTrajectory,
    LinearBounceTrajectory,
    SinusoidalTrajectory,
    Trajectory,
)

VALID_GEOMETRIES = ("box", "sphere", "cylinder")
VALID_MOTION_DIMS = ("2D", "3D")

# Expected number of size values per geometry (MuJoCo geom size convention):
#   box: half extents (x, y, z) | sphere: (radius,) | cylinder: (radius, half_height)
_SIZE_LENGTH = {"box": 3, "sphere": 1, "cylinder": 2}


@dataclass
class ObstacleSpec:
    """Static description of one moving obstacle."""

    obstacle_id: str
    geometry: str
    size: Tuple[float, ...]
    motion_dim: str
    trajectory: Trajectory
    rgba: Tuple[float, float, float, float] = (0.8, 0.3, 0.3, 1.0)

    def __post_init__(self) -> None:
        if self.geometry not in VALID_GEOMETRIES:
            raise ValueError(f"geometry must be one of {VALID_GEOMETRIES}, got {self.geometry!r}")
        if self.motion_dim not in VALID_MOTION_DIMS:
            raise ValueError(
                f"motion_dim must be one of {VALID_MOTION_DIMS}, got {self.motion_dim!r}"
            )
        if len(self.size) != _SIZE_LENGTH[self.geometry]:
            raise ValueError(
                f"{self.geometry} needs {_SIZE_LENGTH[self.geometry]} size values, "
                f"got {len(self.size)}"
            )
        if any(s <= 0 for s in self.size):
            raise ValueError("all size values must be positive")


def default_obstacles() -> List[ObstacleSpec]:
    """Six obstacles matching the design table (O001-O006).

    Arena is 5 x 5 x 3 m centered at the origin in XY (floor at z = 0).
    """
    pi = np.pi
    return [
        # O001: box, 2D, linear bouncing motion on the XY plane.
        ObstacleSpec(
            obstacle_id="O001",
            geometry="box",
            size=(0.15, 0.15, 0.15),
            motion_dim="2D",
            rgba=(0.85, 0.2, 0.2, 1.0),
            trajectory=LinearBounceTrajectory(
                start=(-1.5, -0.5, 0.15),
                velocity=(0.40, 0.25, 0.0),
                bounds_min=(-2.2, -2.2, 0.15),
                bounds_max=(2.2, 2.2, 0.15),
            ),
        ),
        # O002: sphere, 3D, sinusoidal (Lissajous) motion in XYZ.
        ObstacleSpec(
            obstacle_id="O002",
            geometry="sphere",
            size=(0.12,),
            motion_dim="3D",
            rgba=(0.2, 0.6, 0.9, 1.0),
            trajectory=SinusoidalTrajectory(
                center=(1.0, 0.5, 1.2),
                amplitude=(0.8, 0.6, 0.5),
                omega=(0.5, 0.7, 0.9),
                phase=(0.0, pi / 3, pi / 2),
            ),
        ),
        # O003: cylinder, 2D, circular motion on the XY plane.
        ObstacleSpec(
            obstacle_id="O003",
            geometry="cylinder",
            size=(0.12, 0.20),
            motion_dim="2D",
            rgba=(0.3, 0.8, 0.3, 1.0),
            trajectory=CircularTrajectory(
                center=(0.0, 0.5, 0.20), radius=1.2, omega=0.4, phase=0.0
            ),
        ),
        # O004: box, 3D, linear bouncing motion that changes X, Y and Z.
        ObstacleSpec(
            obstacle_id="O004",
            geometry="box",
            size=(0.12, 0.12, 0.12),
            motion_dim="3D",
            rgba=(0.9, 0.7, 0.1, 1.0),
            trajectory=LinearBounceTrajectory(
                start=(-1.0, 1.5, 0.8),
                velocity=(0.30, -0.35, 0.25),
                bounds_min=(-2.2, -2.2, 0.3),
                bounds_max=(2.2, 2.2, 2.5),
            ),
        ),
        # O005: sphere, 3D, circular motion with Z oscillation (nonlinear).
        ObstacleSpec(
            obstacle_id="O005",
            geometry="sphere",
            size=(0.10,),
            motion_dim="3D",
            rgba=(0.7, 0.3, 0.8, 1.0),
            trajectory=CircularTrajectory(
                center=(0.5, 0.5, 1.5),
                radius=0.9,
                omega=-0.8,
                phase=pi / 2,
                z_amplitude=0.4,
                z_omega=1.3,
            ),
        ),
        # O006: cylinder, 2D, variable velocity on the XY plane (Lissajous curve).
        ObstacleSpec(
            obstacle_id="O006",
            geometry="cylinder",
            size=(0.10, 0.20),
            motion_dim="2D",
            rgba=(0.2, 0.8, 0.8, 1.0),
            trajectory=SinusoidalTrajectory(
                center=(0.0, 0.0, 0.20),
                amplitude=(1.8, 1.2, 0.0),
                omega=(0.6, 0.9, 0.0),
                phase=(pi / 4, 0.0, 0.0),
            ),
        ),
    ]


class ObstacleManager:
    """Builds obstacle MJCF and drives mocap bodies along their trajectories."""

    def __init__(self, specs: Sequence[ObstacleSpec]) -> None:
        ids = [s.obstacle_id for s in specs]
        if len(set(ids)) != len(ids):
            raise ValueError("obstacle ids must be unique")
        self._specs: List[ObstacleSpec] = list(specs)
        self._by_id: Dict[str, ObstacleSpec] = {s.obstacle_id: s for s in self._specs}
        self._body_ids: Dict[str, int] = {}
        self._mocap_ids: Dict[str, int] = {}

    # ------------------------------------------------------------------ queries
    @property
    def ids(self) -> List[str]:
        return [s.obstacle_id for s in self._specs]

    @property
    def specs(self) -> List[ObstacleSpec]:
        return list(self._specs)

    def get_spec(self, obstacle_id: str) -> ObstacleSpec:
        return self._by_id[obstacle_id]

    def body_id(self, obstacle_id: str) -> int:
        """MuJoCo body id (valid after bind())."""
        return self._body_ids[obstacle_id]

    def state_at(self, obstacle_id: str, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Exact (position, velocity) of an obstacle at time t."""
        return self._by_id[obstacle_id].trajectory.state(t)

    def states_at(self, t: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {oid: self.state_at(oid, t) for oid in self.ids}

    # ---------------------------------------------------------------- MJCF/model
    def build_mjcf_bodies(self) -> str:
        """Return MJCF <body> elements (one mocap body per obstacle)."""
        lines: List[str] = []
        for spec in self._specs:
            p0 = spec.trajectory.position(0.0)
            size_str = " ".join(f"{s:.6f}" for s in spec.size)
            rgba_str = " ".join(f"{c:.4f}" for c in spec.rgba)
            lines.append(
                f'    <body name="{spec.obstacle_id}" mocap="true" '
                f'pos="{p0[0]:.6f} {p0[1]:.6f} {p0[2]:.6f}">'
            )
            lines.append(
                f'      <geom name="{spec.obstacle_id}_geom" type="{spec.geometry}" '
                f'size="{size_str}" rgba="{rgba_str}" contype="0" conaffinity="0"/>'
            )
            lines.append("    </body>")
        return "\n".join(lines)

    def bind(self, model: mujoco.MjModel) -> None:
        """Resolve body and mocap ids from a compiled model."""
        self._body_ids.clear()
        self._mocap_ids.clear()
        for oid in self.ids:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, oid)
            if body_id < 0:
                raise RuntimeError(f"body {oid!r} not found in model")
            mocap_id = int(model.body_mocapid[body_id])
            if mocap_id < 0:
                raise RuntimeError(f"body {oid!r} is not a mocap body")
            self._body_ids[oid] = int(body_id)
            self._mocap_ids[oid] = mocap_id

    # ------------------------------------------------------------------- update
    def update(self, data: mujoco.MjData, t: float) -> None:
        """Write obstacle positions for time t into data.mocap_pos."""
        for oid, mocap_id in self._mocap_ids.items():
            data.mocap_pos[mocap_id] = self._by_id[oid].trajectory.position(t)