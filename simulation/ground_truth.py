"""Ground-truth logging, used ONLY for evaluation (never as filter input)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from .environment import Environment


@dataclass(frozen=True)
class GroundTruthRecord:
    """Ground-truth state of one obstacle at one time instant."""

    time: float
    obstacle_id: str
    geometry: str
    motion_dim: str
    position: np.ndarray  # (3,) from MuJoCo simulation state (data.xpos)
    velocity: np.ndarray  # (3,) analytic, from the trajectory generator


class GroundTruth:
    """Reads and records obstacle ground truth from an Environment."""

    def __init__(self, env: Environment) -> None:
        self._env = env
        self._times: List[float] = []
        self._snapshots: List[List[GroundTruthRecord]] = []

    def snapshot(self) -> List[GroundTruthRecord]:
        """Ground truth of all obstacles at the current simulation time."""
        env = self._env
        t = env.time
        records: List[GroundTruthRecord] = []
        for oid in env.obstacles.ids:
            spec = env.obstacles.get_spec(oid)
            position = env.data.xpos[env.obstacles.body_id(oid)].copy()
            velocity = spec.trajectory.velocity(t)
            records.append(
                GroundTruthRecord(
                    time=t,
                    obstacle_id=oid,
                    geometry=spec.geometry,
                    motion_dim=spec.motion_dim,
                    position=position,
                    velocity=np.asarray(velocity, dtype=float).copy(),
                )
            )
        return records

    def record(self) -> None:
        """Append the current snapshot to the log."""
        snap = self.snapshot()
        self._times.append(snap[0].time if snap else self._env.time)
        self._snapshots.append(snap)

    def clear(self) -> None:
        self._times.clear()
        self._snapshots.clear()

    def __len__(self) -> int:
        return len(self._snapshots)

    def times(self) -> np.ndarray:
        """Logged times, shape (T,)."""
        return np.asarray(self._times, dtype=float)

    def to_array(self) -> np.ndarray:
        """Logged states, shape (T, N, 6): [px, py, pz, vx, vy, vz] per obstacle.

        The obstacle order follows env.obstacles.ids.
        """
        n = len(self._env.obstacles.ids)
        if not self._snapshots:
            return np.zeros((0, n, 6))
        return np.array(
            [
                [np.concatenate([r.position, r.velocity]) for r in snap]
                for snap in self._snapshots
            ]
        )

    def save_csv(self, path: Union[str, Path]) -> Path:
        """Write the log to CSV (one row per obstacle per logged time)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "time", "obstacle_id", "geometry", "motion_dim",
            "px", "py", "pz", "vx", "vy", "vz",
        ]
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for snap in self._snapshots:
                for r in snap:
                    writer.writerow(
                        [
                            f"{r.time:.6f}", r.obstacle_id, r.geometry, r.motion_dim,
                            *(f"{x:.9f}" for x in r.position),
                            *(f"{x:.9f}" for x in r.velocity),
                        ]
                    )
        return path