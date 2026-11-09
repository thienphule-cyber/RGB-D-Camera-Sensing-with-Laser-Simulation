"""MuJoCo calibration scene: the standard sensor assembly plus a movable checkerboard.

CalibrationScene is an Environment without obstacles. It keeps the exact sensor
geometry of the main simulation, so RGBCamera, Lidar and LaserScanner work on it
unchanged. The checkerboard is a mocap body whose pose is set in the camera frame.

Board frame (OpenCV convention, origin at the board centre): x right, y down,
z pointing away from the camera. The pattern lies on the z = 0 face, the board
body sits behind it (z > 0) and the dark squares protrude 1.5 mm toward the camera.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sensors.rgb_camera import CameraIntrinsics, ground_truth_camera_pose_opencv
from simulation.environment import Environment, EnvironmentConfig

BOARD_BODY_NAME = "calib_board"
BOARD_GEOM_PREFIX = "calib_board"
BOARD_THICKNESS = 0.01  # [m]
SQUARE_HALF_THICKNESS = 0.001  # [m]
SQUARE_CENTER_Z = -0.0005  # [m], dark squares span z in [-1.5 mm, +0.5 mm]

Pose = Tuple[np.ndarray, np.ndarray]


def checkerboard_object_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
    """Inner-corner coordinates (N, 3) in the board frame (z = 0), centred on the board.

    pattern_size = (corners per row, corners per column) as in OpenCV. Points are
    ordered row by row with x varying fastest, like cv2.findChessboardCorners.
    """
    nx, ny = int(pattern_size[0]), int(pattern_size[1])
    if nx < 2 or ny < 2:
        raise ValueError("pattern_size must be at least (2, 2)")
    if square_size <= 0.0:
        raise ValueError("square_size must be positive")
    xs = (np.arange(nx) - (nx - 1) / 2.0) * square_size
    ys = (np.arange(ny) - (ny - 1) / 2.0) * square_size
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])


def sample_board_poses(
    intrinsics: CameraIntrinsics,
    half_extent: Tuple[float, float],
    n_views: int,
    rng: np.random.Generator,
    depth_range: Tuple[float, float] = (1.3, 2.0),
    x_range: Tuple[float, float] = (-0.25, 0.25),
    y_range: Tuple[float, float] = (-0.15, 0.15),
    min_tilt_deg: float = 10.0,
    max_tilt_deg: float = 30.0,
    max_inplane_deg: float = 30.0,
    margin_px: float = 10.0,
    max_attempts: int = 20000,
) -> List[Pose]:
    """Random board poses T_cam_board that keep the whole board inside the image.

    The tilt (angle between the board normal and the optical axis) is limited to
    [min_tilt_deg, max_tilt_deg] with a random tilt direction, which keeps the
    calibration well conditioned. Returns a list of (R_cb, t_cb).
    """
    if n_views <= 0:
        raise ValueError("n_views must be positive")
    if not 0.0 <= min_tilt_deg <= max_tilt_deg < 90.0:
        raise ValueError("need 0 <= min_tilt_deg <= max_tilt_deg < 90")

    hx, hy = half_extent
    outline = np.array([[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]])
    poses: List[Pose] = []
    for _ in range(max_attempts):
        if len(poses) >= n_views:
            break
        tilt = np.radians(rng.uniform(min_tilt_deg, max_tilt_deg))
        direction = rng.uniform(0.0, 2.0 * np.pi)
        axis = np.array([np.cos(direction), np.sin(direction), 0.0])
        inplane = np.radians(rng.uniform(-max_inplane_deg, max_inplane_deg))
        rotation = (Rotation.from_rotvec(tilt * axis) * Rotation.from_euler("z", inplane)).as_matrix()
        translation = np.array(
            [rng.uniform(*x_range), rng.uniform(*y_range), rng.uniform(*depth_range)]
        )

        corners = outline @ rotation.T + translation
        if np.any(corners[:, 2] < 0.2):
            continue
        uv = intrinsics.project(corners)
        inside = (
            np.all(uv[:, 0] >= margin_px)
            and np.all(uv[:, 0] <= intrinsics.width - 1 - margin_px)
            and np.all(uv[:, 1] >= margin_px)
            and np.all(uv[:, 1] <= intrinsics.height - 1 - margin_px)
        )
        if inside:
            poses.append((rotation, translation))
    if len(poses) < n_views:
        raise RuntimeError(
            f"could only sample {len(poses)} of {n_views} board poses; "
            "relax depth_range / tilt limits or use a smaller board"
        )
    return poses


class CalibrationScene(Environment):
    """Environment with no obstacles and a movable checkerboard target."""

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        pattern_size: Tuple[int, int] = (8, 5),
        square_size: float = 0.10,
    ) -> None:
        nx, ny = int(pattern_size[0]), int(pattern_size[1])
        if nx < 2 or ny < 2:
            raise ValueError("pattern_size must be at least (2, 2)")
        if square_size <= 0.0:
            raise ValueError("square_size must be positive")
        # These must exist before Environment.__init__ builds the model.
        self.pattern_size = (nx, ny)
        self.square_size = float(square_size)
        self._board_mocap_id = -1
        self._board_pose: Pose = (np.eye(3), np.array([0.0, 0.0, 1.5]))

        super().__init__(config=config, obstacles=[])

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BOARD_BODY_NAME)
        self._board_body_id = int(body_id)
        self._board_mocap_id = int(self.model.body_mocapid[body_id])
        self.board_geom_ids = np.array(
            [
                g
                for g in range(self.model.ngeom)
                if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
                    BOARD_GEOM_PREFIX
                )
            ],
            dtype=int,
        )
        self.set_board_pose(*self._board_pose)

    # ----------------------------------------------------------------- geometry
    @property
    def half_extent(self) -> Tuple[float, float]:
        """Half size (x, y) of the whole board including a one-square white margin."""
        nx, ny = self.pattern_size
        return ((nx + 1) * self.square_size / 2.0 + self.square_size,
                (ny + 1) * self.square_size / 2.0 + self.square_size)

    @property
    def board_pose(self) -> Pose:
        """Last pose set by set_board_pose: (R_cb, t_cb) in the OpenCV camera frame."""
        return self._board_pose[0].copy(), self._board_pose[1].copy()

    # -------------------------------------------------------------------- model
    def _build_xml(self) -> str:
        xml = super()._build_xml()
        if xml.count("<worldbody>") != 1 or xml.count("</worldbody>") != 1:
            raise RuntimeError("unexpected base MJCF layout")
        asset = (
            "<asset>\n"
            '    <material name="calib_matte" specular="0" shininess="0" reflectance="0"/>\n'
            "  </asset>\n  "
        )
        xml = xml.replace("<worldbody>", asset + "<worldbody>", 1)
        xml = xml.replace("</worldbody>", self._board_xml() + "\n  </worldbody>", 1)
        return xml

    def _board_xml(self) -> str:
        nx, ny = self.pattern_size
        cols, rows = nx + 1, ny + 1  # number of squares
        s = self.square_size
        hx, hy = self.half_extent
        half_t = BOARD_THICKNESS / 2.0
        lines = [
            f'    <body name="{BOARD_BODY_NAME}" mocap="true" pos="0 0 0">',
            f'      <geom name="{BOARD_GEOM_PREFIX}_base" type="box" material="calib_matte" '
            f'pos="0 0 {half_t:.6f}" size="{hx:.6f} {hy:.6f} {half_t:.6f}" '
            f'rgba="0.97 0.97 0.97 1" contype="0" conaffinity="0"/>',
        ]
        for a in range(cols):
            for b in range(rows):
                if (a + b) % 2 != 0:
                    continue
                x = (a + 0.5 - cols / 2.0) * s
                y = (b + 0.5 - rows / 2.0) * s
                lines.append(
                    f'      <geom name="{BOARD_GEOM_PREFIX}_sq_{a}_{b}" type="box" '
                    f'material="calib_matte" pos="{x:.6f} {y:.6f} {SQUARE_CENTER_Z:.6f}" '
                    f'size="{s / 2.0:.6f} {s / 2.0:.6f} {SQUARE_HALF_THICKNESS:.6f}" '
                    f'rgba="0.03 0.03 0.03 1" contype="0" conaffinity="0"/>'
                )
        lines.append("    </body>")
        return "\n".join(lines)

    # ------------------------------------------------------------------- poses
    def set_board_pose(self, rotation, translation) -> None:
        """Place the board at T_cam_board = (rotation, translation) in the OpenCV camera frame.

        The ground-truth camera pose is used only to place the target in the world.
        """
        rot = np.asarray(rotation, dtype=float).reshape(3, 3)
        trans = np.asarray(translation, dtype=float).reshape(3)
        cam_pos, rot_wc = ground_truth_camera_pose_opencv(self)
        rot_wb = rot_wc @ rot
        pos_wb = cam_pos + rot_wc @ trans
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot_wb.reshape(9)))
        self.data.mocap_pos[self._board_mocap_id] = pos_wb
        self.data.mocap_quat[self._board_mocap_id] = quat
        mujoco.mj_forward(self.model, self.data)
        self._board_pose = (rot.copy(), trans.copy())

    def reset(self) -> None:
        """Reset the simulation and restore the last board pose."""
        super().reset()
        if self._board_mocap_id >= 0:
            self.set_board_pose(*self._board_pose)