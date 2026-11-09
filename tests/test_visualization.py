"""Unit tests for the visualization package."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from matplotlib.figure import Figure

from fusion.measurement_fusion import Measurement
from sensors.rgb_camera import CameraIntrinsics
from tracking import ObjectTracker, make_filter_factory
from tracking.track_manager import TrackStatus
from visualization.camera_view import (
    colorize_depth,
    draw_measurements_on_image,
    draw_text_lines,
    draw_tracks_on_image,
    project_world_points,
)
from visualization.common import (
    PALETTE,
    as_points,
    dim_color,
    figure_to_array,
    get_axes,
    id_color,
    new_figure,
    predicted_path,
    recent_trail,
    save_figure,
    status_color,
    status_name,
    to_unit_colors,
    to_unit_rgb,
    track_is_visible,
)
from visualization.pointcloud_view import (
    draw_arena_box,
    draw_box_3d,
    plot_detections_3d,
    plot_point_cloud,
    plot_scene_3d,
    plot_tracks_3d,
    prepare_cloud,
    set_scene_limits,
)
from visualization.tracking_dashboard import DashboardConfig, DashboardFrame, TrackingDashboard
from visualization.trajectory_view import (
    plot_estimate_vs_truth,
    plot_model_probabilities,
    plot_position_error,
    plot_trajectories_3d,
    plot_trajectories_xy,
    plot_velocity_vectors_xy,
    position_errors,
    rmse,
    trajectories_from_tracks,
    truth_trajectories,
)

INTR = CameraIntrinsics.from_fovy(64, 48, 60.0)  # cx = 31.5, cy = 23.5
BOUNDS = ((-2.5, -2.5, 0.0), (2.5, 2.5, 3.0))
CONFIRMED, TENTATIVE, LOST, DELETED = (
    TrackStatus.CONFIRMED,
    TrackStatus.TENTATIVE,
    TrackStatus.LOST,
    TrackStatus.DELETED,
)


# ------------------------------------------------------------------ helpers
class FakeTrack:
    """Minimal stand-in for tracking.Track."""

    def __init__(
        self, track_id, position, velocity=(0.0, 0.0, 0.0), status=CONFIRMED,
        model="CV3D", std=0.02, history=None,
    ):
        self.track_id = track_id
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.status = status
        self.motion_model = model
        self.position_std = std
        self._history = history

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity))

    def trajectory(self):
        if self._history is None:
            return np.zeros(0), np.zeros((0, 3))
        return self._history


def black_image():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def has_color(image, color):
    return bool(np.any(np.all(image == np.asarray(color, dtype=np.uint8), axis=-1)))


def pixel_of(point_cam):
    """Rounded pixel (u, v) of a camera-frame point, as the drawing code rounds it."""
    uv = INTR.project(np.asarray([point_cam], dtype=float))[0]
    return int(round(uv[0])), int(round(uv[1]))


def make_history(n=11, speed=0.5):
    times = np.linspace(0.0, 1.0, n)
    positions = np.column_stack([speed * times, np.zeros(n), np.full(n, 2.0)])
    return times, positions


# ------------------------------------------------------------------- common
class TestCommon:
    def test_id_color_is_stable_cycles_and_distinct(self):
        assert id_color(1) == PALETTE[0]
        assert id_color(1) == id_color(1)
        assert id_color(len(PALETTE) + 1) == id_color(1)
        assert len({id_color(i) for i in range(1, len(PALETTE) + 1)}) == len(PALETTE)

    def test_dim_and_unit_conversions(self):
        assert dim_color((200, 100, 50), 0.5) == (100, 50, 25)
        assert to_unit_rgb((255, 0, 51)) == pytest.approx((1.0, 0.0, 0.2))
        out = to_unit_colors(np.array([[255, 0, 128]], dtype=np.uint8))
        assert np.allclose(out, [[1.0, 0.0, 128 / 255]])
        assert np.allclose(to_unit_colors(np.array([[0.2, 1.5, -1.0]])), [[0.2, 1.0, 0.0]])

    def test_status_name(self):
        assert status_name(SimpleNamespace(status=CONFIRMED)) == "confirmed"
        assert status_name(SimpleNamespace(status="lost")) == "lost"
        assert status_name(SimpleNamespace()) == "unknown"

    def test_track_is_visible(self):
        def track(status):
            return SimpleNamespace(status=status)

        assert track_is_visible(track(CONFIRMED))
        assert not track_is_visible(track(TENTATIVE))
        assert track_is_visible(track(TENTATIVE), show_tentative=True)
        assert track_is_visible(track(LOST))
        assert not track_is_visible(track(LOST), show_lost=False)
        assert not track_is_visible(track(DELETED), show_tentative=True)
        assert not track_is_visible(SimpleNamespace())

    def test_status_color(self):
        base = id_color(3)
        assert status_color(FakeTrack(3, (0, 0, 1))) == base
        assert status_color(FakeTrack(3, (0, 0, 1), status=TENTATIVE)) == (150, 150, 150)
        assert status_color(FakeTrack(3, (0, 0, 1), status=LOST)) == dim_color(base, 0.55)

    def test_as_points(self):
        assert as_points([]).shape == (0, 3)
        assert as_points([1.0, 2.0, 3.0]).shape == (1, 3)
        assert as_points(np.zeros((5, 3))).shape == (5, 3)

    def test_predicted_path(self):
        path = predicted_path([1.0, 0.0, 2.0], [0.5, 0.0, -0.5], horizon=2.0, steps=4)
        assert path.shape == (5, 3)
        assert np.allclose(path[0], [1.0, 0.0, 2.0])
        assert np.allclose(path[-1], [2.0, 0.0, 1.0])
        with pytest.raises(ValueError):
            predicted_path([0, 0, 0], [1, 0, 0], horizon=-1.0)
        with pytest.raises(ValueError):
            predicted_path([0, 0, 0], [1, 0, 0], steps=0)

    def test_recent_trail(self):
        times, positions = make_history(5, speed=1.0)  # times 0, .25, .5, .75, 1.0
        track = FakeTrack(1, positions[-1], history=(times, positions))
        assert len(recent_trail(track, None)) == 5
        assert np.allclose(recent_trail(track, 0.5), positions[-3:])
        assert recent_trail(SimpleNamespace(), 1.0).shape == (0, 3)
        assert recent_trail(FakeTrack(1, (0, 0, 1)), 1.0).shape == (0, 3)

    def test_figure_to_array(self):
        fig = new_figure((3.0, 2.0), dpi=50)
        fig.patch.set_facecolor("red")
        arr = figure_to_array(fig)
        assert arr.shape == (100, 150, 3) and arr.dtype == np.uint8
        assert np.all(arr[0, 0] == [255, 0, 0])

    def test_get_axes_and_figures_are_standalone(self):
        fig = new_figure()
        assert isinstance(fig, Figure)
        ax = fig.add_subplot(111)
        assert get_axes(ax) is ax
        assert get_axes().figure is not fig
        assert get_axes(projection="3d").name == "3d"

    def test_save_figure_writes_a_png(self, tmp_path):
        fig = new_figure((2.0, 2.0), dpi=40)
        fig.add_subplot(111).plot([0, 1], [0, 1])
        out = save_figure(fig, tmp_path / "sub" / "plot.png")
        assert out.exists() and out.read_bytes()[:4] == b"\x89PNG"


# --------------------------------------------------------------- camera view
class TestProjection:
    def test_center_point_projects_to_the_principal_point(self):
        pixels, depth, inside = project_world_points([[0.0, 0.0, 2.0]], INTR, np.eye(4))
        assert np.allclose(pixels[0], [INTR.cx, INTR.cy])
        assert depth[0] == pytest.approx(2.0) and inside[0]

    def test_camera_pose_is_applied(self):
        t_w_c = np.eye(4)
        t_w_c[:3, 3] = [1.0, 0.0, 0.0]  # camera at world x = 1
        pixels, _, inside = project_world_points([[1.0, 0.0, 2.0]], INTR, t_w_c)
        assert np.allclose(pixels[0], [INTR.cx, INTR.cy]) and inside[0]

    def test_outside_and_behind_points_are_not_inside(self):
        _, _, inside = project_world_points(
            [[10.0, 0.0, 2.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]], INTR, np.eye(4)
        )
        assert not inside.any()

    def test_empty_input(self):
        pixels, depth, inside = project_world_points([], INTR, np.eye(4))
        assert pixels.shape == (0, 2) and depth.shape == (0,) and inside.shape == (0,)


class TestColorizeDepth:
    def test_colors_and_invalid_pixels(self):
        depth = np.array([[1.0, 7.0, np.nan], [0.05, 9.0, 4.0]], dtype=np.float32)
        rgb = colorize_depth(depth, 0.1, 8.0)
        assert rgb.shape == (2, 3, 3) and rgb.dtype == np.uint8
        for r, c in ((0, 2), (1, 0), (1, 1)):  # NaN, below min, above max
            assert np.all(rgb[r, c] == 0)
        for r, c in ((0, 0), (0, 1), (1, 2)):
            assert np.any(rgb[r, c] != 0)
        assert np.any(rgb[0, 0] != rgb[0, 1])  # near and far differ

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            colorize_depth(np.zeros((2, 2)), 5.0, 1.0)
        with pytest.raises(ValueError):
            colorize_depth(np.zeros(4))


class TestImageOverlays:
    def test_text_lines_draw_on_a_copy(self):
        img = black_image()
        out = draw_text_lines(img, ["hello", "world"])
        assert out.shape == img.shape and np.any(out > 0)
        assert not np.any(img)  # the input is untouched

    def test_measurement_marker_is_drawn_at_the_projection(self):
        img = black_image()
        out = draw_measurements_on_image(img, [[0.0, 0.0, 2.0]], INTR, np.eye(4), color=(10, 200, 30))
        u, v = pixel_of([0.0, 0.0, 2.0])
        assert tuple(out[v, u]) == (10, 200, 30)
        assert not np.any(img)

    def test_measurements_outside_the_image_are_skipped(self):
        out = draw_measurements_on_image(black_image(), [[10.0, 0.0, 2.0]], INTR, np.eye(4))
        assert not np.any(out)

    def test_labels_must_match(self):
        with pytest.raises(ValueError):
            draw_measurements_on_image(black_image(), [[0.0, 0.0, 2.0]], INTR, np.eye(4), labels=["a", "b"])

    def test_confirmed_track_is_drawn_in_its_id_color(self):
        track = FakeTrack(1, (0.0, 0.0, 2.0))
        img = black_image()
        out = draw_tracks_on_image(img, [track], INTR, np.eye(4))
        assert has_color(out, id_color(1))
        assert np.all(out[0, 0] == 0)  # far from the track
        assert not np.any(img)

    def test_velocity_arrow_points_along_the_motion(self):
        track = FakeTrack(1, (0.0, 0.0, 2.0), velocity=(1.0, 0.0, 0.0))
        out = draw_tracks_on_image(black_image(), [track], INTR, np.eye(4), velocity_horizon=1.0)
        u, v = pixel_of([0.0, 0.0, 2.0])
        assert tuple(out[v, u + 14]) == id_color(1)  # on the arrow shaft, right of the circle
        still = draw_tracks_on_image(black_image(), [FakeTrack(1, (0.0, 0.0, 2.0))], INTR, np.eye(4))
        assert not np.any(np.all(still[v, u + 8 : u + 20] == id_color(1), axis=-1))

    def test_track_visibility_rules(self):
        tentative = FakeTrack(2, (0.0, 0.0, 2.0), status=TENTATIVE)
        lost = FakeTrack(3, (0.0, 0.0, 2.0), status=LOST)
        deleted = FakeTrack(4, (0.0, 0.0, 2.0), status=DELETED)
        img = black_image()
        assert not np.any(draw_tracks_on_image(img, [tentative], INTR, np.eye(4)))
        assert has_color(draw_tracks_on_image(img, [tentative], INTR, np.eye(4), show_tentative=True), (150, 150, 150))
        assert has_color(draw_tracks_on_image(img, [lost], INTR, np.eye(4)), dim_color(id_color(3), 0.55))
        assert not np.any(draw_tracks_on_image(img, [lost], INTR, np.eye(4), show_lost=False))
        assert not np.any(draw_tracks_on_image(img, [deleted], INTR, np.eye(4), show_tentative=True))

    def test_tracks_outside_the_view_are_skipped(self):
        far_right = FakeTrack(1, (10.0, 0.0, 2.0))
        behind = FakeTrack(2, (0.0, 0.0, -1.0))
        assert not np.any(draw_tracks_on_image(black_image(), [far_right, behind], INTR, np.eye(4)))

    def test_invalid_images_raise(self):
        with pytest.raises(ValueError):
            draw_tracks_on_image(np.zeros((48, 64, 3), dtype=np.float32), [], INTR, np.eye(4))
        with pytest.raises(ValueError):
            draw_tracks_on_image(np.zeros((48, 64), dtype=np.uint8), [], INTR, np.eye(4))
        with pytest.raises(ValueError):
            draw_tracks_on_image(np.zeros((10, 10, 3), dtype=np.uint8), [], INTR, np.eye(4))


# --------------------------------------------------------------- point clouds
class TestPrepareCloud:
    def test_drops_non_finite_points_and_keeps_colors_aligned(self):
        pts = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, 1.0, 1.0]])
        cols = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        out_pts, out_cols = prepare_cloud(pts, cols)
        assert np.allclose(out_pts, [[0, 0, 0], [1, 1, 1]])
        assert np.allclose(out_cols, [[1, 0, 0], [0, 0, 1]])

    def test_subsampling_is_reproducible_and_aligned(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(0.0, 1.0, (100, 3))
        out_pts, out_cols = prepare_cloud(pts, pts.copy(), max_points=10)
        assert out_pts.shape == (10, 3)
        assert np.allclose(out_pts, out_cols)  # colour rows follow their points
        assert all(any(np.allclose(p, q) for q in pts) for p in out_pts)
        again, _ = prepare_cloud(pts, pts.copy(), max_points=10)
        assert np.allclose(out_pts, again)

    def test_small_and_empty_clouds_are_unchanged(self):
        pts = np.arange(9, dtype=float).reshape(3, 3)
        assert np.allclose(prepare_cloud(pts, None, max_points=10)[0], pts)
        assert prepare_cloud([], None)[0].shape == (0, 3)

    def test_color_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            prepare_cloud(np.zeros((3, 3)), np.zeros((2, 3)))


class TestPointCloudView:
    def test_plot_point_cloud(self):
        ax = get_axes(projection="3d")
        assert plot_point_cloud(ax, np.random.default_rng(0).uniform(0, 1, (50, 3))) is not None
        colors = np.full((5, 3), 128, dtype=np.uint8)
        assert plot_point_cloud(ax, np.zeros((5, 3)), colors=colors) is not None
        assert plot_point_cloud(ax, np.zeros((0, 3))) is None
        assert plot_point_cloud(ax, np.full((3, 3), np.nan)) is None

    def test_boxes_have_twelve_edges(self):
        ax = get_axes(projection="3d")
        draw_box_3d(ax, (0, 0, 0), (1, 2, 3))
        assert len(ax.lines) == 12
        draw_box_3d(ax, (0, 0, 0), (1, 0, 3))  # a flat box is allowed
        assert len(ax.lines) == 24
        with pytest.raises(ValueError):
            draw_box_3d(ax, (1, 0, 0), (0, 1, 1))

    def test_arena_box(self):
        ax = get_axes(projection="3d")
        draw_arena_box(ax, *BOUNDS)
        assert len(ax.lines) == 12

    def test_scene_limits(self):
        ax = get_axes(projection="3d")
        set_scene_limits(ax, *BOUNDS)
        assert ax.get_xlim() == pytest.approx((-2.5, 2.5))
        assert ax.get_ylim() == pytest.approx((-2.5, 2.5))
        assert ax.get_zlim() == pytest.approx((0.0, 3.0))
        with pytest.raises(ValueError):
            set_scene_limits(ax, (0, 0, 0), (1, 0, 1))

    def test_detections_and_bounding_boxes(self):
        features = SimpleNamespace(bbox_min=np.zeros(3), bbox_max=np.ones(3))
        detections = [
            SimpleNamespace(position=np.array([0.5, 0.5, 0.5]), features=features),
            SimpleNamespace(position=np.array([2.0, 2.0, 2.0]), features=features),
        ]
        ax = get_axes(projection="3d")
        assert plot_detections_3d(ax, detections) is not None
        assert len(ax.lines) == 0
        plot_detections_3d(ax, detections, show_bbox=True)
        assert len(ax.lines) == 24
        assert plot_detections_3d(ax, []) is None

    def test_tracks_are_labelled_and_filtered_by_status(self):
        tracks = [
            FakeTrack(1, (0.0, 0.0, 1.0), (0.2, 0.0, 0.0), history=make_history()),
            FakeTrack(2, (1.0, 0.0, 1.0), status=TENTATIVE),
            FakeTrack(3, (-1.0, 0.0, 1.0), status=LOST),
        ]
        ax = get_axes(projection="3d")
        plot_tracks_3d(ax, tracks)
        texts = [t.get_text() for t in ax.texts]
        assert "ID1" in texts and "ID3?" in texts and "ID2" not in texts
        assert len(ax.lines) == 2  # a trail and a velocity segment for track 1
        shown = get_axes(projection="3d")
        plot_tracks_3d(shown, tracks, show_tentative=True)
        assert "ID2" in [t.get_text() for t in shown.texts]

    def test_scene(self):
        rng = np.random.default_rng(1)
        clouds = {"lidar": rng.uniform(-1.0, 1.0, (100, 3)), "rgbd": rng.uniform(-1.0, 1.0, (50, 3))}
        track = FakeTrack(1, (0.0, 0.0, 1.0), (0.2, 0.0, 0.0), history=make_history())
        ax = plot_scene_3d(
            clouds=clouds,
            tracks=[track],
            detections=[SimpleNamespace(position=np.array([0.0, 0.0, 1.0]))],
            truth={"O001": np.array([0.0, 0.1, 1.0])},
            bounds=BOUNDS,
            sensor_positions={"lidar": (0.0, -3.0, 1.6)},
        )
        assert ax.get_xlim() == pytest.approx((-2.5, 2.5))
        assert len(ax.lines) >= 12  # the arena box
        assert "ID1" in [t.get_text() for t in ax.texts]
        assert ax.get_legend() is not None

    def test_empty_scene(self):
        ax = plot_scene_3d(bounds=BOUNDS)
        assert ax.get_legend() is None
        assert len(ax.lines) == 12


# --------------------------------------------------------------- trajectories
class TestTrajectoryData:
    def test_trajectories_from_tracks(self):
        long_track = FakeTrack(1, (0, 0, 1), history=make_history(5))
        short_track = FakeTrack(2, (0, 0, 1), history=(np.zeros(1), np.zeros((1, 3))))
        out = trajectories_from_tracks([long_track, short_track])
        assert list(out) == [1]
        times, positions = out[1]
        assert times.shape == (5,) and positions.shape == (5, 3)
        assert set(trajectories_from_tracks([long_track, short_track], min_samples=1)) == {1, 2}

    def test_truth_trajectories(self):
        states = np.zeros((4, 2, 6))
        states[:, 1, :3] = [1.0, 2.0, 3.0]
        out = truth_trajectories(np.arange(4.0), states, ["A", "B"])
        assert set(out) == {"A", "B"} and out["B"].shape == (4, 3)
        assert np.allclose(out["B"], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            truth_trajectories(np.arange(4.0), states, ["A"])
        with pytest.raises(ValueError):
            truth_trajectories(np.arange(3.0), states, ["A", "B"])
        with pytest.raises(ValueError):
            truth_trajectories(np.arange(4.0), np.zeros((4, 2)), ["A", "B"])

    def test_position_errors_and_rmse(self):
        errors = position_errors([[0, 0, 0], [1, 0, 0]], [[0, 0, 3], [1, 4, 0]])
        assert np.allclose(errors, [3.0, 4.0])
        assert rmse(errors) == pytest.approx(np.sqrt((9 + 16) / 2))
        assert rmse([np.nan, 3.0, 4.0]) == pytest.approx(np.sqrt(12.5))
        assert np.isnan(rmse([np.nan, np.nan]))
        with pytest.raises(ValueError):
            position_errors(np.zeros((3, 3)), np.zeros((4, 3)))


class TestTrajectoryPlots:
    TRAJECTORIES = {
        1: np.column_stack([np.linspace(-1, 1, 20), np.zeros(20), np.full(20, 1.0)]),
        2: np.column_stack([np.zeros(20), np.linspace(-1, 1, 20), np.full(20, 1.5)]),
    }
    TRUTH = {"O001": np.column_stack([np.linspace(-1, 1, 20), np.full(20, 0.05), np.full(20, 1.0)])}

    def test_xy_plot_draws_one_line_per_trajectory(self):
        ax = plot_trajectories_xy(self.TRAJECTORIES, self.TRUTH, bounds=BOUNDS)
        assert len(ax.lines) == 3
        assert ax.get_xlim() == pytest.approx((-2.5, 2.5))
        texts = [t.get_text() for t in ax.texts]
        assert "ID1" in texts and "ID2" in texts
        assert ax.get_legend() is not None

    def test_xy_plot_handles_empty_input(self):
        ax = plot_trajectories_xy({})
        assert len(ax.lines) == 0 and ax.get_legend() is None

    def test_3d_plot(self):
        ax = plot_trajectories_3d(self.TRAJECTORIES, self.TRUTH, bounds=BOUNDS)
        assert len(ax.lines) == 3
        assert ax.get_zlim() == pytest.approx((0.0, 3.0))

    def test_velocity_vectors(self):
        tracks = [
            FakeTrack(1, (0.0, 0.0, 1.0), (0.5, 0.2, 0.0)),
            FakeTrack(2, (1.0, 0.0, 1.0), (0.0, 0.0, 0.5)),  # no motion in the XY plane
            FakeTrack(3, (0.0, 1.0, 1.0), (0.3, 0.0, 0.0), status=TENTATIVE),
            FakeTrack(4, (0.0, -1.0, 1.0), (0.3, 0.0, 0.0), status=LOST),
        ]
        ax = plot_velocity_vectors_xy(tracks, horizon=2.0)
        assert len(ax.lines) == 1 and len(ax.patches) == 1
        path = ax.lines[0].get_xydata()
        assert path[-1] == pytest.approx([1.0, 0.4])
        shown = plot_velocity_vectors_xy(tracks, horizon=2.0, show_tentative=True, show_lost=True)
        assert len(shown.lines) == 3

    def test_estimate_vs_truth_figure(self):
        t = np.linspace(0.0, 2.0, 20)
        truth = np.column_stack([t, np.zeros(20), np.ones(20)])
        fig = plot_estimate_vs_truth(truth + 0.01, truth, times=t, title="demo")
        assert isinstance(fig, Figure) and len(fig.axes) == 4
        assert fig.axes[3].lines[0].get_ydata() == pytest.approx(0.01 * np.sqrt(3.0))
        assert len(figure_to_array(fig).shape) == 3

    def test_estimate_vs_truth_validates_input(self):
        with pytest.raises(ValueError):
            plot_estimate_vs_truth(np.zeros((5, 3)), np.zeros((4, 3)))
        with pytest.raises(ValueError):
            plot_estimate_vs_truth(np.zeros((5, 3)), np.zeros((5, 3)), times=np.arange(3.0))

    def test_position_error_plot(self):
        t = np.arange(10.0)
        errors = {"EKF": np.full(10, 0.02), "UKF": np.full(10, np.nan), "IMM": np.linspace(0.0, 0.04, 10)}
        ax = plot_position_error(t, errors)
        assert len(ax.lines) == 2  # the all-NaN series is skipped
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert any("EKF" in s and "RMSE 0.020" in s for s in labels)
        with pytest.raises(ValueError):
            plot_position_error(t, {"bad": np.zeros(5)})

    def test_model_probabilities_plot(self):
        t = np.linspace(0.0, 1.0, 10)
        p3d = np.linspace(0.1, 0.9, 10)
        ax = plot_model_probabilities(t, {"CV2D": 1.0 - p3d, "CV3D": p3d})
        assert len(ax.collections) == 2
        assert ax.get_ylim() == pytest.approx((0.0, 1.0))
        with pytest.raises(ValueError):
            plot_model_probabilities(t, {})
        with pytest.raises(ValueError):
            plot_model_probabilities(t, {"CV2D": np.zeros(3)})


# ------------------------------------------------------------------ dashboard
CFG = DashboardConfig(figsize=(6.0, 4.0), dpi=50)  # renders 300 x 200 pixels


def make_dashboard(**kwargs):
    return TrackingDashboard(
        INTR, np.eye(4), sensor_positions={"lidar": (0.0, -0.3, 0.0)}, config=CFG, **kwargs
    )


def make_frame(with_image=True, time=1.0):
    rng = np.random.default_rng(0)
    tracks = [
        FakeTrack(1, (0.5, 0.0, 2.0), (0.5, 0.0, 0.0), history=make_history()),
        FakeTrack(2, (0.5, 0.3, 1.5), status=TENTATIVE),
        FakeTrack(3, (-0.5, 0.0, 2.0), status=LOST),
    ]
    return DashboardFrame(
        time=time,
        image=black_image() if with_image else None,
        tracks=tracks,
        measurements=[SimpleNamespace(position=np.array([0.5, 0.01, 2.0]))],
        clouds={"lidar": rng.uniform([-1.0, -1.0, 0.5], [1.0, 1.0, 2.5], (200, 3))},
        truth={"O001": np.array([0.5, 0.0, 2.0])},
    )


class TestDashboard:
    def test_info_lines(self):
        tracks = [
            FakeTrack(2, (0, 0, 1), status=CONFIRMED, model="CV2D"),
            FakeTrack(1, (0, 0, 1), (0.3, 0.4, 0.0), status=CONFIRMED),
            FakeTrack(3, (0, 0, 1), status=TENTATIVE),
            FakeTrack(4, (0, 0, 1), status=LOST),
            FakeTrack(5, (0, 0, 1), status=DELETED),
        ]
        lines = make_dashboard().info_lines(DashboardFrame(1.25, tracks=tracks, measurements=[1, 2]))
        assert lines[0] == "time: 1.25 s"
        assert "2 confirmed, 1 tentative, 1 lost" in lines[1]
        assert "measurements: 2" in lines[2]
        rows = [line for line in lines[5:] if line.strip()]
        assert [int(r.split()[0]) for r in rows] == [1, 2, 3, 4]  # sorted by id, deleted track hidden
        assert "0.50" in rows[0] and "confirmed" in rows[0]
        assert "CV2D" in rows[1]

    def test_info_lines_truncate_long_lists(self):
        tracks = [FakeTrack(i, (0, 0, 1)) for i in range(1, 16)]
        lines = make_dashboard().info_lines(DashboardFrame(0.0, tracks=tracks), max_rows=12)
        assert lines[-1] == "... +3 more"

    def test_info_lines_handle_missing_attributes(self):
        bare = SimpleNamespace(track_id=7, status="confirmed", position=np.zeros(3))
        lines = make_dashboard().info_lines(DashboardFrame(0.0, tracks=[bare]))
        assert lines[-1].split()[0] == "7" and lines[-1].count("-") >= 1

    def test_render_shape_and_content(self):
        arr = make_dashboard().render(make_frame())
        assert arr.shape == (200, 300, 3) and arr.dtype == np.uint8
        assert len(np.unique(arr.reshape(-1, 3), axis=0)) > 10  # not a blank image

    def test_render_without_image_or_extras(self):
        dash = make_dashboard()
        assert dash.render(make_frame(with_image=False)).shape == (200, 300, 3)
        assert dash.render(DashboardFrame(time=0.0)).shape == (200, 300, 3)

    def test_figure_has_four_panels(self):
        fig = make_dashboard().figure(make_frame())
        assert len(fig.axes) == 4
        assert any(ax.name == "3d" for ax in fig.axes)

    def test_save_frame_and_frames(self, tmp_path):
        dash = make_dashboard()
        out = dash.save_frame(make_frame(), tmp_path / "one.png")
        assert out.exists() and out.read_bytes()[:4] == b"\x89PNG"
        paths = dash.save_frames([make_frame(time=0.0), make_frame(time=0.1)], tmp_path / "seq")
        assert [p.name for p in paths] == ["frame_0000.png", "frame_0001.png"]
        assert all(p.exists() for p in paths)

    def test_write_video(self, tmp_path):
        dash = make_dashboard()
        path = tmp_path / "out.avi"
        try:
            dash.write_video(path, (make_frame(time=0.1 * k) for k in range(3)), fps=5.0, fourcc="MJPG")
        except RuntimeError as exc:  # no video codec in this OpenCV build
            pytest.skip(str(exc))
        assert path.exists() and path.stat().st_size > 0
        capture = cv2.VideoCapture(str(path))
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        capture.release()

    def test_write_video_rejects_bad_arguments(self, tmp_path):
        dash = make_dashboard()
        with pytest.raises(ValueError):
            dash.write_video(tmp_path / "a.avi", [], fourcc="MJPG")
        with pytest.raises(ValueError):
            dash.write_video(tmp_path / "a.avi", [make_frame()], fps=0.0, fourcc="MJPG")
        with pytest.raises(ValueError):
            dash.write_video(tmp_path / "a.avi", [make_frame()], fourcc="MJ")

    def test_renders_real_tracks_from_the_tracker(self):
        rng = np.random.default_rng(0)
        tracker = ObjectTracker(make_filter_factory("ekf", use_imm=False, accel_std=0.2))
        for k in range(10):
            t = 0.1 * k
            position = np.array([0.2 * t, 0.0, 2.0]) + 0.01 * rng.standard_normal(3)
            measurement = Measurement(("lidar",), t, position, np.eye(3) * 0.03**2)
            tracker.update([measurement], t)
        assert len(tracker.confirmed_tracks) == 1

        frame = DashboardFrame(
            time=tracker.time, image=black_image(), tracks=tracker.tracks, measurements=[measurement]
        )
        dash = make_dashboard()
        assert dash.render(frame).shape == (200, 300, 3)
        lines = dash.info_lines(frame)
        assert "1 confirmed" in lines[1]

    def test_invalid_configuration_raises(self):
        for kwargs in (
            {"figsize": (0.0, 4.0)},
            {"dpi": 0},
            {"trail_seconds": -1.0},
            {"velocity_horizon": -1.0},
            {"max_cloud_points": 0},
        ):
            with pytest.raises(ValueError):
                DashboardConfig(**kwargs)
        with pytest.raises(ValueError):
            TrackingDashboard(INTR, np.eye(3))
        with pytest.raises(ValueError):
            TrackingDashboard(INTR, np.eye(4), bounds_min=(0, 0, 0), bounds_max=(1, 0, 1))