"""Per-object features: geometry, shape descriptors, sphere fit and colour."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------- sphere
@dataclass(frozen=True, eq=False)
class SphereFit:
    """Result of a least-squares sphere fit."""

    center: np.ndarray  # (3,)
    radius: float  # [m]
    rms: float  # RMS of |p - c| - r over the inliers [m]
    extent: float  # largest bounding-box side of the fitted points [m]
    inlier_ratio: float  # fraction of the points used in the final fit


def _algebraic_sphere(points: np.ndarray):
    """Algebraic sphere fit: 2 p.c + (r^2 - |c|^2) = |p|^2. Returns (center, radius) or None."""
    if len(points) < 4:
        return None
    centroid = points.mean(axis=0)
    q = points - centroid
    a = np.column_stack([2.0 * q, np.ones(len(q))])
    b = np.einsum("ij,ij->i", q, q)
    solution, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank < 4:
        return None
    c = solution[:3]
    r2 = solution[3] + float(c @ c)
    if not np.isfinite(r2) or r2 <= 0.0:
        return None
    return centroid + c, float(np.sqrt(r2))


def fit_sphere(points, robust_iterations: int = 2, inlier_sigma: float = 3.0) -> Optional[SphereFit]:
    """Least-squares sphere through (N, 3) points; None if the fit is degenerate.

    With robust_iterations > 0 the fit is repeated on the points whose residual is
    within inlier_sigma * (robust sigma from the median residual), with a floor of
    1 mm, which removes flying pixels and other outliers.
    Degenerate cases (fewer than 4 points, coplanar or collinear points) give None.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 4:
        return None

    mask = np.ones(len(pts), dtype=bool)
    center = None
    radius = 0.0
    for iteration in range(robust_iterations + 1):
        solved = _algebraic_sphere(pts[mask])
        if solved is None:
            break
        center, radius = solved
        if iteration == robust_iterations:
            break
        residual = np.abs(np.linalg.norm(pts - center, axis=1) - radius)
        sigma = 1.4826 * float(np.median(residual[mask]))
        new_mask = residual <= max(inlier_sigma * sigma, 1e-3)
        if int(new_mask.sum()) < 4 or np.array_equal(new_mask, mask):
            break
        mask = new_mask

    if center is None:
        return None
    residual = np.linalg.norm(pts[mask] - center, axis=1) - radius
    return SphereFit(
        center=np.asarray(center, dtype=float),
        radius=float(radius),
        rms=float(np.sqrt(np.mean(residual**2))),
        extent=float(np.max(pts.max(axis=0) - pts.min(axis=0))),
        inlier_ratio=float(mask.sum()) / len(pts),
    )


def is_sphere_like(
    fit: SphereFit,
    min_radius: float = 0.03,
    max_radius: float = 0.5,
    max_rel_rms: float = 0.1,
    min_extent_ratio: float = 1.0,
    min_inlier_ratio: float = 0.6,
) -> bool:
    """Plausibility test of a sphere fit (a shape hint, not a classifier).

    Flat patches fit enormous spheres (rejected by max_radius), and a tiny visible
    patch cannot constrain the radius (rejected by min_extent_ratio).
    """
    return bool(
        min_radius <= fit.radius <= max_radius
        and fit.rms <= max_rel_rms * fit.radius
        and fit.extent >= min_extent_ratio * fit.radius
        and fit.inlier_ratio >= min_inlier_ratio
    )


# ------------------------------------------------------------------ geometry
@dataclass(frozen=True, eq=False)
class ObjectFeatures:
    """Geometric description of the visible surface of one object (world frame).

    Only the visible surface is observed, so the centroid is biased toward the
    sensor and the dimensions along the viewing direction are underestimated.
    """

    centroid: np.ndarray  # (3,) mean of the points
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)
    num_points: int
    covariance: np.ndarray  # (3, 3) covariance of the points
    linearity: float  # (l1 - l2) / l1, eigenvalues l1 >= l2 >= l3
    planarity: float  # (l2 - l3) / l1
    sphericity: float  # l3 / l1
    yaw: float  # direction of the principal axis in the XY plane [rad], in [-pi/2, pi/2)
    sphere: Optional[SphereFit]  # plausible sphere fit, else None

    @property
    def dimensions(self) -> np.ndarray:
        """Axis-aligned bounding-box size (3,)."""
        return self.bbox_max - self.bbox_min

    @property
    def bbox_center(self) -> np.ndarray:
        return 0.5 * (self.bbox_min + self.bbox_max)

    @property
    def shape_hint(self) -> str:
        """"sphere" when a plausible sphere fit exists, otherwise "other"."""
        return "sphere" if self.sphere is not None else "other"


def compute_object_features(points, fit_sphere_model: bool = True) -> ObjectFeatures:
    """Compute the geometric features of one cluster (at least 3 points)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = len(pts)
    if n < 3:
        raise ValueError("need at least 3 points")

    centroid = pts.mean(axis=0)
    q = pts - centroid
    covariance = q.T @ q / n

    eigenvalues = np.clip(np.linalg.eigvalsh(covariance)[::-1], 0.0, None)  # descending
    l1, l2, l3 = (float(x) for x in eigenvalues)
    if l1 > 1e-18:
        linearity = (l1 - l2) / l1
        planarity = (l2 - l3) / l1
        sphericity = l3 / l1
    else:
        linearity = planarity = sphericity = 0.0

    _, vectors = np.linalg.eigh(covariance[:2, :2])
    major = vectors[:, 1]
    yaw = float((np.arctan2(major[1], major[0]) + np.pi / 2.0) % np.pi - np.pi / 2.0)

    sphere = None
    if fit_sphere_model:
        fit = fit_sphere(pts)
        thickness = float(np.sqrt(l3))
        if fit is not None and is_sphere_like(fit) and thickness >= 0.05 * fit.radius:
            sphere = fit

    return ObjectFeatures(
        centroid=centroid,
        bbox_min=pts.min(axis=0),
        bbox_max=pts.max(axis=0),
        num_points=n,
        covariance=covariance,
        linearity=float(linearity),
        planarity=float(planarity),
        sphericity=float(sphericity),
        yaw=yaw,
        sphere=sphere,
    )


# -------------------------------------------------------------------- colour
@dataclass(frozen=True, eq=False)
class ColorDescriptor:
    """Brightness-tolerant appearance descriptor of a cluster."""

    chromaticity: np.ndarray  # (3,) mean RGB normalised to sum 1
    hue_deg: float  # saturation-weighted circular mean hue [0, 360); NaN if unsaturated
    saturation: float  # mean saturation [0, 1]
    value: float  # mean brightness [0, 1]


def rgb_to_hsv(colors) -> np.ndarray:
    """Convert (N, 3) RGB (uint8 in 0-255, or float in 0-1) to HSV.

    Returns (N, 3): hue in degrees [0, 360), saturation and value in [0, 1].
    """
    c = np.asarray(colors)
    if np.issubdtype(c.dtype, np.integer):
        c = c.astype(np.float64) / 255.0
    else:
        c = c.astype(np.float64)
    c = c.reshape(-1, 3)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    mx = c.max(axis=1)
    delta = mx - c.min(axis=1)

    hue = np.zeros(len(c))
    nonzero = delta > 1e-12
    r_max = nonzero & (mx == r)
    g_max = nonzero & (mx == g) & ~r_max
    b_max = nonzero & ~r_max & ~g_max
    hue[r_max] = ((g[r_max] - b[r_max]) / delta[r_max]) % 6.0
    hue[g_max] = (b[g_max] - r[g_max]) / delta[g_max] + 2.0
    hue[b_max] = (r[b_max] - g[b_max]) / delta[b_max] + 4.0
    hue *= 60.0

    saturation = np.where(mx > 1e-12, delta / np.maximum(mx, 1e-12), 0.0)
    return np.column_stack([hue, saturation, mx])


def hue_difference_deg(a: float, b: float) -> float:
    """Smallest angular difference [deg] between two hues (0-180); NaN stays NaN."""
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


def color_descriptor(colors, min_saturation: float = 0.2) -> ColorDescriptor:
    """Descriptor of the (N, 3) colours of a cluster.

    The hue only uses pixels with saturation >= min_saturation, so grey or
    specular-white pixels do not disturb it.
    """
    c = np.asarray(colors)
    if c.size == 0:
        raise ValueError("colors must not be empty")
    rgb = c.astype(np.float64) / 255.0 if np.issubdtype(c.dtype, np.integer) else c.astype(np.float64)
    rgb = rgb.reshape(-1, 3)

    mean_rgb = rgb.mean(axis=0)
    total = float(mean_rgb.sum())
    chromaticity = mean_rgb / total if total > 1e-12 else np.full(3, 1.0 / 3.0)

    hsv = rgb_to_hsv(rgb)
    saturated = hsv[:, 1] >= min_saturation
    if saturated.any():
        angle = np.radians(hsv[saturated, 0])
        weight = hsv[saturated, 1]
        hue = float(np.degrees(np.arctan2((weight * np.sin(angle)).sum(), (weight * np.cos(angle)).sum())) % 360.0)
    else:
        hue = float("nan")
    return ColorDescriptor(
        chromaticity=chromaticity,
        hue_deg=hue,
        saturation=float(hsv[:, 1].mean()),
        value=float(hsv[:, 2].mean()),
    )