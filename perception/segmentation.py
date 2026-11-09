"""Euclidean clustering of point clouds (geometric segmentation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass
class ClusterConfig:
    """Parameters of the Euclidean clustering.

    radius: two points closer than this [m] belong to the same cluster.
    min_points: clusters with fewer points are treated as noise. At least 3,
        because object features need three points.
    """

    radius: float = 0.08
    min_points: int = 10

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.min_points < 3:
            raise ValueError("min_points must be >= 3")


def cluster_labels(points, radius: float, min_points: int = 1) -> np.ndarray:
    """Euclidean clustering: connected components of the "closer than radius" graph.

    Returns an int array (N,). Labels are 0..K-1 ordered by decreasing cluster size
    (ties keep the order of first appearance); points of clusters smaller than
    min_points get the label -1 (noise).
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if min_points < 1:
        raise ValueError("min_points must be >= 1")
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=int)
    if not np.isfinite(pts).all():
        raise ValueError("points must be finite")

    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_components, component = connected_components(graph, directed=False)

    sizes = np.bincount(component, minlength=n_components)
    kept = np.flatnonzero(sizes >= min_points)
    order = kept[np.argsort(-sizes[kept], kind="stable")]
    remap = np.full(n_components, -1, dtype=int)
    remap[order] = np.arange(len(order))
    return remap[component]


def split_clusters(labels) -> List[np.ndarray]:
    """Index arrays of the points of each cluster (label 0, 1, ...); noise (-1) is skipped."""
    arr = np.asarray(labels)
    if arr.size == 0:
        return []
    count = int(arr.max()) + 1
    return [np.flatnonzero(arr == k) for k in range(max(count, 0))]


def cluster_points(points, config: ClusterConfig) -> List[np.ndarray]:
    """Cluster a cloud and return the index arrays of the clusters (largest first)."""
    labels = cluster_labels(points, config.radius, config.min_points)
    return split_clusters(labels)