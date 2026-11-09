"""Optimal (Hungarian) assignment with gating.

The solver maximizes the number of matches first and minimizes the total cost
second: a forbidden pair (infinite cost, or a cost above max_cost) is never
matched, and no allowed match is given up to lower the total cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Assignment:
    """Result of an assignment problem with n rows (tracks) and m columns (measurements)."""

    matches: List[Tuple[int, int]]  # (row, col), sorted by row
    unmatched_rows: List[int]
    unmatched_cols: List[int]
    total_cost: float  # sum of the costs of the matches


def solve_assignment(cost, max_cost: float = np.inf) -> Assignment:
    """Minimum-cost assignment of rows to columns with forbidden pairs.

    Args:
        cost: (n, m) cost matrix. Entries that are +inf, or larger than max_cost,
            are forbidden pairs. NaN is not allowed.
        max_cost: gate on the cost of a single match.

    Rows and columns do not need to match in number. Every row and column is used
    at most once.
    """
    arr = np.asarray(cost, dtype=float)
    if arr.ndim != 2:
        raise ValueError("cost must be a 2D array")
    if np.isnan(arr).any():
        raise ValueError("cost must not contain NaN")

    n_rows, n_cols = arr.shape
    all_rows, all_cols = list(range(n_rows)), list(range(n_cols))
    if n_rows == 0 or n_cols == 0:
        return Assignment([], all_rows, all_cols, 0.0)

    allowed = np.isfinite(arr) & (arr <= max_cost)
    if not allowed.any():
        return Assignment([], all_rows, all_cols, 0.0)

    # Shift the allowed costs to be >= 0 and price every forbidden pair above the sum of
    # all allowed costs. One more allowed match then always beats any cost saving, so
    # the cardinality is maximized first. The shift does not change which of the
    # equally sized solutions is cheapest.
    shifted = np.where(allowed, arr - arr[allowed].min(), 0.0)
    big = float(shifted[allowed].sum()) + 1.0
    work = np.where(allowed, shifted, big)
    rows, cols = linear_sum_assignment(work)

    matches = sorted((int(r), int(c)) for r, c in zip(rows, cols) if allowed[r, c])
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    return Assignment(
        matches=matches,
        unmatched_rows=[r for r in all_rows if r not in matched_rows],
        unmatched_cols=[c for c in all_cols if c not in matched_cols],
        total_cost=float(sum(arr[r, c] for r, c in matches)),
    )