"""
Frontier-based exploration: extract frontiers from the occupancy grid,
score them by information gain and distance, select the best one.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ros2_bridge_pkg.local_map import OccupancyGrid, SemanticObjectMap, FREE_THRESH, OCC_THRESH


try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _binary_dilation_3x3(mask: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        k = np.ones((3, 3), dtype=np.uint8)
        return _cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)
    out = mask.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            out |= np.roll(np.roll(mask, dx, axis=0), dy, axis=1)
    return out


def _label_connected(mask: np.ndarray):
    if _HAS_CV2:
        n, labeled = _cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=8,
        )
        return labeled, n
    labeled = np.zeros_like(mask, dtype=np.int32)
    current_label = 0
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] and labeled[i, j] == 0:
                current_label += 1
                stack = [(i, j)]
                while stack:
                    ci, cj = stack.pop()
                    if labeled[ci, cj] != 0:
                        continue
                    labeled[ci, cj] = current_label
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            ni, nj = ci + di, cj + dj
                            if (
                                0 <= ni < h and 0 <= nj < w
                                and mask[ni, nj]
                                and labeled[ni, nj] == 0
                            ):
                                stack.append((ni, nj))
    return labeled, current_label


@dataclass
class FrontierCluster:
    centroid_gx: int
    centroid_gy: int
    centroid_wx: float
    centroid_wy: float
    size: int
    score: float = 0.0


def extract_frontiers(
    grid: OccupancyGrid,
    min_cluster_size: int = 3,
) -> List[FrontierCluster]:
    """Find frontier cells (free cells adjacent to unknown) and cluster them."""
    free_mask = grid.data < FREE_THRESH
    unknown_mask = (grid.data >= FREE_THRESH) & (grid.data <= OCC_THRESH)

    dilated_unknown = _binary_dilation_3x3(unknown_mask)
    frontier_mask = free_mask & dilated_unknown

    labeled, n_labels = _label_connected(frontier_mask)
    clusters: List[FrontierCluster] = []

    for i in range(1, n_labels + 1):
        cells = np.argwhere(labeled == i)
        if len(cells) < min_cluster_size:
            continue
        centroid = cells.mean(axis=0)
        cgx, cgy = int(round(centroid[0])), int(round(centroid[1]))
        cwx, cwy = grid.grid_to_world(cgx, cgy)
        clusters.append(FrontierCluster(
            centroid_gx=cgx,
            centroid_gy=cgy,
            centroid_wx=cwx,
            centroid_wy=cwy,
            size=len(cells),
        ))

    return clusters


def _count_unknown_in_radius(
    grid: OccupancyGrid, gx: int, gy: int, radius_cells: int,
) -> int:
    r = radius_cells
    x_lo = max(0, gx - r)
    x_hi = min(grid.cells, gx + r + 1)
    y_lo = max(0, gy - r)
    y_hi = min(grid.cells, gy + r + 1)
    patch = grid.data[x_lo:x_hi, y_lo:y_hi]
    return int(np.count_nonzero(
        (patch >= FREE_THRESH) & (patch <= OCC_THRESH)
    ))


def score_frontiers(
    frontiers: List[FrontierCluster],
    odom_x: float,
    odom_y: float,
    grid: OccupancyGrid,
    semantic_map: Optional[SemanticObjectMap] = None,
    current_target_id: int = -1,
    info_radius_cells: int = 10,
) -> List[FrontierCluster]:
    """Score each frontier, penalizing already-visited areas."""
    for f in frontiers:
        dist = math.hypot(f.centroid_wx - odom_x, f.centroid_wy - odom_y)
        cost = max(dist, 0.5)

        info_gain = _count_unknown_in_radius(
            grid, f.centroid_gx, f.centroid_gy, info_radius_cells,
        )

        semantic_bonus = 0.0
        if (
            semantic_map is not None
            and current_target_id >= 0
            and current_target_id not in semantic_map.instances
        ):
            semantic_bonus = info_gain * 0.5

        visited_n = grid.visited_count_in_radius(
            f.centroid_gx, f.centroid_gy, info_radius_cells,
        )
        total_n = grid.total_cells_in_radius(
            f.centroid_gx, f.centroid_gy, info_radius_cells,
        )
        visited_ratio = visited_n / max(total_n, 1)
        novelty = 1.0 - 0.8 * visited_ratio

        f.score = (info_gain + semantic_bonus) * novelty / cost

    frontiers.sort(key=lambda f: f.score, reverse=True)
    return frontiers


def select_best_frontier(
    grid: OccupancyGrid,
    odom_x: float,
    odom_y: float,
    semantic_map: Optional[SemanticObjectMap] = None,
    current_target_id: int = -1,
    min_cluster_size: int = 3,
) -> Optional[FrontierCluster]:
    """Extract, score, and return the single best frontier (or None)."""
    frontiers = extract_frontiers(grid, min_cluster_size)
    if not frontiers:
        return None
    scored = score_frontiers(
        frontiers, odom_x, odom_y, grid,
        semantic_map, current_target_id,
    )
    return scored[0]
