"""
A* path planner on the occupancy grid and pure-pursuit path follower.
"""

import heapq
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from ros2_bridge_pkg.local_map import OccupancyGrid, OCC_THRESH

_SQRT2 = math.sqrt(2.0)
_NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),
]
UNKNOWN_COST_MULT = 1.5
# Large inflation so planned paths stay well clear of walls / boxes on the grid.
INFLATE_CELLS = 10
MAX_ASTAR_EXPANSIONS = 30_000


@dataclass
class PlannedPath:
    cells: List[Tuple[int, int]]
    goal_world: Tuple[float, float]
    goal_type: str  # "frontier" | "object"
    timestamp: float = 0.0

    def is_empty(self) -> bool:
        return len(self.cells) == 0


def plan_path(
    grid: OccupancyGrid,
    start_wx: float,
    start_wy: float,
    goal_wx: float,
    goal_wy: float,
    goal_type: str = "frontier",
    now: float = 0.0,
) -> Optional[PlannedPath]:
    """A* from start to goal on the occupancy grid. Returns None if no path."""
    sx, sy = grid.world_to_grid(start_wx, start_wy)
    gx, gy = grid.world_to_grid(goal_wx, goal_wy)

    occ_mask = grid.data > OCC_THRESH
    if _HAS_CV2:
        k = np.ones((3, 3), dtype=np.uint8)
        inflated = _cv2.dilate(
            occ_mask.astype(np.uint8), k, iterations=INFLATE_CELLS,
        ).astype(bool)
    else:
        inflated = occ_mask.copy()
        for _ in range(INFLATE_CELLS):
            grown = inflated.copy()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    grown |= np.roll(np.roll(inflated, dx, axis=0), dy, axis=1)
            inflated = grown

    if not grid._in_bounds(sx, sy) or not grid._in_bounds(gx, gy):
        return None
    if inflated[sx, sy]:
        sx, sy = _find_nearest_free(inflated, sx, sy, grid.cells)
        if sx < 0:
            return None
    if inflated[gx, gy]:
        gx, gy = _find_nearest_free(inflated, gx, gy, grid.cells)
        if gx < 0:
            return None

    cells = _astar(inflated, grid.data, sx, sy, gx, gy, grid.cells)
    if cells is None:
        return None

    return PlannedPath(
        cells=cells,
        goal_world=(goal_wx, goal_wy),
        goal_type=goal_type,
        timestamp=now,
    )


def _find_nearest_free(
    inflated: np.ndarray, gx: int, gy: int, size: int, search_r: int = 40,
) -> Tuple[int, int]:
    """If goal is inside an obstacle, find closest free cell."""
    best_d = float("inf")
    bx, by = -1, -1
    for dx in range(-search_r, search_r + 1):
        for dy in range(-search_r, search_r + 1):
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < size and 0 <= ny < size and not inflated[nx, ny]:
                d2 = dx * dx + dy * dy
                if d2 < best_d:
                    best_d = d2
                    bx, by = nx, ny
    return bx, by


def _astar(
    inflated: np.ndarray,
    raw_grid: np.ndarray,
    sx: int, sy: int,
    gx: int, gy: int,
    size: int,
) -> Optional[List[Tuple[int, int]]]:
    def heuristic(x, y):
        return math.hypot(x - gx, y - gy)

    open_heap: list = []
    heapq.heappush(open_heap, (heuristic(sx, sy), 0.0, sx, sy))
    came_from: dict = {}
    g_score = {(sx, sy): 0.0}
    expanded = 0

    while open_heap and expanded < MAX_ASTAR_EXPANSIONS:
        _, g, cx, cy = heapq.heappop(open_heap)
        expanded += 1

        if cx == gx and cy == gy:
            return _reconstruct(came_from, gx, gy)

        if g > g_score.get((cx, cy), float("inf")):
            continue

        for dx, dy, base_cost in _NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < size and 0 <= ny < size):
                continue
            if inflated[nx, ny]:
                continue
            cost = base_cost
            val = raw_grid[nx, ny]
            if -0.5 <= val <= 0.5:
                cost *= UNKNOWN_COST_MULT
            tentative = g + cost
            if tentative < g_score.get((nx, ny), float("inf")):
                g_score[(nx, ny)] = tentative
                f = tentative + heuristic(nx, ny)
                heapq.heappush(open_heap, (f, tentative, nx, ny))
                came_from[(nx, ny)] = (cx, cy)

    return None


def _reconstruct(
    came_from: dict, gx: int, gy: int,
) -> List[Tuple[int, int]]:
    path = [(gx, gy)]
    while (gx, gy) in came_from:
        gx, gy = came_from[(gx, gy)]
        path.append((gx, gy))
    path.reverse()
    return path


# ======================================================================
# Path follower (pure pursuit style)
# ======================================================================

class PathFollower:
    """Convert a grid path to velocity commands."""

    def __init__(
        self,
        grid: OccupancyGrid,
        lookahead_cells: int = 5,
        heading_kp: float = 2.5,
        max_vx: float = 1.0,
        slow_vx: float = 0.3,
        heading_slow_threshold: float = 0.5,
    ):
        self.grid = grid
        self.lookahead_cells = lookahead_cells
        self.heading_kp = heading_kp
        self.max_vx = max_vx
        self.slow_vx = slow_vx
        self.heading_slow_threshold = heading_slow_threshold

    def follow(
        self,
        path: PlannedPath,
        odom_x: float,
        odom_y: float,
        odom_heading: float,
    ) -> Tuple[float, float, float]:
        """Return (vx, vy, wz)."""
        if path.is_empty():
            return 0.0, 0.0, 0.0

        closest_idx = self._closest_on_path(path, odom_x, odom_y)
        la_idx = min(closest_idx + self.lookahead_cells, len(path.cells) - 1)
        target_gx, target_gy = path.cells[la_idx]
        twx, twy = self.grid.grid_to_world(target_gx, target_gy)

        target_heading = math.atan2(twy - odom_y, twx - odom_x)
        heading_err = _angle_diff(target_heading, odom_heading)

        wz = self.heading_kp * heading_err
        if abs(heading_err) > self.heading_slow_threshold:
            vx = self.slow_vx
        else:
            vx = self.max_vx

        return vx, 0.0, wz

    def is_path_complete(
        self,
        path: PlannedPath,
        odom_x: float,
        odom_y: float,
        threshold_m: float = 0.6,
    ) -> bool:
        if path.is_empty():
            return True
        gx, gy = path.cells[-1]
        gwx, gwy = self.grid.grid_to_world(gx, gy)
        return math.hypot(gwx - odom_x, gwy - odom_y) < threshold_m

    def _closest_on_path(
        self,
        path: PlannedPath,
        ox: float,
        oy: float,
    ) -> int:
        best_i = 0
        best_d = float("inf")
        for i, (gx, gy) in enumerate(path.cells):
            wx, wy = self.grid.grid_to_world(gx, gy)
            d2 = (wx - ox) ** 2 + (wy - oy) ** 2
            if d2 < best_d:
                best_d = d2
                best_i = i
        return best_i


def _angle_diff(target: float, current: float) -> float:
    d = target - current
    return (d + math.pi) % (2.0 * math.pi) - math.pi


# ======================================================================
# Safety layer
# ======================================================================

def obstacle_safety(
    vx: float,
    vy: float,
    wz: float,
    forward_depth: Optional[float],
    soft_threshold: float = 2.8,
    hard_threshold: float = 1.15,
    reverse_threshold: float = 0.45,
) -> Tuple[float, float, float]:
    """Depth-based avoidance: slow early, stop, then reverse when very close."""
    if forward_depth is None or vx <= 0:
        return vx, vy, wz

    if forward_depth < reverse_threshold:
        return -0.35, vy, (0.9 if wz >= 0 else -0.9)

    if forward_depth < hard_threshold:
        return 0.0, vy, (0.75 if wz >= 0 else -0.75)

    if forward_depth < soft_threshold and soft_threshold > hard_threshold:
        t = (forward_depth - hard_threshold) / (soft_threshold - hard_threshold)
        t = max(0.0, min(1.0, t))
        return vx * t, vy, wz

    return vx, vy, wz
