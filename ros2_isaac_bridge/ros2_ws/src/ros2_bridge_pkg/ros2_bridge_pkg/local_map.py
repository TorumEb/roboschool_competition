"""
Occupancy grid (log-odds, depth raycasting) and semantic object map.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

DEPTH_W, DEPTH_H = 848, 480
DEPTH_CX = DEPTH_W / 2.0
DEPTH_FX = DEPTH_CX / math.tan(math.radians(86.0 / 2.0))

FREE_UPDATE = -0.4
OCC_UPDATE = 0.85
LOG_ODDS_MIN = -5.0
LOG_ODDS_MAX = 5.0
FREE_THRESH = -0.5
OCC_THRESH = 0.5


class OccupancyGrid:
    """2-D log-odds occupancy grid built from depth images."""

    def __init__(
        self,
        size_m: float = 30.0,
        resolution: float = 0.5,
        max_ray_m: float = 12.0,
    ):
        self.resolution = resolution
        self.max_ray_m = max_ray_m
        self.cells = int(size_m / resolution)
        self.half = self.cells // 2
        self.data = np.zeros((self.cells, self.cells), dtype=np.float32)
        self.visited = np.zeros((self.cells, self.cells), dtype=bool)

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = int(round(wx / self.resolution)) + self.half
        gy = int(round(wy / self.resolution)) + self.half
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = (gx - self.half) * self.resolution
        wy = (gy - self.half) * self.resolution
        return wx, wy

    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.cells and 0 <= gy < self.cells

    def is_free(self, gx: int, gy: int) -> bool:
        if not self._in_bounds(gx, gy):
            return False
        return float(self.data[gx, gy]) < FREE_THRESH

    def is_occupied(self, gx: int, gy: int) -> bool:
        if not self._in_bounds(gx, gy):
            return True
        return float(self.data[gx, gy]) > OCC_THRESH

    def is_unknown(self, gx: int, gy: int) -> bool:
        if not self._in_bounds(gx, gy):
            return False
        v = float(self.data[gx, gy])
        return FREE_THRESH <= v <= OCC_THRESH

    # ------------------------------------------------------------------
    def update_from_depth(
        self,
        depth_image: np.ndarray,
        odom_x: float,
        odom_y: float,
        odom_heading: float,
    ):
        """Project subsampled depth columns into the grid via raycasting."""
        h, w = depth_image.shape
        row_lo = int(h * 0.30)
        row_hi = int(h * 0.70)
        step = 8

        for u in range(0, w, step):
            col = np.abs(depth_image[row_lo:row_hi, u])
            valid = col[(col > 0.1) & (col < 20.0) & np.isfinite(col)]
            bearing = math.atan2(u - DEPTH_CX, DEPTH_FX)
            ray_angle = odom_heading + bearing

            if valid.size < 2:
                self._trace_ray_free_only(odom_x, odom_y, ray_angle,
                                          self.max_ray_m)
                continue
            d = float(np.median(valid))
            d = min(d, self.max_ray_m)
            self._trace_ray(odom_x, odom_y, ray_angle, d)

    def _trace_ray(
        self, ox: float, oy: float, angle: float, dist: float
    ):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        step = self.resolution * 0.5
        d = step
        while d < dist - self.resolution:
            wx = ox + d * cos_a
            wy = oy + d * sin_a
            gx, gy = self.world_to_grid(wx, wy)
            if self._in_bounds(gx, gy):
                self.data[gx, gy] = max(
                    LOG_ODDS_MIN, self.data[gx, gy] + FREE_UPDATE
                )
            d += step

        ex = ox + dist * cos_a
        ey = oy + dist * sin_a
        gx, gy = self.world_to_grid(ex, ey)
        if self._in_bounds(gx, gy) and dist < self.max_ray_m - 0.1:
            self.data[gx, gy] = min(
                LOG_ODDS_MAX, self.data[gx, gy] + OCC_UPDATE
            )

    def _trace_ray_free_only(
        self, ox: float, oy: float, angle: float, dist: float
    ):
        """Trace a ray marking all cells as free (no occupied endpoint).

        Used when depth has no valid reading — the camera looked there
        and saw nothing, so the space is free.
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        step = self.resolution
        d = step
        while d < dist:
            wx = ox + d * cos_a
            wy = oy + d * sin_a
            gx, gy = self.world_to_grid(wx, wy)
            if self._in_bounds(gx, gy):
                self.data[gx, gy] = max(
                    LOG_ODDS_MIN, self.data[gx, gy] + FREE_UPDATE
                )
            d += step

    # ------------------------------------------------------------------
    # Visited tracking
    # ------------------------------------------------------------------
    def mark_visited(self, wx: float, wy: float, radius_cells: int = 2):
        """Mark cells near the robot position as visited."""
        gx, gy = self.world_to_grid(wx, wy)
        r = radius_cells
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = gx + dx, gy + dy
                if self._in_bounds(nx, ny):
                    self.visited[nx, ny] = True

    def visited_count_in_radius(
        self, gx: int, gy: int, radius_cells: int = 5,
    ) -> int:
        r = radius_cells
        x_lo = max(0, gx - r)
        x_hi = min(self.cells, gx + r + 1)
        y_lo = max(0, gy - r)
        y_hi = min(self.cells, gy + r + 1)
        return int(np.count_nonzero(self.visited[x_lo:x_hi, y_lo:y_hi]))

    def total_cells_in_radius(
        self, gx: int, gy: int, radius_cells: int = 5,
    ) -> int:
        r = radius_cells
        x_lo = max(0, gx - r)
        x_hi = min(self.cells, gx + r + 1)
        y_lo = max(0, gy - r)
        y_hi = min(self.cells, gy + r + 1)
        return (x_hi - x_lo) * (y_hi - y_lo)

    # ------------------------------------------------------------------
    def get_free_mask(self) -> np.ndarray:
        return self.data < FREE_THRESH

    def get_occupied_mask(self) -> np.ndarray:
        return self.data > OCC_THRESH

    def get_unknown_mask(self) -> np.ndarray:
        return (self.data >= FREE_THRESH) & (self.data <= OCC_THRESH)


# ======================================================================
# Semantic object map
# ======================================================================

@dataclass
class ObjectInstance:
    class_name: str
    object_id: int
    world_x: float
    world_y: float
    confidence: float = 0.15
    detections_count: int = 1
    last_seen_time: float = 0.0
    confirmed: bool = False


class SemanticObjectMap:
    """Stores detected object instances keyed by object_id."""

    CONFIRM_COUNT = 5
    EMA_ALPHA = 0.3
    MAX_ASSOC_DIST = 3.0

    def __init__(self):
        self.instances: dict[int, ObjectInstance] = {}

    def update_detection(
        self,
        object_id: int,
        class_name: str,
        world_x: float,
        world_y: float,
        now: float,
    ):
        if object_id in self.instances:
            inst = self.instances[object_id]
            a = self.EMA_ALPHA
            dist = math.hypot(world_x - inst.world_x, world_y - inst.world_y)
            if dist < self.MAX_ASSOC_DIST:
                inst.world_x = inst.world_x * (1 - a) + world_x * a
                inst.world_y = inst.world_y * (1 - a) + world_y * a
            else:
                inst.world_x = world_x
                inst.world_y = world_y
            inst.confidence = min(1.0, inst.confidence + 0.1)
            inst.detections_count += 1
            inst.last_seen_time = now
            if inst.detections_count >= self.CONFIRM_COUNT:
                inst.confirmed = True
        else:
            self.instances[object_id] = ObjectInstance(
                class_name=class_name,
                object_id=object_id,
                world_x=world_x,
                world_y=world_y,
                last_seen_time=now,
            )

    def remove(self, object_id: int):
        self.instances.pop(object_id, None)

    def get(self, object_id: int) -> Optional[ObjectInstance]:
        return self.instances.get(object_id)
