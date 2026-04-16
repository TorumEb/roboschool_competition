"""
Multi-class perception: scans all object classes, estimates world positions.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ros2_bridge_pkg.reference_detector import (
    ReferenceImageDetectorBackend,
    Detection,
)

RGB_W, RGB_H = 640, 360
RGB_CX = RGB_W / 2.0
RGB_CY = RGB_H / 2.0
RGB_FX = RGB_CX / math.tan(math.radians(70.0 / 2.0))
DEPTH_W, DEPTH_H = 848, 480
DEPTH_CX = DEPTH_W / 2.0
DEPTH_CY = DEPTH_H / 2.0
DEPTH_FX = DEPTH_CX / math.tan(math.radians(86.0 / 2.0))

ALL_OBJECTS: Dict[int, str] = {
    0: "backpack",
    1: "bottle",
    2: "chair",
    3: "cup",
    4: "laptop",
}


def rgb_pixel_to_depth_pixel(x_rgb: float, y_rgb: float):
    x_depth = (x_rgb - RGB_CX) * (DEPTH_FX / RGB_FX) + DEPTH_CX
    y_depth = (y_rgb - RGB_CY) * (DEPTH_FX / RGB_FX) + DEPTH_CY
    return x_depth, y_depth


@dataclass
class LocalizedDetection:
    detection: Detection
    object_id: int
    world_x: float
    world_y: float
    depth: float


def _median_depth_at_bbox(
    depth_image: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Optional[float]:
    """Median depth in the central 50% of bbox (mapped to depth coords)."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    hw = (x2 - x1) * 0.25
    hh = (y2 - y1) * 0.25

    dx1, dy1 = rgb_pixel_to_depth_pixel(cx - hw, cy - hh)
    dx2, dy2 = rgb_pixel_to_depth_pixel(cx + hw, cy + hh)

    dh, dw = depth_image.shape
    ix1 = max(0, int(dx1))
    iy1 = max(0, int(dy1))
    ix2 = min(dw, int(dx2))
    iy2 = min(dh, int(dy2))
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    roi = np.abs(depth_image[iy1:iy2, ix1:ix2])
    valid = roi[(roi > 0.1) & (roi < 20.0) & np.isfinite(roi)]
    if valid.size < 4:
        return None
    return float(np.median(valid))


def bbox_to_world_pos(
    bbox: Tuple[int, int, int, int],
    depth_image: np.ndarray,
    odom_x: float,
    odom_y: float,
    odom_heading: float,
) -> Optional[Tuple[float, float, float]]:
    """Return (world_x, world_y, depth) from a bbox + depth + odom."""
    d = _median_depth_at_bbox(depth_image, bbox)
    if d is None:
        return None
    cx_rgb = (bbox[0] + bbox[2]) / 2.0
    bearing = math.atan2(cx_rgb - RGB_CX, RGB_FX)
    wx = odom_x + d * math.cos(odom_heading + bearing)
    wy = odom_y + d * math.sin(odom_heading + bearing)
    return wx, wy, d


class MultiClassScanner:
    """Scan all (or a subset of) object classes each tick."""

    def __init__(
        self,
        detector: ReferenceImageDetectorBackend,
        min_score: float = 0.12,
    ):
        self.detector = detector
        self.min_score = min_score
        self._found_ids: set[int] = set()

    def mark_found(self, object_id: int):
        """Mark an object as fully reported — skip it in future scans."""
        self._found_ids.add(object_id)

    def scan(
        self,
        image_rgb: np.ndarray,
        depth_image: Optional[np.ndarray],
        odom_x: float,
        odom_y: float,
        odom_heading: float,
        priority_ids: Optional[List[int]] = None,
    ) -> List[LocalizedDetection]:
        """Detect objects in the current frame."""
        results: List[LocalizedDetection] = []

        scan_order: List[int] = []
        if priority_ids:
            scan_order.extend(
                oid for oid in priority_ids if oid not in self._found_ids
            )
        for oid in ALL_OBJECTS:
            if oid not in self._found_ids and oid not in scan_order:
                scan_order.append(oid)

        for oid in scan_order:
            name = ALL_OBJECTS[oid]
            dets = self.detector.detect(image_rgb, name)
            if not dets:
                continue
            best = max(dets, key=lambda d: d.score)
            if best.score < self.min_score:
                continue

            if depth_image is not None:
                wp = bbox_to_world_pos(
                    best.bbox_xyxy, depth_image,
                    odom_x, odom_y, odom_heading,
                )
            else:
                wp = None

            if wp is not None:
                results.append(LocalizedDetection(
                    detection=best,
                    object_id=oid,
                    world_x=wp[0],
                    world_y=wp[1],
                    depth=wp[2],
                ))
            else:
                results.append(LocalizedDetection(
                    detection=best,
                    object_id=oid,
                    world_x=float("nan"),
                    world_y=float("nan"),
                    depth=float("nan"),
                ))

        return results


class ConfirmationBuffer:
    """Require N consecutive frames of consistent detection."""

    def __init__(self, required: int = 3, max_depth_spread: float = 1.5):
        self.required = required
        self.max_depth_spread = max_depth_spread
        self._buffer: List[LocalizedDetection] = []

    def reset(self):
        self._buffer.clear()

    def push(self, det: Optional[LocalizedDetection]) -> bool:
        if det is None:
            self._buffer.clear()
            return False
        self._buffer.append(det)
        if len(self._buffer) < self.required:
            return False
        self._buffer = self._buffer[-self.required:]
        depths = [
            d.depth for d in self._buffer if math.isfinite(d.depth)
        ]
        if len(depths) >= 2:
            if max(depths) - min(depths) > self.max_depth_spread:
                self._buffer.clear()
                return False
        return True
