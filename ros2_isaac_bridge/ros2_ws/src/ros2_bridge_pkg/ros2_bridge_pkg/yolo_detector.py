"""
YOLO-based object detector using ultralytics YOLOv8.

Detects COCO objects relevant to the competition:
backpack, bottle, chair, cup, laptop.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO as _YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False


@dataclass
class Detection:
    class_name: str
    score: float
    bbox_xyxy: Tuple[int, int, int, int]


COCO_NAME_TO_MISSION: Dict[str, int] = {
    "backpack": 0,
    "bottle": 1,
    "chair": 2,
    "cup": 3,
    "laptop": 4,
}

MISSION_ID_TO_NAME: Dict[int, str] = {v: k for k, v in COCO_NAME_TO_MISSION.items()}

RELEVANT_COCO_NAMES = set(COCO_NAME_TO_MISSION.keys())


class YoloDetector:
    """Run YOLOv8-nano and return detections for competition objects."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.25,
        device: str = "cpu",
    ):
        if not _HAS_YOLO:
            raise RuntimeError(
                "ultralytics is not installed. "
                "Run: pip install ultralytics"
            )
        self.model = _YOLO(model_name)
        self.model.to(device)
        self.confidence = confidence

    def detect(self, image_rgb: np.ndarray) -> List[Detection]:
        """Run YOLO on a single RGB frame, return competition-relevant detections."""
        results = self.model.predict(
            image_rgb,
            conf=self.confidence,
            verbose=False,
        )
        if not results:
            return []

        detections: List[Detection] = []
        r = results[0]
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            if cls_name not in RELEVANT_COCO_NAMES:
                continue
            score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                class_name=cls_name,
                score=score,
                bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
            ))

        return detections
