"""Traditional feature detector for track_mode=1.

This module implements the "traditional" tracking path documented in
``CLAUDE.md``: detect bright blobs, take the largest one as the tracking
target, and report its centroid and projected area.  It reuses the existing
``NwpuThresholdDetector`` so tuning parameters stay in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from config import THRESH_FALLBACK_PARAMS
from nwpu_threshold_detector import Detection, DetectionResult, NwpuThresholdDetector


@dataclass
class TraditionalTarget:
    """A single traditional-detector target."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]          # cx, cy
    area_px: float
    detection: Detection             # raw threshold detector detection


class TraditionalFeatureDetector:
    """Traditional blob detector for lateral docking track_mode=1.

    Detects bright blobs using ``NwpuThresholdDetector`` and selects the
    largest-area blob as the final tracking target, matching the description
    in ``CLAUDE.md``.

    Args:
        params: Optional kwargs passed to ``NwpuThresholdDetector``.  When
            ``None``, ``THRESH_FALLBACK_PARAMS`` from ``config.py`` is used.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        if params is None:
            params = THRESH_FALLBACK_PARAMS
        self.detector = NwpuThresholdDetector(**params)
        self.last_result: Optional[DetectionResult] = None

    def detect(self, image: np.ndarray) -> Optional[TraditionalTarget]:
        """Run traditional detection and return the largest target.

        Args:
            image: Input BGR image.

        Returns:
            A ``TraditionalTarget`` for the largest blob, or ``None`` if no
            blob is detected.
        """
        result = self.detector.detect(image)
        self.last_result = result
        if not result.detections:
            return None

        best = max(result.detections, key=lambda d: d.area)
        return TraditionalTarget(
            bbox=best.bbox,
            center=best.center,
            area_px=float(best.area),
            detection=best,
        )

    def visualize(
        self,
        image: np.ndarray,
        target: Optional[TraditionalTarget] = None,
    ) -> np.ndarray:
        """Draw the traditional target on the image.

        Args:
            image: Input BGR image.
            target: Target to draw.  If ``None``, the last detection is used.

        Returns:
            Annotated BGR image.
        """
        import cv2

        vis = image.copy()
        if target is None and self.last_result is not None:
            candidates = [
                TraditionalTarget(
                    bbox=d.bbox,
                    center=d.center,
                    area_px=float(d.area),
                    detection=d,
                )
                for d in self.last_result.detections
            ]
            if candidates:
                target = max(candidates, key=lambda t: t.area_px)

        if target is None:
            cv2.putText(
                vis,
                "TRAD: no target",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return vis

        x1, y1, x2, y2 = target.bbox
        cx, cy = target.center
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"TRAD: area={int(target.area_px)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return vis
