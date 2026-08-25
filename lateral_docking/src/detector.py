"""LED array detector with YOLO.

Detection pipeline (Ye Li et al. §4.3 inspired):
  1. YOLO predicts bounding boxes (replaces Mean-Shift candidate extraction).
  2. Bounding-box centers are used directly as feature points.
  3. Geometric validation via rectangular constraint (replaces SVM filtering).
"""

from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    DEBUG,
    YOLO_CENTER_CLASS_ID,
    YOLO_CORNER_CLASS_ID,
    GEOMETRY_REJECT_INVALID,
    GEOMETRY_MIN_AREA_PX2,
    GEOMETRY_MAX_ASPECT_RATIO,
    GEOMETRY_MIN_ANGLE_DEG,
)
from corner_refinement import validate_rectangular_geometry


class ObjectDetector:
    """Detects underwater LED targets using YOLO.

    This class wraps the YOLO model to detect four corner points and a
    dedicated center-LED point.  The center point is taken directly from the
    YOLO ``YOLO_CENTER_CLASS_ID`` output (no extra geometric in-corners
    check); when present alongside four corners it enables 5-point PnP mode,
    otherwise 4-point PnP mode is used.

    The raw YOLO bounding-box centers are used directly as feature points.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        debug: bool = False,
    ) -> None:
        """Initializes the detector.

        Args:
            model_path: Path to the YOLO model weights.
            debug: Whether to enable debug printouts.
        """
        self.model = YOLO(model_path)
        self.debug = debug
        self.model.to("cuda")
        self.results = None
        self._raw_targets: List[List] = []

    def detect(self, image: np.ndarray) -> None:
        """Runs YOLO inference on the input image.

        Args:
            image: Input BGR image.
        """
        self.results = self.model.predict(
            source=image, conf=0.01, iou=0.6, verbose=False
        )
        self._parse_results()

    def _parse_results(self) -> None:
        """Parses the latest YOLO results into an internal target list."""
        self._raw_targets = []
        if self.results is None or len(self.results) == 0:
            return

        for box in self.results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if (x2 - x1) < 5.0 or (y2 - y1) < 5.0:
                continue
            self._raw_targets.append([
                float(box.conf[0]),
                [float(x1), float(y1), float(x2), float(y2)],
                int(box.cls[0]),
            ])

    def get_points(
        self,
        image: np.ndarray,
        vis_frame: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], int, np.ndarray]:
        """Extracts the final point list for PnP solving using YOLO only.

        Args:
            image: Original BGR image (kept for API compatibility).
            vis_frame: Frame to draw annotations on.

        Returns:
            A tuple of:
                - points: List of (x, y) pixel coordinates.
                - mode: 0 for 4-point PnP, 1 for 5-point PnP. A negative value
                  indicates that not enough points were found.
                - vis_frame: Annotated visualization frame.
        """
        points, mode, vis_frame = self._extract_yolo_points(vis_frame)
        vis_frame = self._draw_selected_points(vis_frame, points, mode)
        return points, mode, vis_frame

    def _draw_selected_points(
        self,
        frame: np.ndarray,
        points: List[Tuple[int, int]],
        mode: int,
    ) -> np.ndarray:
        """Draw the selected PnP points on the frame.

        Corner points are drawn as green circles with indices 1-4 and
        connected by a green polygon.  If mode == 1, the centre point is
        drawn as a red circle with index 5 and linked to the corners.

        Args:
            frame: Input BGR image.
            points: List of selected (x, y) pixel coordinates.
            mode: 0 for 4-point, 1 for 5-point, negative for insufficient
                points.

        Returns:
            Annotated frame.
        """
        if not points:
            return frame

        vis = frame.copy()
        corner_colour = (255, 0, 0)      # Blue
        centre_colour = (0, 0, 255)      # Red
        text_colour = (255, 255, 255)    # White
        font = cv2.FONT_HERSHEY_SIMPLEX
        half_size = 6
        thickness = 2

        num_corners = len(points) if mode < 1 else len(points) - 1
        corners = points[:num_corners]

        def _draw_box(
            x: int, y: int, colour: Tuple[int, int, int]
        ) -> None:
            cv2.rectangle(
                vis,
                (x - half_size, y - half_size),
                (x + half_size, y + half_size),
                colour,
                thickness,
                cv2.LINE_AA,
            )

        # Draw corner points and indices
        for idx, (cx, cy) in enumerate(corners, start=1):
            _draw_box(int(cx), int(cy), corner_colour)

        # Draw centre point if present
        if mode == 1 and len(points) >= 5:
            cx, cy = points[4]
            _draw_box(int(cx), int(cy), centre_colour)
            cv2.putText(
                vis,
                "5",
                (int(cx) + 8, int(cy) - 8),
                font,
                0.6,
                text_colour,
                thickness,
                cv2.LINE_AA,
            )

        return vis

    def _extract_yolo_points(
        self,
        vis_frame: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], int, np.ndarray]:
        """Extracts points using YOLO bounding-box centers.

        Uses the center of each YOLO bounding box directly as the feature
        point, then validates the quadrilateral geometry.  When the YOLO model
        also predicts a center-LED point, it is trusted as-is and appended to
        the four corners for 5-point PnP.

        Args:
            vis_frame: Frame to draw annotations on.

        Returns:
            See :meth:`get_points`.
        """
        # Use the YOLO bounding-box centers directly as feature points.
        corners_raw: List[Tuple[float, float]] = []
        centers_raw: List[Tuple[float, float]] = []
        for conf, bbox, cls_id in self._raw_targets:
            if conf < 0.1:
                continue
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if cls_id == YOLO_CORNER_CLASS_ID:
                corners_raw.append((cx, cy))
            elif cls_id == YOLO_CENTER_CLASS_ID:
                centers_raw.append((cx, cy))

        # Sort corners by confidence (already sorted in _raw_targets).
        raw_corners: List[Tuple[int, int]] = [
            (int(round(c[0])), int(round(c[1]))) for c in corners_raw
        ]

        # Keep only the best 4 corners for PnP.
        if len(raw_corners) > 4:
            raw_corners = raw_corners[:4]

        points: List[Tuple[int, int]] = raw_corners

        # Validate rectangular geometry (paper §4.3 SVM replacement).
        if len(points) == 4:
            diag: dict = {}
            valid_geom = validate_rectangular_geometry(
                points,
                min_area=GEOMETRY_MIN_AREA_PX2,
                max_aspect_ratio=GEOMETRY_MAX_ASPECT_RATIO,
                min_angle_deg=GEOMETRY_MIN_ANGLE_DEG,
                diagnostics=diag,
            )
            if not valid_geom:
                if self.debug:
                    area = diag.get("area")
                    aspect = diag.get("aspect_ratio")
                    min_angle = diag.get("min_angle")
                    area_str = f"{area:.1f}" if area is not None else "n/a"
                    aspect_str = f"{aspect:.2f}" if aspect is not None else "n/a"
                    angle_str = f"{min_angle:.1f}" if min_angle is not None else "n/a"
                    print(
                        f"Corner geometry validation failed (points={points}, "
                        f"reason={diag.get('reason', 'unknown')}, "
                        f"area={area_str}, "
                        f"aspect={aspect_str}, "
                        f"min_angle={angle_str})."
                    )
                if GEOMETRY_REJECT_INVALID:
                    points = []

        # Trust the YOLO-predicted center point directly (no extra geometric
        # in-corners check).  The center LED class is trained specifically for
        # this role, so the first (highest-confidence) center is appended to
        # the corners for 5-point PnP.
        if centers_raw and len(points) >= 4:
            cx, cy = int(round(centers_raw[0][0])), int(round(centers_raw[0][1]))
            points.append((cx, cy))
            if self.debug:
                print(
                    f"YOLO center detected at ({cx}, {cy}), using 5-point PnP."
                )
            return points, 1, vis_frame

        if len(points) >= 4:
            if self.debug:
                print("No YOLO center detected, using 4-point PnP.")
            return points, 0, vis_frame

        if self.debug:
            print("Not enough corner points detected.")
        return points, -1, vis_frame

