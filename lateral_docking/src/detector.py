"""LED array detector using YOLO only."""

from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    DEBUG,
    YOLO_CENTER_CLASS_ID,
    YOLO_CORNER_CLASS_ID,
)


class ObjectDetector:
    """Detects underwater LED targets using YOLO only.

    This class wraps the YOLO model to detect corner points and optionally
    a center point. If a center point is detected alongside 4 corners,
    5-point PnP mode is used; otherwise 4-point PnP mode is used.
    """

    def __init__(self, model_path: str = "yolov8n.pt", debug: bool = False) -> None:
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
            source=image, conf=0.1, iou=0.8, verbose=False
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

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Returns the YOLO-annotated frame.

        Returns:
            The annotated BGR image, or None if no results are available.
        """
        if self.results is None or len(self.results) == 0:
            return None
        return self.results[0].plot()

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

    def _is_center_in_corners(
        self,
        center: Tuple[int, int],
        corners: List[Tuple[int, int]],
    ) -> bool:
        """Check whether the centre point lies inside the convex hull of corners.

        Args:
            center: (x, y) of the candidate centre point.
            corners: List of (x, y) corner points.

        Returns:
            True if the centre point is inside or on the edge of the convex
            hull formed by the corners.
        """
        if len(corners) < 4:
            return False
        contour = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(contour)
        return cv2.pointPolygonTest(hull, center, False) >= 0

    def _extract_yolo_points(
        self,
        vis_frame: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], int, np.ndarray]:
        """Extracts points using YOLO class separation (mode 1).

        Args:
            vis_frame: Frame to draw annotations on.

        Returns:
            See :meth:`get_points`.
        """
        corner_list: List[Tuple[float, List[float]]] = []
        center_list: List[Tuple[float, List[float]]] = []

        for conf, box, cls_id in self._raw_targets:
            if cls_id == YOLO_CENTER_CLASS_ID:
                center_list.append((conf, box))
            elif cls_id == YOLO_CORNER_CLASS_ID:
                corner_list.append((conf, box))

        raw_corners: List[Tuple[int, int]] = []
        for _, box in corner_list:
            x1, y1, x2, y2 = map(int, box)
            raw_corners.append(((x1 + x2) // 2, (y1 + y2) // 2))

        # Keep only the bottom-most 4 corners for PnP.
        if len(raw_corners) > 4:
            raw_corners.sort(key=lambda p: p[1], reverse=True)
            raw_corners = raw_corners[:4]

        points: List[Tuple[int, int]] = raw_corners

        if center_list and len(points) >= 4:
            center_list.sort(key=lambda x: x[0], reverse=True)
            _, box = center_list[0]
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if self._is_center_in_corners((cx, cy), points):
                points.append((cx, cy))
                if self.debug:
                    print(
                        f"YOLO center detected at ({cx}, {cy}), "
                        f"using 5-point PnP."
                    )
                return points, 1, vis_frame
            if self.debug:
                print(
                    f"YOLO center at ({cx}, {cy}) is outside corner "
                    f"region, falling back to 4-point PnP."
                )

        if len(points) >= 4:
            if self.debug:
                print("No YOLO center detected, using 4-point PnP.")
            return points, 0, vis_frame

        if self.debug:
            print("Not enough corner points detected.")
        return points, -1, vis_frame

    @staticmethod
    def _draw_yolo_center(
        frame: np.ndarray,
        cx: int,
        cy: int,
    ) -> np.ndarray:
        """Draws the YOLO-detected center point on the frame.

        Args:
            frame: Input BGR image.
            cx: Center x-coordinate.
            cy: Center y-coordinate.

        Returns:
            The annotated frame.
        """
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            "YOLO Center",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        return frame
