"""LED array detector integrating YOLO and traditional center-point methods."""

from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CENTER_POINT_MODE,
    DEBUG,
    YOLO_CENTER_CLASS_ID,
    YOLO_CORNER_CLASS_ID,
)
from red_led_detector import TargetPointDetector


class ObjectDetector:
    """Detects underwater LED targets using YOLO or traditional red-LED methods.

    This class wraps the YOLO model and optionally falls back to a traditional
    image-processing-based red LED detector for the center point. The detection
    mode is controlled by :data:`config.CENTER_POINT_MODE`.
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
        self._red_detector = TargetPointDetector()

    def detect(self, image: np.ndarray) -> None:
        """Runs YOLO inference on the input image.

        Args:
            image: Input BGR image.
        """
        self.results = self.model.predict(
            source=image, conf=0.1, iou=0.6, verbose=False
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
        """Extracts the final point list for PnP solving.

        Depending on :data:`CENTER_POINT_MODE`, this either uses YOLO class
        separation or the traditional red-LED detector to identify the center
        point.

        Args:
            image: Original BGR image (used for traditional mode).
            vis_frame: Frame to draw annotations on.

        Returns:
            A tuple of:
                - points: List of (x, y) pixel coordinates.
                - mode: 0 for 4-point PnP, 1 for 5-point PnP. A negative value
                  indicates that not enough points were found.
                - vis_frame: Annotated visualization frame.
        """
        # Step 1: Always use YOLO for corner points first.
        points, mode, vis_frame = self._extract_yolo_points(vis_frame)

        # Step 2: If YOLO provides 4 corners but no centre, fall back to the
        # traditional red-LED detector for the centre point.
        if mode == 0 and len(points) >= 4:
            red_targets = self._red_detector.detect_all(image)
            if red_targets:
                rx, ry = red_targets[0]["center"]
                if self._is_center_in_corners((rx, ry), points):
                    points.append((rx, ry))
                    mode = 1
                    if self.debug:
                        print(
                            f"Traditional center detected at ({rx}, {ry}), "
                            f"using 5-point PnP."
                        )
                elif self.debug:
                    print(
                        f"Traditional center at ({rx}, {ry}) is outside "
                        f"corner region, using 4-point PnP."
                    )

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

    def _extract_traditional_points(
        self,
        image: np.ndarray,
        vis_frame: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], int, np.ndarray]:
        """Extracts points using the traditional red-LED detector (mode 0).

        Args:
            image: Original BGR image for red-LED detection.
            vis_frame: Frame to draw annotations on.

        Returns:
            See :meth:`get_points`.
        """
        # Filter out YOLO center-class boxes to avoid interference.
        corner_targets: List[List] = []
        for conf, box, cls_id in self._raw_targets:
            if cls_id == YOLO_CENTER_CLASS_ID:
                continue
            corner_targets.append([conf, box])

        # Detect the red LED center using the traditional method.
        red_targets = self._red_detector.detect_all(image)

        # Remove YOLO boxes that overlap with the red LED.
        if red_targets:
            rx, ry = red_targets[0]["center"]
            filtered: List[List] = []
            for conf, box in corner_targets:
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                dist = ((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5
                if dist < 10:
                    if self.debug:
                        print(
                            f"Filtering YOLO box at "
                            f"({cx},{cy}) overlapping with red LED"
                        )
                    continue
                filtered.append([conf, box])
            corner_targets = filtered

        if len(corner_targets) < 4:
            if self.debug:
                print("Not enough points detected.")
            return [], -1, vis_frame

        points: List[Tuple[int, int]] = []
        for conf, box in corner_targets:
            x1, y1, x2, y2 = map(int, box)
            points.append(((x1 + x2) // 2, (y1 + y2) // 2))

        if red_targets:
            rx, ry = red_targets[0]["center"]
            points.append((rx, ry))

            if self.debug:
                print(
                    f"Red LED detected at ({rx}, {ry}), "
                    f"using 5-point PnP."
                )
            return points, 1, vis_frame

        if self.debug:
            print("No red LED detected, using 4-point PnP.")
        return points, 0, vis_frame

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
