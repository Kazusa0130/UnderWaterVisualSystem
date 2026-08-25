"""Rectangular geometry validation for the LED target (Ye Li et al. §4.3).

Paper §4.3 three-level processing:
  1. Mean-Shift  → candidate bright regions (we use YOLO instead)
  2. Snake       → smooth contour + precise center (we use YOLO box centers)
  3. SVM         → filter false positives (we use geometric validation)

This module retains only the third level: validating that the four detected
corner points form a reasonable quadrilateral.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


def validate_rectangular_geometry(
    points: List[Tuple[float, float]],
    min_area: float = 100.0,
    max_aspect_ratio: float = 5.0,
    min_angle_deg: float = 30.0,
    diagnostics: Optional[dict] = None,
) -> bool:
    """Validate that 4 points form a reasonable quadrilateral (paper §4.3 SVM).

    Replaces the paper's SVM geometric-relationship check with direct
    geometric constraints:
    - Minimum area (reject degenerate configurations).
    - Aspect ratio (reject overly elongated shapes).
    - Interior angles (reject non-rectangular shapes).

    Args:
        points: List of 4 ``(x, y)`` points.
        min_area: Minimum contour area (px²).
        max_aspect_ratio: Maximum width/height ratio.
        min_angle_deg: Minimum interior angle (degrees).
        diagnostics: Optional dict that will be populated with ``area``,
            ``aspect_ratio``, ``min_angle`` and ``reason`` keys for debugging.

    Returns:
        True if the points form a valid quadrilateral.
    """
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics["reason"] = "ok"

    if len(points) < 4:
        if diagnostics is not None:
            diagnostics["reason"] = "too_few_points"
        return False

    pts = np.array(points[:4], dtype=np.float32)

    # Area check.
    contour = pts.reshape(-1, 1, 2)
    area = cv2.contourArea(contour)
    if diagnostics is not None:
        diagnostics["area"] = float(area)
    if area < min_area:
        if diagnostics is not None:
            diagnostics["reason"] = "area_too_small"
        return False

    # Aspect ratio check.
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    w, h = x_max - x_min, y_max - y_min
    if w < 1e-3 or h < 1e-3:
        if diagnostics is not None:
            diagnostics["reason"] = "degenerate_bbox"
        return False
    aspect = max(w, h) / (min(w, h) + 1e-3)
    if diagnostics is not None:
        diagnostics["aspect_ratio"] = float(aspect)
    if aspect > max_aspect_ratio:
        if diagnostics is not None:
            diagnostics["reason"] = "aspect_ratio_too_large"
        return False

    # Interior angle check (rectangular shapes have angles near 90 deg).
    hull = cv2.convexHull(contour, returnPoints=True)
    if len(hull) < 4:
        if diagnostics is not None:
            diagnostics["reason"] = "hull_too_small"
        return False
    hull = hull.reshape(-1, 2)
    min_angle = 180.0
    for i in range(4):
        p_prev = hull[(i - 1) % 4]
        p_curr = hull[i]
        p_next = hull[(i + 1) % 4]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        min_angle = min(min_angle, angle)
        if angle < min_angle_deg:
            if diagnostics is not None:
                diagnostics["reason"] = "interior_angle_too_small"
                diagnostics["min_angle"] = float(angle)
            return False

    if diagnostics is not None:
        diagnostics["min_angle"] = float(min_angle)
    return True
