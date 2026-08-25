"""Threshold-based detector for the NWPU/Xianxia underwater LED target.

Target appearance
-----------------
- A saturated white/bright core.
- A surrounding green halo caused by the underwater LED and optics.
- Very dark background.

Algorithm
---------
1. Convert BGR to HSV.
2. Build a white-core mask from high Value pixels.
3. Build a green-halo mask from Hue/Saturation/Value ranges.
4. Combine masks, apply light morphology, run connected-components.
5. Filter blobs by area, aspect ratio, and circularity.
6. Return bounding boxes and centroids in both absolute and YOLO-normalized
   formats.

The module can be run standalone on the NWPU dataset, a single image, or the
raw video file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tunable defaults (conservative starting points)
# ---------------------------------------------------------------------------
DEFAULT_BRIGHTNESS_THRESH = 200       # HSV Value for white core
DEFAULT_HALO_HUE_RANGE = (35, 95)     # OpenCV Hue range 0-179
DEFAULT_HALO_SAT_THRESH = 60          # Min saturation for green halo
DEFAULT_HALO_VAL_THRESH = 60          # Min value for green halo
DEFAULT_MIN_AREA = 50
DEFAULT_MAX_AREA = 5000
DEFAULT_MIN_CIRCULARITY = 0.0
DEFAULT_MAX_ASPECT_RATIO = 5.0
DEFAULT_MORPH_KERNEL = 3
DEFAULT_CORE_SHRINK_FACTOR = 0.62


@dataclass
class Detection:
    """A single threshold-based detection."""

    class_id: int
    bbox: Tuple[int, int, int, int]       # x1, y1, x2, y2 in pixel coords
    yolo_bbox: Tuple[float, float, float, float]  # x_c, y_c, w, h normalized
    center: Tuple[int, int]
    area: float
    circularity: float
    is_core: bool
    is_halo: bool


@dataclass
class DetectionResult:
    """Result of running the detector on one frame."""

    detections: List[Detection] = field(default_factory=list)
    core_mask: Optional[np.ndarray] = None
    halo_mask: Optional[np.ndarray] = None
    combined_mask: Optional[np.ndarray] = None


class NwpuThresholdDetector:
    """Detects bright white-core + green-halo targets using thresholding.

    Args:
        brightness_thresh: HSV Value threshold for the saturated white core.
        halo_hue_range: (min, max) Hue range for the green halo (0-179).
        halo_sat_thresh: Minimum Saturation for halo pixels.
        halo_val_thresh: Minimum Value for halo pixels.
        min_area: Minimum blob area in pixels.
        max_area: Maximum blob area in pixels.
        min_circularity: Minimum circularity (0-1) for a valid blob.
        max_aspect_ratio: Maximum width/height aspect ratio.
        morph_kernel_size: Size of the elliptical morphological opening kernel.
        core_shrink_factor: Scale factor applied to the core bbox around its
            center (0 < factor <= 1).  Values < 1 tighten the box to compensate
            for blooming around the saturated white core.
        debug: Whether to print diagnostic messages.
    """

    def __init__(
        self,
        brightness_thresh: int = DEFAULT_BRIGHTNESS_THRESH,
        halo_hue_range: Tuple[int, int] = DEFAULT_HALO_HUE_RANGE,
        halo_sat_thresh: int = DEFAULT_HALO_SAT_THRESH,
        halo_val_thresh: int = DEFAULT_HALO_VAL_THRESH,
        min_area: int = DEFAULT_MIN_AREA,
        max_area: int = DEFAULT_MAX_AREA,
        min_circularity: float = DEFAULT_MIN_CIRCULARITY,
        max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO,
        morph_kernel_size: int = DEFAULT_MORPH_KERNEL,
        core_shrink_factor: float = DEFAULT_CORE_SHRINK_FACTOR,
        top_k: int = 0,
        core_dominant: bool = True,
        halo_overlap_ratio_thresh: float = 0.0,
        debug: bool = False,
    ) -> None:
        self.brightness_thresh = brightness_thresh
        self.halo_hue_range = halo_hue_range
        self.halo_sat_thresh = halo_sat_thresh
        self.halo_val_thresh = halo_val_thresh
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.max_aspect_ratio = max_aspect_ratio
        self.morph_kernel_size = max(1, morph_kernel_size)
        self.core_shrink_factor = float(core_shrink_factor)
        self.top_k = top_k
        self.core_dominant = core_dominant
        self.halo_overlap_ratio_thresh = halo_overlap_ratio_thresh
        self.debug = debug

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run threshold detection on a BGR image.

        Args:
            image: Input BGR image.

        Returns:
            ``DetectionResult`` with detections and intermediate masks.
        """
        result = DetectionResult()
        if image is None or image.size == 0:
            return result

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]

        core_mask = self._build_core_mask(hsv)
        halo_mask = self._build_halo_mask(hsv)

        if self.core_dominant:
            detections = self._detect_core_dominant(
                core_mask, halo_mask, h, w
            )
        else:
            detections = self._detect_combined(
                core_mask, halo_mask, h, w
            )

        if self.top_k > 0 and len(detections) > self.top_k:
            detections = sorted(
                detections, key=lambda d: d.area, reverse=True
            )[: self.top_k]

        if self.debug:
            print(f"Detected {len(detections)} blob(s)")

        result.detections = detections
        result.core_mask = core_mask
        result.halo_mask = halo_mask
        result.combined_mask = cv2.bitwise_or(core_mask, halo_mask)
        return result

    def _detect_core_dominant(
        self,
        core_mask: np.ndarray,
        halo_mask: np.ndarray,
        h: int,
        w: int,
    ) -> List[Detection]:
        """Detect targets using the bright core as the primary cue.

        A valid target must have a bright core blob that is surrounded by (or
        overlaps) green halo pixels.  The output bounding box is computed from
        the original (non-closed) core pixels inside each closed component, so
        the box stays tight around the true white core rather than the closure
        dilation.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        # Closing merges fragmented bright-core pixels into a single blob.
        closed_core = cv2.morphologyEx(core_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            closed_core, connectivity=8
        )

        detections: List[Detection] = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            bx = int(stats[label_id, cv2.CC_STAT_LEFT])
            by = int(stats[label_id, cv2.CC_STAT_TOP])
            bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            if not (self.min_area <= area <= self.max_area):
                continue

            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect > self.max_aspect_ratio:
                continue

            closed_mask = (labels == label_id).astype(np.uint8) * 255
            # Tight box from the original bright pixels inside the closed region.
            original_core_in_region = cv2.bitwise_and(core_mask, closed_mask)
            if cv2.countNonZero(original_core_in_region) == 0:
                continue

            ys, xs = np.where(original_core_in_region > 0)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1

            # Tighten the box around the saturated core to compensate for
            # optical blooming/scattering that expands the bright region.
            if self.core_shrink_factor < 1.0:
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                bw, bh = (x2 - x1) * self.core_shrink_factor, (y2 - y1) * self.core_shrink_factor
                x1 = int(round(cx - bw * 0.5))
                y1 = int(round(cy - bh * 0.5))
                x2 = int(round(cx + bw * 0.5))
                y2 = int(round(cy + bh * 0.5))

            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            component_mask = original_core_in_region
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            circularity = 0.0
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            # Require green halo overlap to suppress bright specks.
            halo_overlap = cv2.countNonZero(
                cv2.bitwise_and(closed_mask, halo_mask)
            )
            halo_overlap_ratio = halo_overlap / (area + 1e-6)
            if halo_overlap_ratio < self.halo_overlap_ratio_thresh:
                continue

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            yolo_xc = (x1 + x2) / 2.0 / w
            yolo_yc = (y1 + y2) / 2.0 / h
            yolo_w = (x2 - x1) / w
            yolo_h = (y2 - y1) / h

            detections.append(
                Detection(
                    class_id=1,
                    bbox=(x1, y1, x2, y2),
                    yolo_bbox=(yolo_xc, yolo_yc, yolo_w, yolo_h),
                    center=(cx, cy),
                    area=float(area),
                    circularity=float(circularity),
                    is_core=True,
                    is_halo=halo_overlap > 0,
                )
            )

        return detections

    def _detect_combined(
        self,
        core_mask: np.ndarray,
        halo_mask: np.ndarray,
        h: int,
        w: int,
    ) -> List[Detection]:
        """Detect targets from the union of core and halo masks (legacy mode)."""
        combined = cv2.bitwise_or(core_mask, halo_mask)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined, connectivity=8
        )

        detections: List[Detection] = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            bx = int(stats[label_id, cv2.CC_STAT_LEFT])
            by = int(stats[label_id, cv2.CC_STAT_TOP])
            bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[label_id]

            if not (self.min_area <= area <= self.max_area):
                continue

            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect > self.max_aspect_ratio:
                continue

            component_mask = (labels == label_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            circularity = 0.0
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(w, bx + bw), min(h, by + bh)
            yolo_xc = (x1 + x2) / 2.0 / w
            yolo_yc = (y1 + y2) / 2.0 / h
            yolo_w = (x2 - x1) / w
            yolo_h = (y2 - y1) / h

            core_overlap = cv2.countNonZero(
                cv2.bitwise_and(component_mask, core_mask)
            )
            halo_overlap = cv2.countNonZero(
                cv2.bitwise_and(component_mask, halo_mask)
            )

            detections.append(
                Detection(
                    class_id=1,
                    bbox=(x1, y1, x2, y2),
                    yolo_bbox=(yolo_xc, yolo_yc, yolo_w, yolo_h),
                    center=(int(round(cx)), int(round(cy))),
                    area=float(area),
                    circularity=float(circularity),
                    is_core=core_overlap > 0,
                    is_halo=halo_overlap > 0,
                )
            )

        return detections

    def _build_core_mask(self, hsv: np.ndarray) -> np.ndarray:
        """High-value mask for the saturated white core."""
        _, _, v = cv2.split(hsv)
        _, mask = cv2.threshold(v, self.brightness_thresh, 255, cv2.THRESH_BINARY)
        return mask

    def _build_halo_mask(self, hsv: np.ndarray) -> np.ndarray:
        """HSV range mask for the green halo."""
        h_min, h_max = self.halo_hue_range
        lower = np.array([h_min, self.halo_sat_thresh, self.halo_val_thresh])
        upper = np.array([h_max, 255, 255])
        return cv2.inRange(hsv, lower, upper)

    @staticmethod
    def visualize(
        image: np.ndarray,
        result: DetectionResult,
        show_masks: bool = False,
    ) -> np.ndarray:
        """Draw detection boxes and centroids on a copy of the image."""
        vis = image.copy()
        h, w = vis.shape[:2]

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0) if det.is_core else (0, 255, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis, det.center, 4, (0, 0, 255), -1)
            label = f"C={det.circularity:.2f} A={int(det.area)}"
            cv2.putText(
                vis,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        if show_masks and result.combined_mask is not None:
            mask_color = cv2.applyColorMap(result.combined_mask, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)

        cv2.putText(
            vis,
            f"detections: {len(result.detections)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return vis


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    """Compute intersection-over-union of two absolute-pixel bboxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / (union_area + 1e-6)


def evaluate_on_dataset(
    image_dir: Path,
    label_dir: Path,
    detector: NwpuThresholdDetector,
    iou_thresh: float = 0.5,
    max_images: int = 0,
) -> dict:
    """Evaluate detector against YOLO ground-truth boxes.

    Returns:
        Dict with TP, FP, FN, precision, recall, F1, and mean-matched-IoU.
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    image_paths.sort(key=_natural_sort_key)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious: List[float] = []

    for idx, img_path in enumerate(image_paths):
        if 0 < max_images <= idx:
            break

        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        gt_boxes = _read_yolo_label(label_path, w, h)
        if not gt_boxes:
            continue

        result = detector.detect(image)
        pred_boxes = [d.bbox for d in result.detections]

        matched_pred = [False] * len(pred_boxes)
        for gt_box in gt_boxes:
            best_iou = 0.0
            best_pred_idx = -1
            for p_idx, pred_box in enumerate(pred_boxes):
                if matched_pred[p_idx]:
                    continue
                iou = _iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = p_idx

            if best_iou >= iou_thresh and best_pred_idx >= 0:
                total_tp += 1
                matched_pred[best_pred_idx] = True
                matched_ious.append(best_iou)
            else:
                total_fn += 1

        total_fp += sum(1 for m in matched_pred if not m)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "images_evaluated": min(len(image_paths), max_images or len(image_paths)),
    }


# ---------------------------------------------------------------------------
# Dataset-driven parameter estimation
# ---------------------------------------------------------------------------
def _read_yolo_label(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, int, int, int]]:
    """Read YOLO-format labels and return absolute pixel bboxes."""
    boxes: List[Tuple[int, int, int, int]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = parts
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
        x1 = int((xc - bw / 2) * img_w)
        y1 = int((yc - bh / 2) * img_h)
        x2 = int((xc + bw / 2) * img_w)
        y2 = int((yc + bh / 2) * img_h)
        boxes.append((x1, y1, x2, y2))
    return boxes


def estimate_params_from_dataset(
    image_dir: Path,
    label_dir: Path,
    brightness_percentile: float = 90.0,
    area_margin_ratio: float = 0.5,
) -> dict:
    """Estimate detector parameters from annotated ground-truth boxes.

    Args:
        image_dir: Directory containing JPG/PNG images.
        label_dir: Directory containing YOLO .txt labels.
        brightness_percentile: Percentile of V-channel values inside boxes used
            to set the core brightness threshold.
        area_margin_ratio: Fractional margin around the annotated area range.

    Returns:
        A dict of suggested constructor kwargs.
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    core_v_values: List[float] = []
    halo_h_values: List[float] = []
    halo_s_values: List[float] = []
    halo_v_values: List[float] = []
    areas: List[float] = []

    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for img_path in image_paths:
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)
        b, g, r = cv2.split(image)

        boxes = _read_yolo_label(label_path, w, h)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            roi_v = v_chan[y1:y2, x1:x2]
            roi_h = h_chan[y1:y2, x1:x2]
            roi_s = s_chan[y1:y2, x1:x2]
            roi_b = b[y1:y2, x1:x2]
            roi_g = g[y1:y2, x1:x2]
            roi_r = r[y1:y2, x1:x2]

            core_v_values.extend(roi_v.reshape(-1).tolist())
            areas.append(float((x2 - x1) * (y2 - y1)))

            # Identify likely halo pixels: greenish and not too dark/gray.
            greenness = roi_g.astype(np.float32) - np.maximum(roi_r, roi_b).astype(np.float32)
            halo_mask = (greenness > 10) & (roi_s > 20) & (roi_v > 20)
            if np.count_nonzero(halo_mask) < 10:
                continue
            halo_h_values.extend(roi_h[halo_mask].tolist())
            halo_s_values.extend(roi_s[halo_mask].tolist())
            halo_v_values.extend(roi_v[halo_mask].tolist())

    if not core_v_values or not areas:
        raise ValueError("No valid annotations found for parameter estimation.")

    brightness_thresh = int(np.percentile(core_v_values, brightness_percentile))
    brightness_thresh = int(np.clip(brightness_thresh, 150, 250))

    if halo_h_values:
        hue_min = max(0, int(np.percentile(halo_h_values, 2.5)))
        hue_max = min(179, int(np.percentile(halo_h_values, 97.5)))
        # Keep saturation/value thresholds conservative.  The annotated boxes
        # contain both bright core and dim halo edge; using low percentiles
        # here would drop the outer halo and fragment the target.
        sat_thresh = DEFAULT_HALO_SAT_THRESH
        val_thresh = DEFAULT_HALO_VAL_THRESH
    else:
        hue_min, hue_max = DEFAULT_HALO_HUE_RANGE
        sat_thresh = DEFAULT_HALO_SAT_THRESH
        val_thresh = DEFAULT_HALO_VAL_THRESH

    areas = np.array(areas)
    # Use the 25th percentile of annotated box areas as the lower bound.  The
    # threshold detector works on the closed bright core, whose area is usually
    # comparable to the annotated box; the 10th percentile used previously was
    # too permissive and admitted many small bright specks.
    min_area = max(100, int(np.percentile(areas, 25)))
    max_area = int(np.percentile(areas, 95) * (1 + area_margin_ratio))

    return {
        "brightness_thresh": brightness_thresh,
        "halo_hue_range": (hue_min, hue_max),
        "halo_sat_thresh": sat_thresh,
        "halo_val_thresh": val_thresh,
        "min_area": min_area,
        "max_area": max_area,
    }


def tune_parameters(
    image_dir: Path,
    label_dir: Path,
    max_images: int = 100,
    iou_thresh: float = 0.5,
) -> Tuple[dict, dict]:
    """Grid-search detector parameters using annotations as ground truth.

    Args:
        image_dir: Directory containing dataset images.
        label_dir: Directory containing YOLO labels.
        max_images: Number of images to use for tuning (0 = all).
        iou_thresh: IoU threshold for a true positive.

    Returns:
        Tuple of (best_params_dict, best_metrics_dict).
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    base = estimate_params_from_dataset(image_dir, label_dir)
    estimated_hue_min, estimated_hue_max = base["halo_hue_range"]

    # Search grids centred on the estimated values.
    grid = {
        "brightness_thresh": sorted(
            set(
                [
                    max(150, base["brightness_thresh"] - 20),
                    base["brightness_thresh"],
                    min(250, base["brightness_thresh"] + 10),
                ]
            )
        ),
        "halo_hue_min": sorted(
            set(
                [
                    max(0, estimated_hue_min - 10),
                    estimated_hue_min,
                    max(0, estimated_hue_min + 5),
                ]
            )
        ),
        "halo_hue_max": sorted(
            set(
                [
                    min(179, estimated_hue_max - 5),
                    estimated_hue_max,
                    min(179, estimated_hue_max + 10),
                ]
            )
        ),
        "halo_sat_thresh": [30, 50, 70, 90],
        "halo_val_thresh": [30, 50, 70, 90],
        "min_area": sorted(
            set(
                [
                    max(100, int(base["min_area"] * 0.5)),
                    base["min_area"],
                    int(base["min_area"] * 1.5),
                    int(base["min_area"] * 2.0),
                ]
            )
        ),
        "morph_kernel_size": [3, 5],
        "core_shrink_factor": [0.55, 0.62, 0.70, 0.80],
        "min_circularity": [0.0, 0.15, 0.25],
    }

    best_params: Optional[dict] = None
    best_metrics: Optional[dict] = None
    best_score = -1.0

    total = (
        len(grid["brightness_thresh"])
        * len(grid["halo_hue_min"])
        * len(grid["halo_hue_max"])
        * len(grid["halo_sat_thresh"])
        * len(grid["halo_val_thresh"])
        * len(grid["min_area"])
        * len(grid["morph_kernel_size"])
        * len(grid["core_shrink_factor"])
        * len(grid["min_circularity"])
    )
    print(f"Tuning: {total} parameter combinations on up to {max_images} images...")

    for brightness in grid["brightness_thresh"]:
        for h_min in grid["halo_hue_min"]:
            for h_max in grid["halo_hue_max"]:
                if h_min >= h_max:
                    continue
                for sat in grid["halo_sat_thresh"]:
                    for val in grid["halo_val_thresh"]:
                        for min_area in grid["min_area"]:
                            for morph in grid["morph_kernel_size"]:
                                for shrink in grid["core_shrink_factor"]:
                                    for circ in grid["min_circularity"]:
                                        params = {
                                            "brightness_thresh": brightness,
                                            "halo_hue_range": (h_min, h_max),
                                            "halo_sat_thresh": sat,
                                            "halo_val_thresh": val,
                                            "min_area": min_area,
                                            "max_area": base["max_area"],
                                            "min_circularity": circ,
                                            "max_aspect_ratio": DEFAULT_MAX_ASPECT_RATIO,
                                            "morph_kernel_size": morph,
                                            "core_shrink_factor": shrink,
                                            "top_k": 0,
                                            "debug": False,
                                        }
                                        detector = NwpuThresholdDetector(**params)
                                        metrics = evaluate_on_dataset(
                                            image_dir,
                                            label_dir,
                                            detector,
                                            iou_thresh=iou_thresh,
                                            max_images=max_images,
                                        )
                                        score = metrics["f1"]
                                        if score > best_score:
                                            best_score = score
                                            best_params = params
                                            best_metrics = metrics
                                            print(
                                                f"New best F1={score:.4f} "
                                                f"P={metrics['precision']:.3f} "
                                                f"R={metrics['recall']:.3f} "
                                                f"IoU={metrics['mean_iou']:.3f} "
                                                f"params={params}"
                                            )

    if best_params is None or best_metrics is None:
        raise RuntimeError("No valid parameter combination found.")

    return best_params, best_metrics


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
def _natural_sort_key(p: Path) -> List:
    import re

    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"([0-9]+)", p.name)
    ]


def run_on_images(
    image_dir: Path,
    detector: NwpuThresholdDetector,
    output_dir: Optional[Path],
    wait_ms: int,
    max_frames: int = 0,
) -> None:
    """Process all images in a directory."""
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    image_paths.sort(key=_natural_sort_key)

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    for idx, img_path in enumerate(image_paths):
        if 0 < max_frames <= idx:
            break

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = detector.detect(image)
        vis = detector.visualize(image, result, show_masks=False)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / (img_path.stem + "_det.jpg")
            cv2.imwrite(str(out_path), vis)
            print(f"Saved {out_path} ({len(result.detections)} detections)")
        else:
            cv2.imshow("NWPU Threshold Detector", vis)
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    if output_dir is None:
        cv2.destroyAllWindows()


def run_on_video(
    video_path: Path,
    detector: NwpuThresholdDetector,
    output_dir: Optional[Path],
    wait_ms: int,
    max_frames: int = 0,
) -> None:
    """Process a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Unable to open video: {video_path}")
        return

    writer: Optional[cv2.VideoWriter] = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = output_dir / (video_path.stem + "_det.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_id = 0
    while True:
        if 0 < max_frames <= frame_id:
            break
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        vis = detector.visualize(frame, result, show_masks=False)

        if writer is not None:
            writer.write(vis)
            if frame_id % 30 == 0:
                print(f"Frame {frame_id}: {len(result.detections)} detections")
        else:
            cv2.imshow("NWPU Threshold Detector", vis)
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved video: {out_path}")
    else:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threshold-based detector for NWPU/Xianxia white-core + green-halo targets."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="Directory containing dataset images.",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default="",
        help="Directory containing YOLO labels (used for parameter estimation).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Path to a video file to process.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Path to a single image to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="If set, save output images/video instead of displaying interactively.",
    )
    parser.add_argument(
        "--brightness-thresh",
        type=int,
        default=DEFAULT_BRIGHTNESS_THRESH,
        help="HSV Value threshold for the white core.",
    )
    parser.add_argument(
        "--halo-hue-min",
        type=int,
        default=DEFAULT_HALO_HUE_RANGE[0],
        help="Minimum Hue for the green halo (0-179).",
    )
    parser.add_argument(
        "--halo-hue-max",
        type=int,
        default=DEFAULT_HALO_HUE_RANGE[1],
        help="Maximum Hue for the green halo (0-179).",
    )
    parser.add_argument(
        "--halo-sat-thresh",
        type=int,
        default=DEFAULT_HALO_SAT_THRESH,
        help="Minimum Saturation for the green halo.",
    )
    parser.add_argument(
        "--halo-val-thresh",
        type=int,
        default=DEFAULT_HALO_VAL_THRESH,
        help="Minimum Value for the green halo.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help="Minimum blob area.",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=DEFAULT_MAX_AREA,
        help="Maximum blob area.",
    )
    parser.add_argument(
        "--min-circularity",
        type=float,
        default=DEFAULT_MIN_CIRCULARITY,
        help="Minimum blob circularity.",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=DEFAULT_MAX_ASPECT_RATIO,
        help="Maximum width/height aspect ratio.",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=DEFAULT_MORPH_KERNEL,
        help="Morphological opening kernel size.",
    )
    parser.add_argument(
        "--core-shrink-factor",
        type=float,
        default=DEFAULT_CORE_SHRINK_FACTOR,
        help="Scale the core bbox around its center (0 < factor <= 1).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Keep only the K largest detections (0 = keep all).",
    )
    parser.add_argument(
        "--core-dominant",
        action="store_true",
        default=True,
        help="Use bright core as primary detection cue (tighter boxes).",
    )
    parser.add_argument(
        "--no-core-dominant",
        action="store_false",
        dest="core_dominant",
        help="Use core+halo union as detection cue (larger boxes).",
    )
    parser.add_argument(
        "--halo-overlap-ratio-thresh",
        type=float,
        default=0.0,
        help="Min halo/core overlap ratio when --core-dominant is enabled.",
    )
    parser.add_argument(
        "--estimate-params",
        action="store_true",
        help="Override thresholds using statistics from the annotation dataset.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate detector against annotations and print metrics.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Grid-search parameters using annotations as ground truth.",
    )
    parser.add_argument(
        "--save-params",
        type=str,
        default="",
        help="If set, save estimated/used parameters to this JSON file.",
    )
    parser.add_argument(
        "--max-eval-images",
        type=int,
        default=100,
        help="Maximum images to use for evaluation/tuning.",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for a true positive.",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=1,
        help="Delay between frames in ms (0 = pause for keypress).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum images/frames to process (0 = unlimited).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print diagnostic messages.",
    )
    args = parser.parse_args()

    params = {
        "brightness_thresh": args.brightness_thresh,
        "halo_hue_range": (args.halo_hue_min, args.halo_hue_max),
        "halo_sat_thresh": args.halo_sat_thresh,
        "halo_val_thresh": args.halo_val_thresh,
        "min_area": args.min_area,
        "max_area": args.max_area,
        "min_circularity": args.min_circularity,
        "max_aspect_ratio": args.max_aspect_ratio,
        "morph_kernel_size": args.morph_kernel,
        "core_shrink_factor": args.core_shrink_factor,
        "top_k": args.top_k,
        "core_dominant": args.core_dominant,
        "halo_overlap_ratio_thresh": args.halo_overlap_ratio_thresh,
        "debug": args.debug,
    }

    if args.estimate_params:
        if not args.image_dir or not args.label_dir:
            print("--estimate-params requires both --image-dir and --label-dir")
            sys.exit(1)
        estimated = estimate_params_from_dataset(
            Path(args.image_dir), Path(args.label_dir)
        )
        params.update(estimated)
        print("Estimated parameters from annotations:")
        print(json.dumps(params, indent=2, ensure_ascii=False))

    detector = NwpuThresholdDetector(**params)

    if args.tune:
        if not args.image_dir or not args.label_dir:
            print("--tune requires both --image-dir and --label-dir")
            sys.exit(1)
        best_params, best_metrics = tune_parameters(
            Path(args.image_dir),
            Path(args.label_dir),
            max_images=args.max_eval_images,
            iou_thresh=args.iou_thresh,
        )
        print("\nBest parameters:")
        print(json.dumps(best_params, indent=2, ensure_ascii=False))
        print("\nBest metrics:")
        print(json.dumps(best_metrics, indent=2, ensure_ascii=False))
        params = best_params
        detector = NwpuThresholdDetector(**params)

    if args.evaluate:
        if not args.image_dir or not args.label_dir:
            print("--evaluate requires both --image-dir and --label-dir")
            sys.exit(1)
        metrics = evaluate_on_dataset(
            Path(args.image_dir),
            Path(args.label_dir),
            detector,
            iou_thresh=args.iou_thresh,
            max_images=args.max_eval_images,
        )
        print("\nEvaluation metrics:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.save_params:
        Path(args.save_params).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_params, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        print(f"Parameters saved to {args.save_params}")

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Unable to read image: {args.image}")
            sys.exit(1)
        result = detector.detect(image)
        vis = detector.visualize(image, result, show_masks=True)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / (Path(args.image).stem + "_det.jpg")
            cv2.imwrite(str(out_path), vis)
            print(f"Saved {out_path}")
        else:
            cv2.imshow("NWPU Threshold Detector", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif args.video:
        run_on_video(
            Path(args.video),
            detector,
            output_dir,
            args.wait,
            max_frames=args.max_frames,
        )
    elif args.image_dir:
        run_on_images(
            Path(args.image_dir),
            detector,
            output_dir,
            args.wait,
            max_frames=args.max_frames,
        )
    elif not args.evaluate and not args.tune:
        print("Specify one of --image, --video, --image-dir, --evaluate, or --tune")
        sys.exit(1)


if __name__ == "__main__":
    main()
