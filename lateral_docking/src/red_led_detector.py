"""Red LED target detector for underwater scenes.

Problem characteristics
-----------------------
- Very dark background (underwater scene).
- Four bright green LEDs arranged in a rectangle.
- One weak red LED at the centre of the rectangle.
- All LED cores are saturated (white), so core colour cannot distinguish them.
- The red LED's halo is much less green than the green LEDs' halos.
- The red LED's saturated core area is significantly smaller than the
  green ones.

Algorithm
---------
1. High brightness threshold (≈150–200) to isolate only the saturated white
   cores.  This naturally separates adjacent LEDs because their centres are
   over-exposed.
2. Connected-component analysis to get each core's geometry.
3. For each blob sample a halo ring (dilated_mask − original_mask) to obtain
   colour information that survives outside the saturated core.
4. Compute "greenness" = G − R in the halo.  Green LEDs score ≈ +130~+160,
   the red target scores ≈ +40~+70.
5. Detect all targets using a dual strategy:
   - Median strategy (dense arrays): global median greenness separates red
     from green when at least four LEDs are visible.
   - Fallback (sparse scenes): absolute greenness threshold catches isolated
     red targets that lack neighbouring LEDs.
6. A maximum of two targets are returned per view (original + reflection).

Example:
    import cv2
    from red_led_detector import detect_red_led

    image = cv2.imread("test.png")
    detected, positions = detect_red_led(image)
    if detected:
        print(f"Found {len(positions)} red LED(s) at {positions}")
    else:
        print("No red LED detected.")
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
HIGH_THRESH = 150           # Threshold for saturated LED cores (0-255)
                            # 150 works for weaker video frames; static images
                            # still work because their target greenness is far
                            # below the median even at this lower threshold.
AREA_MIN = 5                # Ignore salt-and-pepper noise
AREA_MAX = 2000             # Upper bound for target blob area
HALO_DILATE = 7             # Pixels to expand for halo colour sampling
GREENNESS_CUTOFF = 100.0    # Above this we consider a blob clearly green
AREA_PENALTY_SCALE = 0.05   # Extra score per pixel above 50 (soft penalty)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class TargetPointDetector:
    """Detects the red target point in an underwater LED array image."""

    def __init__(
        self,
        high_thresh: int = HIGH_THRESH,
        area_min: int = AREA_MIN,
        area_max: int = AREA_MAX,
        halo_dilate: int = HALO_DILATE,
        greenness_cutoff: float = GREENNESS_CUTOFF,
        area_penalty_scale: float = AREA_PENALTY_SCALE,
    ):
        self.high_thresh = high_thresh
        self.area_min = area_min
        self.area_max = area_max
        self.halo_dilate = halo_dilate
        self.greenness_cutoff = greenness_cutoff
        self.area_penalty_scale = area_penalty_scale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_cores(
        self, gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Isolate saturated LED cores with a high global threshold.

        Args:
            gray: Input grayscale image.

        Returns:
            A tuple containing:
                - num_labels: Number of connected components.
                - labels: Label map for each pixel.
                - stats: Statistics for each component (x, y, w, h, area).
                - centroids: Centroid coordinates for each component.
        """
        # For this scene the background is uniformly dark and the LED cores
        # are saturated; a fixed high threshold is more stable than Otsu.
        _, mask = cv2.threshold(
            gray, self.high_thresh, 255, cv2.THRESH_BINARY
        )

        # Light opening removes thin bridges if two cores nearly touch.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(mask, connectivity=8)
        )
        return num_labels, labels, stats, centroids

    def _compute_halo_features(
        self,
        img_rgb: np.ndarray,
        labels: np.ndarray,
        blob_id: int,
    ) -> Dict[str, float]:
        """Compute halo colour features for a blob.

        The halo is the ring obtained by dilating the blob and subtracting the
        original mask.  This avoids the saturated white core where chromatic
        information is destroyed.
        """
        component_mask = (labels == blob_id).astype(np.uint8) * 255

        halo_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.halo_dilate * 2 + 1, self.halo_dilate * 2 + 1),
        )
        dilated = cv2.dilate(component_mask, halo_kernel, iterations=1)
        halo_mask = cv2.subtract(dilated, component_mask)

        # If the blob is so small that the halo ring is empty, fall back to
        # the blob itself.  In practice the area cue dominates for such tiny
        # blobs.
        if cv2.countNonZero(halo_mask) < 3:
            halo_mask = component_mask

        mean_rgb = cv2.mean(img_rgb, mask=halo_mask)[:3]
        r, g, b = mean_rgb

        # Greenness = G - R  (positive -> green-dominant, lower -> less green)
        greenness = float(g) - float(r)

        return {
            "mean_rgb": (float(r), float(g), float(b)),
            "greenness": greenness,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """Run detection and return the best target blob, or None.

        Args:
            image: Input BGR image.

        Returns:
            Dictionary with target information, or None if no valid target
            is found.

        Raises:
            ValueError: If the input image is empty or None.
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image provided.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mild blur suppresses sensor noise without destroying tiny cores.
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        num_labels, labels, stats, centroids = self._extract_cores(gray)

        candidates: List[Dict] = []
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area < self.area_min:
                continue

            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            feats = self._compute_halo_features(rgb, labels, i)

            # Soft area penalty: large blobs get a small score boost.
            # The penalty is gentle so that a genuinely non-green large blob
            # would still outrank a small green one, but for this scene the
            # green LEDs are both large AND strongly green, so they are
            # penalised twice and sink to the bottom of the ranking.
            size_penalty = max(0, area - 50) * self.area_penalty_scale
            score = feats["greenness"] + size_penalty

            candidates.append(
                {
                    "id": i,
                    "bbox": (int(x), int(y), int(bw), int(bh)),
                    "center": (cx, cy),
                    "area": int(area),
                    **feats,
                    "score": float(score),
                }
            )

        if not candidates:
            return None

        # --- Safety filter ------------------------------------------------
        # Blobs that are BOTH huge AND strongly green are obvious green LEDs.
        # We keep them in the pool (the score will eliminate them), but we
        # also use them to sanity-check the final decision.
        obvious_green = [
            c
            for c in candidates
            if c["area"] > 500 and c["greenness"] > self.greenness_cutoff
        ]

        # The target should be the blob with the lowest composite score.
        target = min(candidates, key=lambda c: c["score"])

        # Extra safety: if the best candidate is still greener than the
        # obvious green blobs, something is wrong (no red target present).
        if obvious_green:
            min_obvious_greenness = min(
                g["greenness"] for g in obvious_green
            )
            if target["greenness"] > min_obvious_greenness - 20:
                # The "best" candidate is almost as green as a known green
                # LED.
                return None

        # Final hard guard: if the selected blob is extremely green on its
        # own, refuse to annotate it.
        if target["greenness"] > self.greenness_cutoff + 30:
            return None

        return target

    def detect_all(self, image: np.ndarray) -> List[Dict]:
        """Detect all red target points in the image.

        Two strategies are used:
        1. Median strategy (dense arrays): when at least four blobs are
           present, the global median greenness separates red targets from
           green LEDs.  Targets must also have two neighbours within 150 px.
        2. Fallback (sparse scenes): if the median strategy yields nothing,
           blobs whose greenness is below 70 are returned directly.  This
           handles isolated red targets that are far from neighbouring LEDs.

        A maximum of two targets are returned per view (original +
        reflection).

        Returns:
            A list of target dictionaries, one per detected red target.
            Maximum length is 2.
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image provided.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        num_labels, labels, stats, centroids = self._extract_cores(gray)

        candidates: List[Dict] = []
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area < self.area_min:
                continue

            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            feats = self._compute_halo_features(rgb, labels, i)

            candidates.append(
                {
                    "id": i,
                    "bbox": (int(x), int(y), int(bw), int(bh)),
                    "center": (cx, cy),
                    "area": int(area),
                    **feats,
                }
            )

        validated: List[Dict] = []

        # --- Median strategy (dense LED arrays) ---------------------------
        # When at least four blobs are present we can use the global median
        # greenness as a reference.  This works well for standard LED arrays.
        if len(candidates) >= 4:
            median_greenness = float(
                np.median([c["greenness"] for c in candidates])
            )

            # Targets are blobs whose halo is substantially less green than
            # the median.  The -20 offset was determined empirically: it
            # comfortably separates green LEDs (+130~+160) from red targets
            # (+40~+80) while tolerating moderate halo contamination.
            targets = [
                c
                for c in candidates
                if c["greenness"] < median_greenness - 20
                and c["area"] < self.area_max
            ]

            # Inter-target consistency filter
            if len(targets) > 1:
                min_greenness = min(t["greenness"] for t in targets)
                targets = [
                    t for t in targets
                    if t["greenness"] < min_greenness + 40
                ]

            # Geometric neighbourhood validation
            for t in targets:
                tx, ty = t["center"]
                neighbour_count = 0
                for c in candidates:
                    if c["id"] == t["id"]:
                        continue
                    cx, cy = c["center"]
                    if (tx - cx) ** 2 + (ty - cy) ** 2 < 150 ** 2:
                        neighbour_count += 1
                if neighbour_count >= 2:
                    validated.append(t)

        # --- Fallback for sparse scenes -----------------------------------
        # In scenes with fewer LEDs or an isolated red target that is far
        # from its neighbours, the median strategy may fail.  We fall back
        # to an absolute greenness threshold to catch obviously red blobs.
        # The threshold (70) sits between typical red targets (+40~+70) and
        # weak green glares (+80~+110) while staying well below green LEDs
        # (+130~+160).
        if not validated:
            red_candidates = [
                c for c in candidates
                if c["greenness"] < 70 and c["area"] < self.area_max
            ]
            if red_candidates:
                red_candidates = sorted(
                    red_candidates, key=lambda c: c["greenness"]
                )[:2]
                validated = red_candidates

        # At most two red targets per view (original + reflection).
        if len(validated) > 2:
            validated = sorted(
                validated, key=lambda t: t["greenness"]
            )[:2]

        return validated

    def visualize_all(
        self,
        image: np.ndarray,
        targets: List[Dict],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Draw annotations for multiple target points on the input image.

        Args:
            image: Input BGR image.
            targets: List of target dictionaries from :meth:`detect_all`.
            output_path: Optional path to save the annotated image.

        Returns:
            Annotated image with bounding boxes and labels.
        """
        vis = image.copy()
        h, w = vis.shape[:2]

        # Track used text positions to avoid overlapping labels.
        used_rects: List[Tuple[int, int, int, int]] = []

        for idx, target in enumerate(targets):
            x, y, bw, bh = target["bbox"]

            padding = max(6, int(max(bw, bh) * 0.6))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w - 1, x + bw + padding)
            y2 = min(h - 1, y + bh + padding)

            # Red rectangle
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Label with index if multiple targets exist
            if len(targets) == 1:
                text = "Target Point"
            else:
                text = f"Target Point {idx + 1}"
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)

            # Default position: above the box
            tx = x1
            ty = y1 - 6

            # If overlap with previous labels, shift downwards
            for _ in range(10):  # safety cap
                overlap = False
                for rx, ry, rw, rh in used_rects:
                    if not (tx + tw < rx or tx > rx + rw or
                            ty < ry or ty - th > ry + rh):
                        overlap = True
                        break
                if not overlap:
                    break
                ty += th + 4

            # Clamp to image bounds
            ty = max(th + 2, min(h - 2, ty))
            tx = max(2, min(w - tw - 2, tx))

            cv2.putText(
                vis,
                text,
                (tx, ty),
                FONT,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
            )
            used_rects.append((tx, ty - th, tw, th))

            # Centre marker
            cv2.drawMarker(
                vis,
                target["center"],
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=8,
                thickness=1,
            )

        if output_path:
            cv2.imwrite(output_path, vis)

        return vis

    def visualize(
        self,
        image: np.ndarray,
        target: Dict,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Draw red bounding box and 'Target Point' label on the input image.

        Args:
            image: Input BGR image.
            target: Target dictionary from :meth:`detect`.
            output_path: Optional path to save the annotated image.

        Returns:
            Annotated image with bounding box and label.
        """
        vis = image.copy()
        h, w = vis.shape[:2]

        x, y, bw, bh = target["bbox"]

        # Add padding so the box is clearly visible.
        padding = max(8, int(max(bw, bh) * 0.8))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w - 1, x + bw + padding)
        y2 = min(h - 1, y + bh + padding)

        # Red rectangle in BGR
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Label text
        text = "Target Point"
        font_scale = 0.7
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)

        # Place text above the box if space permits, otherwise below.
        tx = x1
        ty = y1 - 10 if y1 - 10 > th else y2 + th + 10

        # Clamp to image bounds
        ty = max(th + 4, min(h - 4, ty))
        tx = max(4, min(w - tw - 4, tx))

        cv2.putText(
            vis,
            text,
            (tx, ty),
            FONT,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )

        # Centre cross-hair for quick visual verification
        cv2.drawMarker(
            vis,
            target["center"],
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )

        if output_path:
            cv2.imwrite(output_path, vis)

        return vis


def detect_red_led(
    image: np.ndarray,
) -> Tuple[bool, List[Tuple[int, int]]]:
    """Detect red LED target points in the input image.

    This is a thin convenience wrapper around :class:`TargetPointDetector`.

    Args:
        image: Input BGR image from cv2.imread().

    Returns:
        Tuple containing:
            - detected: True if at least one red LED is found, else False.
            - positions: List of (x, y) pixel coordinates for each detected
              red LED. Empty list if none are found.
    """
    if image is None or image.size == 0:
        return False, []

    detector = TargetPointDetector()
    targets = detector.detect_all(image)

    if not targets:
        return False, []

    positions = [target["center"] for target in targets]
    return True, positions
