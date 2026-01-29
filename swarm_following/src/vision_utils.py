import numpy as np
import cv2

import config as cfg


def find_black_center(roi: np.ndarray):
    lower_black = np.array(cfg.BLACK_LOWER, dtype=np.uint8)
    upper_black = np.array(cfg.BLACK_UPPER, dtype=np.uint8)
    mask = cv2.inRange(roi, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    image_center = np.array([roi.shape[1] // 2, roi.shape[0] // 2])
    min_distance = np.inf
    nearest = None
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        c_x = int(moments["m10"] / moments["m00"])
        c_y = int(moments["m01"] / moments["m00"])
        dist = np.linalg.norm(np.array([c_x, c_y]) - image_center)
        if dist < min_distance:
            min_distance = dist
            nearest = (c_x, c_y)
    return nearest


def classify_in_roi(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    label, area = "none", 0
    if x2 - x1 < cfg.MIN_ROI_SIZE or y2 - y1 < cfg.MIN_ROI_SIZE:
        return label, area
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask_red_1 = cv2.inRange(hsv, np.array(cfg.RED_LOWER_1, dtype=np.uint8), np.array(cfg.RED_UPPER_1, dtype=np.uint8))
    mask_red_2 = cv2.inRange(hsv, np.array(cfg.RED_LOWER_2, dtype=np.uint8), np.array(cfg.RED_UPPER_2, dtype=np.uint8))
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)
    mask_yellow = cv2.inRange(hsv, np.array(cfg.YELLOW_LOWER, dtype=np.uint8), np.array(cfg.YELLOW_UPPER, dtype=np.uint8))

    red_area = int(cv2.countNonZero(mask_red))
    yellow_area = int(cv2.countNonZero(mask_yellow))

    if yellow_area > cfg.MIN_COLOR_AREA or red_area > cfg.MIN_COLOR_AREA:
        if yellow_area > red_area:
            label = "yellow"
            mask = mask_yellow
        else:
            label = "red"
            mask = mask_red
        area = max(yellow_area, red_area)
        if cfg.SHOW_MASK:
            mask_bool = mask > 0
            if np.any(mask_bool):
                color = np.array([0, 255, 0], dtype=np.float32)
                roi_f = roi.astype(np.float32)
                blended = roi_f.copy()
                blended[mask_bool] = roi_f[mask_bool] * (1.0 - cfg.MASK_ALPHA) + color * cfg.MASK_ALPHA
                frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return label, area
