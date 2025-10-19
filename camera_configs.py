# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Calibration data (2024-12-27, Camera 7)
# --------------------------------------------------------------------------- #
LEFT_CAMERA_MATRIX = np.array([
    [446.1312, -0.1607, 360.5116],
    [0, 446.5769, 308.2938],
    [0, 0, 1.0000]
])

RIGHT_CAMERA_MATRIX = np.array([
    [445.7741, 0.1990, 342.0431],
    [0, 446.7158, 283.7390],
    [0, 0, 1.0000]
])

LEFT_DISTORTION = np.array([-0.229800, 0.140721, 0.001900, -0.000648, -0.029280])
RIGHT_DISTORTION = np.array([-0.220908, 0.111177, 0.000023, -0.000142, -0.009207])

R = np.array([
    [1.0000, -0.0044, -0.0053],
    [0.0044, 0.9999, 0.0098],
    [0.0052, -0.0099, 0.9999]
])

T = np.array([-95.7117, -0.3257, -0.4032])

SIZE = (640, 480)

# --------------------------------------------------------------------------- #
# Rectification maps
# --------------------------------------------------------------------------- #
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    LEFT_CAMERA_MATRIX, LEFT_DISTORTION,
    RIGHT_CAMERA_MATRIX, RIGHT_DISTORTION,
    SIZE, R, T, alpha=0
)

LEFT_MAP1, LEFT_MAP2 = cv2.initUndistortRectifyMap(
    LEFT_CAMERA_MATRIX, LEFT_DISTORTION, R1, P1, SIZE, cv2.CV_16SC2
)
RIGHT_MAP1, RIGHT_MAP2 = cv2.initUndistortRectifyMap(
    RIGHT_CAMERA_MATRIX, RIGHT_DISTORTION, R2, P2, SIZE, cv2.CV_16SC2
)