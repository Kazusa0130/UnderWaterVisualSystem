from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

import config as cfg


@dataclass
class StereoConfig:
    k1: np.ndarray
    d1: np.ndarray
    k2: np.ndarray
    d2: np.ndarray
    r: np.ndarray
    t: np.ndarray
    image_size: tuple


class StereoDepthEstimator:
    def __init__(self, config: StereoConfig, sgbm_params: dict) -> None:
        self.config = config
        self.sgbm_params = sgbm_params
        self._init_rectify_maps()
        self._init_matchers()

    def _init_rectify_maps(self) -> None:
        flags = cv2.CALIB_ZERO_DISPARITY
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            self.config.k1,
            self.config.d1,
            self.config.k2,
            self.config.d2,
            self.config.image_size,
            self.config.r,
            self.config.t,
            flags=flags,
            alpha=0,
        )
        self.q = q
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.config.k1,
            self.config.d1,
            r1,
            p1,
            self.config.image_size,
            cv2.CV_16SC2,
        )
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.config.k2,
            self.config.d2,
            r2,
            p2,
            self.config.image_size,
            cv2.CV_16SC2,
        )

    def _init_matchers(self) -> None:
        self.left_matcher = cv2.StereoSGBM_create(**self.sgbm_params)
        self.right_matcher = None
        self.wls_filter = None
        if hasattr(cv2, "ximgproc"):
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            self.wls_filter.setLambda(8000.0)
            self.wls_filter.setSigmaColor(1.3)
            self.wls_filter.setLRCthresh(24)
            self.wls_filter.setDepthDiscontinuityRadius(3)

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple:
        left_rectified = cv2.remap(left, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        return left_rectified, right_rectified

    def compute_disparity(self, left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        left_disp = self.left_matcher.compute(left_gray, right_gray)
        if self.wls_filter is not None and self.right_matcher is not None:
            right_disp = self.right_matcher.compute(right_gray, left_gray)
            filtered = self.wls_filter.filter(left_disp, left_gray, disparity_map_right=right_disp)
            return filtered.astype(np.float32) / 16.0
        return left_disp.astype(np.float32) / 16.0

    def project_to_3d(self, x: int, y: int, disparity: float) -> tuple:
        if disparity <= 0:
            return 0.0, 0.0, 0.0
        vec = np.array([x, y, disparity, 1.0], dtype=np.float32)
        point = self.q @ vec
        if point[3] == 0:
            return 0.0, 0.0, 0.0
        x3d = point[0] / point[3]
        y3d = point[1] / point[3]
        z3d = point[2] / point[3]
        return x3d, y3d, z3d


def load_stereo_config(yaml_path: str, image_size: tuple) -> StereoConfig:
    with Path(yaml_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    k1 = np.array(data["Left"]["CameraMatrix"]["data"], dtype=np.float32).reshape(3, 3)
    d1 = np.array(data["Left"]["distortion_coefficients"]["data"], dtype=np.float32).reshape(1, 5)
    k2 = np.array(data["Right"]["CameraMatrix"]["data"], dtype=np.float32).reshape(3, 3)
    d2 = np.array(data["Right"]["distortion_coefficients"]["data"], dtype=np.float32).reshape(1, 5)
    r = np.array(data["RotationMatrix"]["data"], dtype=np.float32).reshape(3, 3)
    t = np.array(data["TranslationVector"]["data"], dtype=np.float32).reshape(3, 1)
    return StereoConfig(k1=k1, d1=d1, k2=k2, d2=d2, r=r, t=t, image_size=image_size)


def build_sgbm_params(image_size: tuple) -> dict:
    width, _ = image_size
    num_disp = max(16, (width // 8) // 16 * 16)
    block_size = cfg.SGBM_BLOCK_SIZE
    return {
        "minDisparity": 0,
        "numDisparities": num_disp,
        "blockSize": block_size,
        "P1": 8 * 3 * block_size * block_size,
        "P2": 32 * 3 * block_size * block_size,
        "disp12MaxDiff": 0,
        "preFilterCap": cfg.SGBM_PREFILTER_CAP,
        "uniquenessRatio": cfg.SGBM_UNIQUENESS,
        "speckleWindowSize": cfg.SGBM_SPECKLE_WINDOW,
        "speckleRange": cfg.SGBM_SPECKLE_RANGE,
    }
