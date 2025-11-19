import cv2
import numpy as np
import camera_configs

class SGBM:
    def __init__(self):
        self.blockSize = 5
        self.param = {
            "minDisparity": 0,
            "numDisparities": 16 * 6,
            "blockSize": self.blockSize,
            "P1": 8 * 3 * self.blockSize * self.blockSize,
            "P2": 32 * 3 * self.blockSize * self.blockSize,
            "disp12MaxDiff": 0,
            "preFilterCap": 15,
            "uniquenessRatio": 8,
            "speckleWindowSize": 100,
            "speckleRange": 2,
        }
        self.left_matcher = cv2.StereoSGBM_create(**self.param)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.disp = None
        self.threeD = None
        
    def compute_disparity(self, img, Debug=False):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgL = img_gray[:, 0:640]
        imgR = img_gray[:, 640:1280]
        cv2.imwrite("LeftImage.jpg", img_gray[:, 0:640])
        cv2.imwrite("RightImage.jpg", img_gray[:, 640:1280])
        # Remap
        img1_rectified = cv2.remap(imgL, camera_configs.LEFT_MAP1, camera_configs.LEFT_MAP2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(imgR, camera_configs.RIGHT_MAP1, camera_configs.RIGHT_MAP2, cv2.INTER_LINEAR)

        left_disparity = self.left_matcher.compute(img1_rectified, img2_rectified)
        right_disparity = self.right_matcher.compute(img2_rectified, img1_rectified)
        print(left_disparity.dtype)
        # cv2.imshow("left disp", left_disparity)
        # filter
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        wls_filter.setLambda(8000.)
        wls_filter.setSigmaColor(1.3)
        wls_filter.setLRCthresh(24)
        wls_filter.setDepthDiscontinuityRadius(3)
        # confidence_map = np.ones_like(left_disparity, dtype=np.uint8) * 255

        filtered_disparity = wls_filter.filter(left_disparity, imgL, disparity_map_right=right_disparity)
        self.threeD = cv2.reprojectImageTo3D(filtered_disparity.astype(np.float32) / 16., camera_configs.Q)
        cv2.imshow("filtered_disparity", filtered_disparity)
        filtered_disparity[filtered_disparity < 0] = 0
        filtered_disparity[filtered_disparity > 1000] = 1000

        self.disp = cv2.normalize(filtered_disparity, filtered_disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.left_disparity = cv2.normalize(left_disparity, left_disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if Debug:
            cv2.imshow("disparity", self.disp)
            cv2.imshow("left disp", self.left_disparity)
            cv2.waitKey(1)
        return self.disp

    def GetPosition_World(self, x, y):
        if self.threeD is None:
            raise ValueError("Disparity map not computed yet. Call compute_disparity() first.")
        return self.threeD[y, x]