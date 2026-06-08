import cv2
import yaml
import numpy as np
from config import *

class Solver:
    def __init__(self, config_path, obj_width=0.05, obj_length=0.05) -> None:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        self.intrinsic_matrix = np.array(config_data['Left']['CameraMatrix']['data']).reshape(3,3)
        self.dist_coeffs = np.array(config_data['Left']['distortion_coefficients']['data'])
        self.obj_width = obj_width
        self.obj_length = obj_length
        self.obj_points = np.array([
            (-0.100, -0.300, 0.000),   # 左上
            ( 0.100, -0.300, 0.000),   # 右上
            ( 0.245,  0.300, 0.000),   # 右下
            (-0.245,  0.300, 0.000),   # 左下
        ])
        self.obj_points_5 = np.array([
            (-0.100, -0.300, 0.000),   # 左上
            ( 0.100, -0.300, 0.000),   # 右上
            ( 0.245,  0.300, 0.000),   # 右下
            (-0.245,  0.300, 0.000),   # 左下
            (0, 0, -0.98)
        ])
        self.tvec = None
        self.rvec = None
        self.mode = 0 # 0: 4 points, 1: 5 points

    def solver(self, points):
        """Solves PnP with the given image points.

        Automatically selects 4-point or 5-point mode based on the number
        of detected points.

        Args:
            points: List of (x, y) pixel coordinates.

        Returns:
            Tuple of (success, rvec, tvec).
        """
        target_points = []
        if len(points) >= 5:
            self.mode = 1
            if len(points) > 5:
                points.sort(key=lambda item: item[1])
                target_points = points[-5:]
            elif len(points) == 5:
                target_points = points
            else:
                print("Not enough points for 5-point PnP.")
                return False, None, None
            success, rvec, tvec = self.solve_pnp_5p(target_points)
        elif len(points) == 4:
            self.mode = 0
            target_points = points
            success, rvec, tvec = self.solve_pnp(target_points)
        else:
            print("Not enough points for PnP (need at least 4).")
            return False, None, None

        if not success:
            print("PnP solving failed.")
        return success, rvec, tvec

    def solve_pnp_5p(self, target_points):
        points = self.sort_points_(target_points)
        points = np.array([
            [points[0][0], points[0][1]],
            [points[1][0], points[1][1]],
            [points[2][0], points[2][1]],
            [points[3][0], points[3][1]],
            [points[4][0], points[4][1]]],
            dtype=np.double
        )
        success, self.rvec, self.tvec = cv2.solvePnP(  
            self.obj_points_5, 
            points, 
            self.intrinsic_matrix, 
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )
        if success and self.rvec is not None and self.tvec is not None and  np.linalg.norm(self.tvec) < 50:
            self.tvec = self.tvec.flatten()
            self.rvec = self.rvec.flatten()
            if self.tvec[2] < 0.4:
                success = False
            return success, self.rvec, self.tvec
        else:
            return success, None, None
    
    def solve_pnp(self, target_points):
        points = self.sort_points_(target_points)
        points = np.array([
            [points[0][0], points[0][1]],
            [points[1][0], points[1][1]],
            [points[2][0], points[2][1]],
            [points[3][0], points[3][1]]], 
            dtype=np.double
        )
        success, self.rvec, self.tvec = cv2.solvePnP(  
            self.obj_points, 
            points, 
            self.intrinsic_matrix, 
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success and self.rvec is not None and self.tvec is not None and  np.linalg.norm(self.tvec) < 50:
            self.tvec = self.tvec.flatten()
            self.rvec = self.rvec.flatten()
            if self.tvec[2] < 0.4:
                success = False
            return success, self.rvec, self.tvec
        else:
            return success, None, None

    def visualize_pose(self, image, length=0.1, show_rotation=True):
        if show_rotation:
            # Reproject 3D axes to image plane
            axis_points_4 = np.float32([
                [0, 0, 0],
                [length, 0, 0],
                [0, length, 0],
                [0, 0, length]
            ]).reshape(-1, 3)
            img_points, _ = cv2.projectPoints(
                axis_points_4, self.rvec, self.tvec,
                self.intrinsic_matrix, self.dist_coeffs
            )
            img_points = img_points.reshape(-1, 2).astype(int)
            origin = tuple(img_points[0])
            x_axis = tuple(img_points[1])
            y_axis = tuple(img_points[2])
            z_axis = tuple(img_points[3])

            rotation_matrix, _ = cv2.Rodrigues(self.rvec)
            sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                yaw = 0.0

            yaw_deg = np.degrees(yaw)
            pitch_deg = np.degrees(pitch)
            roll_deg = np.degrees(roll)

            cv2.arrowedLine(image, origin, x_axis, (0, 0, 255), 3)
            cv2.arrowedLine(image, origin, y_axis, (0, 255, 0), 3)
            cv2.arrowedLine(image, origin, z_axis, (255, 0, 0), 3)
            cv2.putText(image, f"Yaw: {yaw_deg:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Pitch: {pitch_deg:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Roll: {roll_deg:.2f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, f"X: {self.tvec[0]:.2f}m", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Y: {self.tvec[1]:.2f}m", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Z: {self.tvec[2]:.2f}m", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(image, f"X: {self.tvec[0]:.2f}m", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Y: {self.tvec[1]:.2f}m", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Z: {self.tvec[2]:.2f}m", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return image

    def point_selector(self, points):
        best_area = -1.0
        best_points = []
        remain_point = None

        for idx in range(5):
            candidate = [points[j] for j in range(5) if j != idx]
            center_x = sum(point[0] for point in candidate) / 4.0
            center_y = sum(point[1] for point in candidate) / 4.0
            ordered_candidate = sorted(
                candidate,
                key=lambda point: np.arctan2(point[1] - center_y, point[0] - center_x)
            )

            contour = np.array(ordered_candidate, dtype=np.float32).reshape(-1, 1, 2)
            area = cv2.contourArea(contour)

            if area > best_area:
                best_area = area
                best_points = ordered_candidate
                remain_point = points[idx]
    
        best_points.append(remain_point)
        return best_points

    def sort_points_(self, points) -> np.ndarray:
        points = np.array(points).reshape(-1, 2)
        points = points[np.argsort(points[:, 1])]
        center = np.mean(points, axis=0)
        if self.mode == 0:
            tmp_points = points
        else:
            points = self.point_selector(points)
            tmp_points = np.array(points[:4])
            center_point = np.array(points[4])
        angles = []
        for tmp_point in tmp_points:
            angle = np.arctan2(tmp_point[1] - center[1],
                               tmp_point[0] - center[0])
            angles.append(angle)

        sorted_indices = np.argsort(angles)
        sorted_points = tmp_points[sorted_indices]
        if self.mode == 1:
            sorted_points = np.vstack([sorted_points, center_point])
        return sorted_points

    def get_camera_pose_in_target_frame(
        self,
        rvec: np.ndarray | None = None,
        tvec: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Calculate camera pose in target coordinate frame.

        OpenCV solvePnP returns rvec, tvec representing target pose in camera
        frame. This method converts to camera pose in target frame, consistent
        with the ``target-as-origin`` visualization mode in
        :class:`visualizer_3d.PoseVisualizer3D`.

        Args:
            rvec: Target rotation vector in camera frame (3,). Uses
                :attr:`self.rvec` if None.
            tvec: Target translation vector in camera frame (3,). Uses
                :attr:`self.tvec` if None.

        Returns:
            Tuple of ``(rvec_cam, tvec_cam)`` where:
                - ``rvec_cam``: Rotation vector of camera in target frame.
                - ``tvec_cam``: Position of camera in target frame.
            Returns ``(None, None)`` if inputs are invalid.
        """
        if rvec is None:
            rvec = self.rvec
        if tvec is None:
            tvec = self.tvec
        if rvec is None or tvec is None:
            return None, None

        R_target_to_cam, _ = cv2.Rodrigues(np.array(rvec).flatten())
        R_cam_to_target = R_target_to_cam.T
        rvec_cam, _ = cv2.Rodrigues(R_cam_to_target)
        tvec_cam = -R_cam_to_target @ np.array(tvec).flatten()
        return rvec_cam.flatten(), tvec_cam.flatten()