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
            (-self.obj_width/2, -self.obj_length/2, 0),
            (-self.obj_width/2,  self.obj_length/2, 0),
            ( self.obj_width/2,  self.obj_length/2, 0),
            ( self.obj_width/2, -self.obj_length/2, 0),
        ])
        self.tvec = None
        self.rvec = None

    def solver(self, result):
        # 如果多于4/5个点，则寻找最下方的5个点
        # 基于图像y轴进行排序
        final_points = []
        if len(result) >= 4:
            cmp = lambda item: item[1]
            result.sort(key=cmp)
            final_points = result[-4:]
            pass
        if(len(results) == 4):
            final_points = result
        for conf, box in results:
            x1, y1, x2, y2 = map(int, box)
            x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 2)
            target_point.append((x_center, y_center))
        success, rvec, tvec = solver.solve_pnp(target_point)
        if success == False:
            print("PnP solving failed.")
            pass
        msg = f"{tvec[0]:.2f},{tvec[1]:.2f},{abs(tvec[2]):.2f}, {rvec[0]:.2f},{rvec[1]:.2f},{rvec[2]:.2f}\r\n"
        # print("Pose:", msg.strip())
        # ser.write(msg.encode())
        if DEBUG:
            out_frame = solver.visualize_pose(left, length=0.05)
            cv2.imshow("Pose Visualization", out_frame)
        pass
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

    def visualize_pose(self, image, length=0.01):
        axis_points = np.float32([
            [0, 0, 0],           # 原点
            [length, 0, 0],      # X轴
            [0, length, 0],      # Y轴  
            [0, 0, length]       # Z轴
        ]).reshape(-1, 3)
        
        # Reproject 3D points to image plane
        img_points, _ = cv2.projectPoints(axis_points, self.rvec, self.tvec, self.intrinsic_matrix, self.dist_coeffs)
        img_points = img_points.reshape(-1, 2).astype(int)
        
        origin = tuple(img_points[0])
        x_axis = tuple(img_points[1])
        y_axis = tuple(img_points[2]) 
        # z_axis = tuple(img_points[3])
        
        # visualize the axes(x, y, z) -> (red, green, blue)
        cv2.arrowedLine(image, origin, x_axis, (0, 0, 255), 3)
        cv2.arrowedLine(image, origin, y_axis, (0, 255, 0), 3)
        # cv2.arrowedLine(image, origin, z_axis, (255, 0, 0), 3)
        cv2.putText(image, 'X', x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, 'Y', y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv2.putText(image, 'Z', z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, f"X: {self.tvec[0]:.2f}m", (origin[0]+50, origin[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(image, f"Y: {self.tvec[1]:.2f}m", (origin[0]+50, origin[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(image, f"Z: {self.tvec[2]:.2f}m", (origin[0]+50, origin[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        return image

    def sort_points_(self, points) -> np.ndarray:
        points = np.array(points).reshape(-1, 2)
        if POINT_MODULE == 0:
            # 需要剔除镜像点
            if len(points) > 4:
                points = points[np.argsort(points[:, 1])][-4:]
        else:
            if len(points) > 5:
                points = points[np.argsort(points[:, 1])][-5:]
                # 手动选择中心点
                pass
        center = np.mean(points, axis=0)
        
        angles = []
        for point in points:
            angle = np.arctan2(point[1] - center[1], point[0] - center[0])
            angles.append(angle)

        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        return sorted_points