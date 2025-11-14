import cv2
import yaml
import numpy as np

class Solver:
    def __init__(self, config_path, obj_size=0.05):
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        self.intrinsic_matrix = np.array(config_data['Left']['CameraMatrix']['data']).reshape(3,3)
        self.dist_coeffs = np.array(config_data['Left']['distortion_coefficients']['data'])
        self.obj_size = obj_size
        self.obj_points = np.array([
            (-self.obj_size/2, -self.obj_size/2, 0),
            (-self.obj_size/2,  self.obj_size/2, 0),
            ( self.obj_size/2,  self.obj_size/2, 0),
            ( self.obj_size/2, -self.obj_size/2, 0),
            ]
        )
        self.tvec = None
        self.rvec = None

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
            return success, self.rvec, self.tvec
        else:
            return success, None, None

    def visualize_pose(self, image, length=0.01):
        """
        在图像上绘制表示位姿的3D坐标轴
        
        参数:
        image: 输入图像
        camera_matrix: 相机内参矩阵
        dist_coeffs: 相机畸变系数
        rvec: 旋转向量 (3x1)
        tvec: 平移向量 (3x1)
        length: 坐标轴长度 (米)
        """
        
        # 定义3D坐标轴的点 (X, Y, Z轴)
        axis_points = np.float32([
            [0, 0, 0],           # 原点
            [length, 0, 0],      # X轴
            [0, length, 0],      # Y轴  
            [0, 0, length]       # Z轴
        ]).reshape(-1, 3)
        
        # 将3D点投影到2D图像平面
        img_points, _ = cv2.projectPoints(axis_points, self.rvec, self.tvec, self.intrinsic_matrix, self.dist_coeffs)
        img_points = img_points.reshape(-1, 2).astype(int)
        
        # 提取各个点的坐标
        origin = tuple(img_points[0])
        x_axis = tuple(img_points[1])
        y_axis = tuple(img_points[2]) 
        # z_axis = tuple(img_points[3])
        
        # 绘制坐标轴线条
        # X轴 - 红色
        cv2.arrowedLine(image, origin, x_axis, (0, 0, 255), 3)
        # Y轴 - 绿色
        cv2.arrowedLine(image, origin, y_axis, (0, 255, 0), 3)
        # Z轴 - 蓝色
        # cv2.arrowedLine(image, origin, z_axis, (255, 0, 0), 3)
        
        # 在轴末端添加标签
        cv2.putText(image, 'X', x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, 'Y', y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv2.putText(image, 'Z', z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, f"X: {self.tvec[0][0]:.2f}m", (origin[0]+50, origin[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(image, f"Y: {self.tvec[1][0]:.2f}m", (origin[0]+50, origin[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.putText(image, f"Z: {self.tvec[2][0]:.2f}m", (origin[0]+50, origin[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        return image
    def sort_points_(self, points):
        points = np.array(points).reshape(4, 2)
        
        center = np.mean(points, axis=0)
        
        angles = []
        for point in points:
            angle = np.arctan2(point[1] - center[1], point[0] - center[0])
            angles.append(angle)

        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        return sorted_points
    

