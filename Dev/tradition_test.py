import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from itertools import combinations
from collections import defaultdict

VIDEO_PATH = "./videos/npu_test.mp4" # 0

def sort_parallelogram_points(points):
    """将平行四边形的点按顺时针顺序排序"""
    # 计算中心点
    center_x = sum(p[0] for p in points) / 4
    center_y = sum(p[1] for p in points) / 4
    
    # 按角度排序
    angle_from_center = lambda point: np.arctan2(point[1] - center_y, point[0] - center_x)
    
    return sorted(points, key=angle_from_center)

def find_parallelograms_optimized(points: list[tuple[float, float]]) -> list[list[tuple]]:
    """
    使用对角线中点性质优化查找平行四边形
    
    性质：平行四边形的对角线互相平分
    """
    # 构建中点到点对的映射
    midpoint_to_pairs = defaultdict(list)
    
    # 计算所有点对的中点
    for i, j in combinations(range(len(points)), 2):
        p1, p2 = points[i], points[j]
        midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        midpoint_to_pairs[midpoint].append((i, j))
	# 合并相近的中点（例如距离小于阈值则视为同一个中点）
    merged_midpoints = []
    merged_map = dict()
    threshold = 50  # 可根据实际情况调整
    print(midpoint_to_pairs.keys())
    for midpoint in midpoint_to_pairs.keys():
        # 尝试将当前中点归并到已存在的中点集合
        for idx, m in enumerate(merged_midpoints):
            dist = np.hypot(midpoint[0] - m[0], midpoint[1] - m[1])
            if dist < threshold:
                merged_map[midpoint] = idx
            break
        else:
            # 没有找到合适的归并对象，新建一个
            merged_map[midpoint] = len(merged_midpoints)
            merged_midpoints.append(midpoint)

	# 新的合并后的映射
    merged_midpoint_to_pairs = defaultdict(list)
    for midpoint, pairs in midpoint_to_pairs.items():
        idx = merged_map[midpoint]
        merged_midpoint_to_pairs[merged_midpoints[idx]].extend(pairs)

    midpoint_to_pairs = merged_midpoint_to_pairs
    parallelograms = []
    
    # 对于每个中点，如果有两对以上的点对共享该中点，则形成平行四边形
    for midpoint, pairs in midpoint_to_pairs.items():

        if len(pairs) >= 2:
            # 这些点对形成平行四边形
            for pair1, pair2 in combinations(pairs, 2):
                # 获取四个不同的点索引
                all_indices = set(pair1) | set(pair2)
                if len(all_indices) == 4:
                    parallelogram_points = [points[idx] for idx in all_indices]
                    # 验证确实是平行四边形
                    sorted_points = sort_parallelogram_points(parallelogram_points)
                    # if verify_parallelogram(sorted_points):  
                    parallelograms.append(sorted_points)
    
    return parallelograms
def verify_parallelogram(points):
    p1, p2, p3, p4 = points
    vector = lambda p1, p2: (p2[0] - p1[0], p2[1] - p1[1])
    is_parallel = lambda v1, v2: abs(v1[0] * v2[1] - v1[1] * v2[0]) < 100
    v12 = vector(p1, p2)
    v34 = vector(p3, p4)
    v13 = vector(p1, p3)
    v24 = vector(p2, p4)
    return is_parallel(v12, v34)

# 他这里得选两种轮廓，一种是内轮廓，另一种没有内轮廓的轮廓
def get_contours(contours, hierarchy):
    target_contours = []
    hierarchy = hierarchy[0]  # 去掉外层维度
    for i, contour in enumerate(contours):
        if hierarchy[i][3] != -1 and hierarchy[i][2] == -1:
            target_contours.append(contour)
        if hierarchy[i][3] == -1 and hierarchy[i][2] == -1:
            area = cv2.contourArea(contour)
            # print("Contour area:", area)
            if 30 <= area <= 80:
                target_contours.append(contour)
    return target_contours


def process_frame(img):
	"""从 BGR 图像提取绿色点的二值 mask（G 与 R/B 相减的简单实现）。"""
	# 校验输入
	if img is None:
		return None
	# 获取G、B、R通道（OpenCV 使用 BGR 顺序）
	B, G, R = cv2.split(img)
	# G - B 和 G - R
	GB = cv2.subtract(G, B)
	GR = cv2.subtract(G, R)
	# 合并两者（按位或），强调绿色部分
	mask = cv2.bitwise_or(GB, GR)
	mask = cv2.GaussianBlur(mask, (5, 5), 0)
	# 自适应二值化（OTSU）
	_, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
	return binary_mask


def run_with_source(source, save=False):
	"""根据 source（文件路径或摄像头编号）运行逐帧处理并显示结果。"""
	# 如果 source 是字符串数字则转为 int
	if isinstance(source, str) and source.isdigit():
		source = int(source)

	cap = cv2.VideoCapture(source)
	if not cap.isOpened():
		# 可能是图片路径，尝试读取图片
		img = cv2.imread(str(source))
		if img is None:
			print('无法打开输入:', source)
			return
		mask = process_frame(img)
		cv2.imshow('Input', img)
		cv2.imshow('Mask', mask)
		cv2.waitKey(0)
		if save:
			out_path = os.path.splitext(str(source))[0] + '_mask.png'
			cv2.imwrite(out_path, mask)
			print('已保存:', out_path)
		cv2.destroyAllWindows()
		return

	frame_id = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		mask = process_frame(frame[:, :640])
		# 只要内轮廓, 他这里得选两种轮廓，一种是内轮廓，另一种没有内轮廓的轮廓
		contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		inner_contours = get_contours(contours, hierarchy)

		# cv2.drawContours(frame, contours, -1, (0, 0, 255), 1)
		# cv2.drawContours(frame, inner_contours, -1, (0, 0, 255), 2)
		# 自此，可以通过这个部分完成对特征的粗提取
		point_list = []
		for contour in inner_contours:
			moments = cv2.moments(contour)
			x = int( moments['m10'] / moments['m00'] )
			y =  int( moments['m01'] / moments['m00'] )
			cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 3)
			(x, y), radius = cv2.minEnclosingCircle(contour)
			point_list.append((x, y))   
            # cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
		parallelograms = find_parallelograms_optimized(point_list)
		for para in parallelograms:
			for i in range(4):
				pt1 = (int(para[i][0]), int(para[i][1]))
				pt2 = (int(para[(i + 1) % 4][0]), int(para[(i + 1) % 4][1]))
				cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
		if mask is None:
			break
		cv2.imshow('Frame', frame)
		cv2.imshow('Mask', mask)
		key = cv2.waitKey(30) & 0xFF
		if key == ord('q'):
			break
		if key == ord('s') and save:
			out_path = f'frame_{frame_id}_mask.png'
			cv2.imwrite(out_path, mask)
			print('已保存:', out_path)
		frame_id += 1
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	run_with_source(VIDEO_PATH, save=False)