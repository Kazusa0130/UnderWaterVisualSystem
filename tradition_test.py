import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox


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
	# 自适应二值化（OTSU）
	_, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return binary_mask


def choose_input_via_gui():
	"""弹出对话框让用户选择打开文件或使用摄像头（0）。

	返回：文件路径字符串、整数 0（表示摄像头），或 None 表示取消。
	"""
	root = tk.Tk()
	root.withdraw()
	res = messagebox.askquestion('输入选择', '选择输入类型：\n\nYes = 打开文件\nNo = 使用摄像头(0)')
	if res == 'yes':
		file_path = filedialog.askopenfilename(title='选择图片或视频文件',
											   filetypes=[('Media', '*.mp4 *.avi *.mov *.mkv *.jpg *.png *.jpeg'),
														  ('All files', '*.*')])
		root.destroy()
		if not file_path:
			return None
		return file_path
	else:
		root.destroy()
		return 0


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
		mask = process_frame(frame)
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
	src = choose_input_via_gui()
	if src is None or src == '':
		print('未选择输入，退出')
	else:
		run_with_source(src, save=False)