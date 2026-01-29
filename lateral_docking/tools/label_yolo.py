import json
import os

import cv2
from tqdm import tqdm
from ultralytics import YOLO


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "lateral_docking", "models", "best.pt")
VIDEO_PATH = os.path.join(ROOT_DIR, "lateral_docking", "videos", "input.mp4")
OUTPUT_DIR = os.path.join(ROOT_DIR, "lateral_docking", "outputs", "labels")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "label_yolo_config.json")

CONF = 0.25
IOU = 0.5
IMGSZ = 640
DEVICE = "cuda"
SAVE_EMPTY = False
SAVE_IMAGES = False


def load_config():
	global MODEL_PATH, VIDEO_PATH, OUTPUT_DIR, CONF, IOU, IMGSZ, DEVICE, SAVE_EMPTY, SAVE_IMAGES
	if not os.path.exists(CONFIG_PATH):
		return
	try:
		with open(CONFIG_PATH, "r", encoding="utf-8") as f:
			data = json.load(f)
	except Exception:
		return

	MODEL_PATH = str(data.get("model_path", MODEL_PATH))
	VIDEO_PATH = str(data.get("video_path", VIDEO_PATH))
	OUTPUT_DIR = str(data.get("output_dir", OUTPUT_DIR))
	CONF = float(data.get("conf", CONF))
	IOU = float(data.get("iou", IOU))
	IMGSZ = int(data.get("imgsz", IMGSZ))
	DEVICE = str(data.get("device", DEVICE))
	SAVE_EMPTY = bool(data.get("save_empty", SAVE_EMPTY))
	SAVE_IMAGES = bool(data.get("save_images", SAVE_IMAGES))


def _xyxy_to_yolo(xyxy, img_w, img_h):
	x1, y1, x2, y2 = xyxy
	x1 = max(0.0, min(float(x1), img_w - 1))
	y1 = max(0.0, min(float(y1), img_h - 1))
	x2 = max(0.0, min(float(x2), img_w - 1))
	y2 = max(0.0, min(float(y2), img_h - 1))
	w = max(0.0, x2 - x1)
	h = max(0.0, y2 - y1)
	cx = x1 + w / 2.0
	cy = y1 + h / 2.0
	return cx / img_w, cy / img_h, w / img_w, h / img_h


def _write_labels(txt_path, labels):
	lines = [
		f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
		for cls_id, x, y, w, h in labels
	]
	with open(txt_path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines))


def infer_video_to_labels():
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	image_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "images")
	if SAVE_IMAGES:
		os.makedirs(image_dir, exist_ok=True)

	model = YOLO(MODEL_PATH)
	if DEVICE:
		model.to(DEVICE)

	cap = cv2.VideoCapture(VIDEO_PATH)
	if not cap.isOpened():
		raise FileNotFoundError(f"无法打开视频: {VIDEO_PATH}")

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	video_stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
	frame_idx = 0

	progress = tqdm(total=total_frames if total_frames > 0 else None, desc="推理并生成label")

	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frame = frame[:, 640:, :]
		frame = cv2.flip(frame, -1)
		img_h, img_w = frame.shape[:2]
		results = model.predict(
			source=frame,
			conf=CONF,
			iou=IOU,
			imgsz=IMGSZ,
			verbose=False,
			device=DEVICE if DEVICE else None,
		)

		labels = []
		for box in results[0].boxes:
			cls_id = int(box.cls[0].item())
			x, y, w, h = _xyxy_to_yolo(box.xyxy[0].tolist(), img_w, img_h)
			labels.append((cls_id, x, y, w, h))

		txt_name = f"{video_stem}_{frame_idx:06d}.txt"
		txt_path = os.path.join(OUTPUT_DIR, txt_name)
		if labels or SAVE_EMPTY:
			_write_labels(txt_path, labels)

		if SAVE_IMAGES:
			img_name = f"{video_stem}_{frame_idx:06d}.jpg"
			cv2.imwrite(os.path.join(image_dir, img_name), frame)

		frame_idx += 1
		progress.update(1)

	progress.close()
	cap.release()
def main():
	load_config()
	infer_video_to_labels()


if __name__ == "__main__":
	main()
