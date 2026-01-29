from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", debug=False) -> None:
        self.model = YOLO(model_path)
        self.debug = debug
        self.model.to("cuda")
        self.results = None
        self.target_list = []

    def detect(self, image):
        self.results = self.model.track(source=image, conf=0.7, iou=0.5, verbose=False)
        return self.results
    def get_target_list(self):
        self.target_list = []
        for box in self.results[0].boxes:
            # 如果box的宽度和高度小于某个阈值，则忽略
            if (box.xyxy[0][2] - box.xyxy[0][0]) < 5 or (box.xyxy[0][3] - box.xyxy[0][1]) < 5:
                continue
            self.target_list.append([float(box.conf[0]), box.xyxy[0].tolist()])
        return self.target_list