from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image) -> list:
        target_list = [];
        results = self.model.predict(source=image, conf=0.7, iou=0.8, verbose=False)
        for box in results[0].boxes:
            target_list.append([float(box.conf[0]), box.xyxy[0].tolist()])
        return target_list