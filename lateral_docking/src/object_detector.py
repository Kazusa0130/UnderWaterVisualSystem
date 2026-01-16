from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", debug=False) -> None:
        self.model = YOLO(model_path)
        self.debug = debug
        self.model.to("cuda")

    def detect(self, image) -> list:
        target_list = [];
        results = self.model.predict(source=image, conf=0.8, iou=0.5, verbose=False)
        if self.debug:
            annotated_frame = results[0].plot()
            cv2.imshow("Detection", annotated_frame)
        for box in results[0].boxes:
            target_list.append([float(box.conf[0]), box.xyxy[0].tolist()])
        return target_list