from SGBM import *
from ObjectDetector import *
import cv2
import numpy as np

MODEL_PATH = "./weights/best.pt"
VIDEO_PATH = "./videos/npu_test.mp4" # 0

DEBUG = True
SAVE_OUTPUT = True

def main():
    sgbm = SGBM()
    detector = ObjectDetector(model_path=MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    if not cap.isOpened():
        print("❌ 无法打开视频：", VIDEO_PATH)
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取视频帧，可能已到达视频末尾。")
            break
        
        disparity_map = sgbm.compute_disparity(frame, Debug=DEBUG)
        
        results = detector.detect(frame[:, 0:640])
        
        for conf, box in results:
            x1, y1, x2, y2 = map(int, box)
            x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame[:, 0:640], (x1, y1), (x2, y2), (0, 255, 0), 2)
            position_world = sgbm.GetPosition_World(x_center, y_center)
            cv2.putText(frame[:, 0:640], f"Pos: ({position_world[0]:.2f}, {position_world[1]:.2f})", (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Frame", frame[:, 0:640])
        if DEBUG:
            cv2.imshow("Disparity Map", disparity_map)
        
        if SAVE_OUTPUT:
            out.write(frame[:, 0:640])
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
