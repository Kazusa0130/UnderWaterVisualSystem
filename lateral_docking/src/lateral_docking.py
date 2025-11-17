# from src.SGBM import *
from src.object_detector import *
import cv2
import numpy as np
from src.solver import *


MODEL_PATH = "../weights/best.pt"
VIDEO_PATH = "../videos/npu_test.mp4" # 0
CONFIG_PATH = "../config/stereo_camera_npu/camera_parameters.yaml"
SAVE_PATH = "../videos/pose_output.mp4"

DEBUG = True
SAVE_OUTPUT = False
SGM = False

def main():
    detector = ObjectDetector(model_path=MODEL_PATH)
    solver = Solver(config_path=CONFIG_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(SAVE_PATH, fourcc, 20.0, (640, 480))
    if not cap.isOpened():
        print("❌ 无法打开视频：", VIDEO_PATH)
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取视频帧，可能已到达视频末尾。")
            break
        left = frame[:, 0:640]
        right = frame[:, 640:1280]
        if SGM:
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            disp = sg.compute(left_gray, right_gray)
        if DEBUG:
            cv2.imwrite("LeftImage.jpg", left)
            cv2.imwrite("RightImage.jpg", right)
        
        results = detector.detect(left)
        target_point = []
        if(len(results) == 4):
            for conf, box in results:
                x1, y1, x2, y2 = map(int, box)
                x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                target_point.append((x_center, y_center))
            success, rvec, tvec = solver.solve_pnp(target_point)
            if success == False: continue

            print(f"X: {tvec[0]:.2f}m Y: {tvec[1]:.2f}m Z: {tvec[2]:.2f}m")
        
            if DEBUG:
                out_frame = solver.visualize_pose(frame[:, 0:640], length=0.05)
                cv2.imshow("Pose Visualization", out_frame)
        
        if SAVE_OUTPUT:
            out.write(out_frame)
            # out.write(disparity_map)
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
