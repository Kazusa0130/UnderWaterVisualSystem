import cv2
import numpy as np
import yaml
# import serial

# from src.SGBM import *
from detector import *
from solver import *
from tools import *

MODEL_PATH = "./lateral_docking/weights/best.pt"
VIDEO_PATH = "./lateral_docking/videos/npu_test.mp4"
# VIDEO_PATH = 0
CONFIG_PATH = "./lateral_docking/config/stereo_camera_npu/camera_parameters.yaml"
SAVE_PATH = "./lateral_docking/videos/"

DEBUG = True
SAVE_OUTPUT = True
SGM = False

SERIAL_PORT = '/dev/ttyTHS1'
SERIAL_BAUD = 115200

OBJ_LENGTH = 0.05  # m
OBJ_WIDTH = 0.05   # m

def main():
    detector = ObjectDetector(model_path=MODEL_PATH, debug=DEBUG)
    solver = Solver(config_path=CONFIG_PATH, obj_width=OBJ_WIDTH, obj_length=OBJ_LENGTH)
    # ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    cap = cv2.VideoCapture(VIDEO_PATH)

    raw_data_count, output_data_count = count_files_in_directory(SAVE_PATH)
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        raw_data_out = cv2.VideoWriter(SAVE_PATH+"raw_data/"+f"raw_data_{raw_data_count}.mp4", fourcc, 20.0, (640, 480))
        output_data_out = cv2.VideoWriter(SAVE_PATH+"output_data/"+f"output_{output_data_count}.mp4", fourcc, 20.0, (640, 480))
    if not cap.isOpened():
        print("Unable to open video:", VIDEO_PATH)
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("VideoStream end or cannot fetch the frame.")
            break
        right = frame[:, 0:640,:]
        left = frame[:, 640:1280,:]
        if SAVE_OUTPUT:
            raw_data_out.write(left)
        results = detector.detect(left)
        target_point = []
        if(len(results) == 4):
            for conf, box in results:
                x1, y1, x2, y2 = map(int, box)
                x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                target_point.append((x_center, y_center))
            success, rvec, tvec = solver.solve_pnp(target_point)
            if success == False: 
                continue
            msg = f"{tvec[0]:.2f},{tvec[1]:.2f},{abs(tvec[2]):.2f}, {rvec[0]:.2f},{rvec[1]:.2f},{rvec[2]:.2f}\r\n"
            # print("Pose:", msg.strip())
            # ser.write(msg.encode())
            if DEBUG:
                out_frame = solver.visualize_pose(left, length=0.05)
                cv2.imshow("Pose Visualization", out_frame)
            if SAVE_OUTPUT:
                output_data_out.write(out_frame)
        elif SAVE_OUTPUT:
            output_data_out.write(left)
            cv2.imshow("Pose Visualization", left)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # ser.write(msg.encode())
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
