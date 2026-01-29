import cv2
import numpy as np
import yaml
# import serial

from detector import *
from solver import *
from tools import *
from config import *

def main():
    detector = ObjectDetector(model_path=MODEL_PATH, debug=DEBUG)
    solver = Solver(config_path=CONFIG_PATH, obj_width=OBJ_WIDTH, obj_length=OBJ_LENGTH)
    # ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    cap = cv2.VideoCapture(VIDEO_PATH)

    raw_data_count, output_data_count = count_files_in_directory(SAVE_PATH)
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        raw_data_out = cv2.VideoWriter(SAVE_PATH+"raw_data/"+f"raw_data_{raw_data_count}.avi", fourcc, 20.0, (640, 480))
        output_data_out = cv2.VideoWriter(SAVE_PATH+"output_data/"+f"output_{output_data_count}.avi", fourcc, 20.0, (640, 480))
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
        if FLIP:
            left = cv2.flip(left, -1)
            right = cv2.flip(right, -1)

        if SAVE_OUTPUT:
            raw_data_out.write(left)
        results = detector.detect(left)
        target_list = detector.get_target_list()
        out_frame = results[0].plot()
        if len(target_list) < 4:
            print("Not enough points detected.")
        else:
            target_point = []
            for conf, box in target_list:
                x1, y1, x2, y2 = map(int, box)
                x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                # cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 2)
                target_point.append((x_center, y_center))
            success, rvec, tvec = solver.solve_pnp(target_point)
            if success == True and rvec is not None and tvec is not None:
                msg = f"{tvec[0]:.2f},{tvec[1]:.2f},{tvec[2]:.2f}, {rvec[0]:.2f},{rvec[1]:.2f},{rvec[2]:.2f}\r\n"
                print("Pose:", msg.strip())
                # ser.write(msg.encode())
                out_frame = solver.visualize_pose(out_frame, length=0.05)
        if DEBUG:
            cv2.imshow("Pose Visualization", out_frame)
        if SAVE_OUTPUT:
            output_data_out.write(out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # ser.write(msg.encode())
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
