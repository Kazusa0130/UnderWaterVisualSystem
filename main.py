# from src.SGBM import *
from src.ObjectDetector import *
import cv2
import numpy as np
from Solver import *
import src.sgbm

MODEL_PATH = "./weights/best.pt"
VIDEO_PATH = "./videos/npu_test.mp4" # 0
CONFIG_PATH = "./config/stereo_camera_npu/camera_parameters.yaml"

DEBUG = True
SAVE_OUTPUT = False

def main():
    # sgbm = SGBM()
    detector = ObjectDetector(model_path=MODEL_PATH)
    solver = Solver(config_path=CONFIG_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('pose_output.mp4', fourcc, 20.0, (640, 480))
    if not cap.isOpened():
        print("❌ 无法打开视频：", VIDEO_PATH)
        exit()
    # 使用pybind11绑定的cuda-sgm
    # w, h = 640, 480
    sg = sgbm.SGBMWrapper()
    sg.init("config/stereo_camera_itr/camera_parameters.yaml", w, h)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取视频帧，可能已到达视频末尾。")
            break
        left = frame[:, 0:640]
        right = frame[:, 640:1280]
        disparity_map = sgbm.compute_disparity(frame, Debug=DEBUG)
        disp = sg.compute(left, right)

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
            tvec = tvec.flatten()
            rvec = rvec.flatten()
            print(f"X: {tvec[0]:.2f}m Y: {tvec[1]:.2f}m Z: {tvec[2]:.2f}m")
            out_frame = solver.visualize_pose(frame[:, 0:640], length=0.05)
            if DEBUG:
                cv2.imshow("Pose Visualization", out_frame)
        disp = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # # 生成深度图（颜色图）
        disp[disp < 0] = 0
        disp[disp > 1000] = 1000
        dis_color = cv2.UMat(disp)
        dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dis_color = cv2.applyColorMap(dis_color, cv2.COLORMAP_TURBO)

        cv2.imshow("Frame", dis_color)
        cv2.imshow("Left", frame[:, 0:640])
        if SAVE_OUTPUT:
            out.write(out_frame)
            out.write(disparity_map)
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
