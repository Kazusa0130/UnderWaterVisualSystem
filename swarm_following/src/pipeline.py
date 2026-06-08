import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

import config as cfg
from stereo_depth import StereoDepthEstimator, load_stereo_config, build_sgbm_params
from vision_utils import find_black_center, classify_in_roi
from tools import open_video_writers, open_serial_log


def _setup_camera():
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, cfg.FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    return cap


def process_stream():
    start_time = time.time()
    cap = _setup_camera()

    image_size = (cfg.WIDTH, cfg.HEIGHT)
    stereo_config = load_stereo_config(cfg.CAMERA_YAML, image_size)
    sgbm_params = build_sgbm_params(image_size)
    depth_estimator = StereoDepthEstimator(stereo_config, sgbm_params)

    model = YOLO(cfg.WEIGHTS_PATH).to("cuda")

    # 串口直接初始化（不使用多线程）
    ser = None
    if cfg.SERIAL_ENABLED:
        try:
            import serial
            ser = serial.Serial(cfg.SERIAL_PORT, cfg.SERIAL_BAUD, timeout=0.1)
            time.sleep(2)  # 等待串口初始化
            print(f"Serial port opened: {cfg.SERIAL_PORT}")
        except Exception as e:
            print(f"Failed to open serial port: {e}")
            ser = None

    recent_distances = deque(maxlen=10)
    non_count = 0
    raw_writer = None
    out_writer = None
    serial_log = None

    try:
        if cfg.SAVE_OUTPUT:
            raw_writer, out_writer, _, _ = open_video_writers(cfg.SAVE_PATH, cfg.FPS, (cfg.WIDTH, cfg.HEIGHT))
            print("Video writers opened.")
            if cfg.SAVE_SERIAL_LOG:
                serial_log, _ = open_serial_log(cfg.SAVE_PATH)

        while True:
            loop_start_time = time.time()

            # 直接读取帧（不使用多线程）
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame or video ended.")
                break

            frame = cv2.resize(frame, (cfg.WIDTH * 2, cfg.HEIGHT), interpolation=cv2.INTER_AREA)
            frame_right = frame[:, : cfg.WIDTH]
            frame_left = frame[:, cfg.WIDTH :]
            frame_left = cv2.flip(frame_left, -1)
            frame_right = cv2.flip(frame_right, -1)

            left_rect, right_rect = depth_estimator.rectify(frame_left, frame_right)

            if cfg.SAVE_OUTPUT and cfg.SAVE_RAW_VIDEO and raw_writer is not None:
                raw_writer.write(left_rect)

            results = list(model.track(left_rect, persist=True, stream=False, conf=cfg.CONF))

            if not results or results[0].boxes.id is None:
                non_count += 1
                if cfg.SEND_ON_EMPTY and non_count % 5 == 0 and ser is not None:
                    serial_data = "[0.00,0.00,0.00,0.00,0.00,0.00]\r\n"
                    ser.write(serial_data.encode("utf-8"))
                    if serial_log is not None:
                        serial_log.write(f"{time.time():.3f} {serial_data}")
                if cfg.SAVE_OUTPUT and cfg.SAVE_OUTPUT_VIDEO and out_writer is not None:
                    out_writer.write(left_rect)

                end_time = time.time()
                if cfg.SHOW_FPS:
                    print(f"Total CostTime: {end_time - loop_start_time:.4f}")

                if cfg.SHOW_IMSHOW:
                    cv2.imshow(cfg.WINDOW_NAME, left_rect)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            annotated_frame = results[0].plot()
            height, width = left_rect.shape[:2]

            for box in results[0].boxes.data.tolist():
                x1, y1, x2, y2, _, score, class_id = box
                x1 = max(0, min(int(x1), width - 1))
                y1 = max(0, min(int(y1), height - 1))
                x2 = max(0, min(int(x2), width))
                y2 = max(0, min(int(y2), height))
                if x2 - x1 <= 1 or y2 - y1 <= 1:
                    continue
                roi = left_rect[y1:y2, x1:x2]
                roi_right = right_rect[y1:y2, x1:x2]

                label, area = classify_in_roi(annotated_frame, x1, y1, x2, y2)

                local_center = find_black_center(roi)

                if local_center is None:
                    continue
                centerpoint = (local_center[0] + x1, local_center[1] + y1)

                disparity_roi = depth_estimator.compute_disparity(roi, roi_right)
                disp_value = float(disparity_roi[local_center[1], local_center[0]])
                cam_x, cam_y, dis = depth_estimator.project_to_3d(centerpoint[0], centerpoint[1], disp_value)

                recent_distances.append(dis)
                smoothed_dis = float(np.mean(recent_distances))

                if 0 < smoothed_dis < cfg.MAX_DEPTH and abs(cam_x) < cfg.MAX_AXIS and abs(cam_y) < cfg.MAX_AXIS:
                    non_count = 0
                    cv2.putText(annotated_frame, f"z: {smoothed_dis:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"cam_x: {cam_x:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"cam_y: {cam_y:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    cam_x /= 10.0
                    cam_y /= 10.0
                    dis /= 10.0
                    x_serial = f"{(-cam_x):.2f}"
                    y_serial = f"{(-cam_y):.2f}"
                    z_serial = f"{abs(dis):.2f}"
                    distance_serial = f"{(cam_x * cam_x + cam_y * cam_y + dis * dis) ** 0.5:.2f}"
                    if ser is not None:
                        serial_data = f"[{x_serial},{y_serial},{z_serial},{distance_serial},{y_serial},{z_serial}]\r\n"
                        ser.write(serial_data.encode("utf-8"))
                        if serial_log is not None:
                            serial_log.write(f"{time.time():.3f} {serial_data}")

                cv2.putText(annotated_frame, f"x: {centerpoint[0]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"y: {centerpoint[1]}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"cls:{label} area:{area}", (x1, max(0, y1 - 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,)
                cv2.circle(annotated_frame, centerpoint, 5, (0, 0, 255), -1)

            if cfg.SHOW_IMSHOW:
                cv2.imshow(cfg.WINDOW_NAME, annotated_frame)
            if cfg.SAVE_OUTPUT and cfg.SAVE_OUTPUT_VIDEO and out_writer is not None:
                out_writer.write(annotated_frame)

            end_time = time.time()
            if cfg.SHOW_FPS:
                print(f"Total CostTime: {end_time - loop_start_time:.4f}")

            if cfg.SHOW_IMSHOW:
                if cv2.waitKey(1) & 0xFF == ord("w"):
                    cv2.waitKey(0)
    finally:
        cap.release()
        if ser is not None:
            ser.close()
            print("Serial port closed.")
        if raw_writer is not None:
            raw_writer.release()
        if out_writer is not None:
            out_writer.release()
        if serial_log is not None:
            serial_log.close()
        cv2.destroyAllWindows()
