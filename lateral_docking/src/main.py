import cv2
import numpy as np
import yaml
import time
# import serial

from detector import *
from solver import *
from tools import *
from config import *
from red_led_detector import TargetPointDetector

def visualize_frame(
    frame: np.ndarray,
    rvec: np.ndarray | None,
    tvec: np.ndarray | None,
    solver: Solver,
    is_valid: bool = True,
    show_rotation: bool = True,
) -> np.ndarray:
    """可视化单帧图像和2D位姿。

    在2D图像上绘制位姿坐标轴。

    Args:
        frame: 输入图像帧 (H, W, 3) BGR格式。
        rvec: 旋转向量 (3,)，目标在相机坐标系中的旋转。
        tvec: 平移向量 (3,)，目标在相机坐标系中的位置。
        solver: Solver实例，用于绘制2D位姿可视化。
        is_valid: 位姿是否有效，影响显示颜色。
        show_rotation: 是否显示旋转信息（坐标轴+Yaw/Pitch/Roll）。
            4点模式时建议设为 False。

    Returns:
        绘制了位姿信息的输出图像帧。
    """
    # 绘制检测结果
    out_frame = frame.copy()

    # 如果有有效的位姿，绘制坐标轴
    if rvec is not None and tvec is not None:
        solver.rvec = rvec
        solver.tvec = tvec
        out_frame = solver.visualize_pose(
            out_frame, length=0.2, show_rotation=show_rotation
        )

        # 在2D图像右上角显示状态，避免与位姿信息重叠
        if DEBUG:
            status_text = "Valid" if is_valid else "Invalid"
            color = (0, 255, 0) if is_valid else (0, 0, 255)
            h, w = out_frame.shape[:2]
            text = f"Status: {status_text}"
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.putText(
                out_frame, text,
                (w - tw - 10, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
            )

    return out_frame


def main():
    detector = ObjectDetector(model_path=MODEL_PATH, debug=DEBUG)
    solver = Solver(config_path=CONFIG_PATH, obj_width=OBJ_WIDTH, obj_length=OBJ_LENGTH)
    target_point_detector = TargetPointDetector()
    # ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    cap = cv2.VideoCapture(VIDEO_PATH)

    raw_data_count, output_data_count, traj_data_count = count_files_in_directory(SAVE_PATH)

    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # ty:ignore[unresolved-attribute]
        raw_data_out = cv2.VideoWriter(SAVE_PATH+"raw_data/"+f"raw_data_{raw_data_count}.avi", fourcc, 20.0, (640, 480))
        output_data_out = cv2.VideoWriter(SAVE_PATH+"output_data/"+f"output_{output_data_count}.avi", fourcc, 20.0, (640, 480))
        # 创建轨迹数据文件 (CSV格式)
        traj_file = open(SAVE_PATH+"traj_data/"+f"traj_{traj_data_count}.csv", 'w')
        traj_file.write("timestamp,frame_id,x,y,z,yaw,pitch,roll\n")
    else:
        traj_file = None

    if not cap.isOpened():
        print("Unable to open video:", VIDEO_PATH)
        exit()

    start_timestamp = time.time()
    frame_id = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("VideoStream end or cannot fetch the frame.")
            break
        right = frame[:, 0:640, :]
        left = frame[:, 640:1280, :]
        if FLIP:
            left = cv2.flip(left, -1)
            right = cv2.flip(right, -1)
        if SAVE_OUTPUT:
            raw_data_out.write(left)

        # 目标检测
        results = detector.detect(left)
        target_list = detector.get_target_list()
        out_frame = results[0].plot()

        rvec, tvec = None, None
        is_valid = False

        # 先尝试检测红色中心光点
        red_targets = target_point_detector.detect_all(left)

        # 若检测到红色光点，从 YOLO 结果中剔除与之重叠的检测框
        # （避免中心点被 YOLO 和 red_led_detector 重复送入 PnP）
        if red_targets:
            rx, ry = red_targets[0]['center']
            filtered_target_list = []
            for conf, box in target_list:
                x1, y1, x2, y2 = map(int, box)
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2
                dist = ((x_center - rx) ** 2 + (y_center - ry) ** 2) ** 0.5
                if dist < 10:
                    if DEBUG:
                        print(
                            f"Filtering YOLO box at "
                            f"({x_center},{y_center}) overlapping with red LED"
                        )
                    continue
                filtered_target_list.append([conf, box])
            target_list = filtered_target_list

        if len(target_list) < 4:
            print("Not enough points detected.")
        else:
            center_points = []
            for conf, box in target_list:
                x1, y1, x2, y2 = map(int, box)
                x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                center_points.append((x_center, y_center))

            if red_targets:
                rx, ry = red_targets[0]['center']
                center_points.append((rx, ry))
                solver.mode = 1
                if DEBUG:
                    print(
                        f"Red LED detected at ({rx}, {ry}), "
                        f"using 5-point PnP."
                    )
                out_frame = target_point_detector.visualize_all(
                    out_frame, red_targets
                )
            else:
                solver.mode = 0
                if DEBUG:
                    print("No red LED detected, using 4-point PnP.")

            success, rvec_raw, tvec_raw = solver.solver(center_points)

            if success and rvec_raw is not None and tvec_raw is not None:
                is_valid = True
                tvec = tvec_raw.flatten()
                rvec = rvec_raw.flatten()

                msg = f"{tvec[0]:.2f},{tvec[1]:.2f},{tvec[2]:.2f}, {rvec[0]:.2f},{rvec[1]:.2f},{rvec[2]:.2f}\r\n"
                print("Pose:", msg.strip())
                # ser.write(msg.encode())

        # 保存轨迹数据 (CSV格式: timestamp,frame_id,x,y,z,yaw,pitch,roll)
        if SAVE_OUTPUT and traj_file is not None:
            timestamp = time.time() - start_timestamp
            if rvec is not None and tvec is not None:
                # yaw=rvec[2], pitch=rvec[1], roll=rvec[0]
                traj_file.write(f"{timestamp:.6f},{frame_id},{tvec[0]:.6f},{tvec[1]:.6f},{tvec[2]:.6f},{rvec[2]:.6f},{rvec[1]:.6f},{rvec[0]:.6f}\n")
                traj_file.flush()
            else:
                traj_file.write(f"{timestamp:.6f},{frame_id},0.0,0.0,0.0,0.0,0.0,0.0\n")
                traj_file.flush()
            frame_id += 1

        # 调用可视化函数处理显示
        # 5点模式显示旋转信息，4点模式仅显示位姿坐标
        out_frame = visualize_frame(
            frame=out_frame,
            rvec=rvec,
            tvec=tvec,
            solver=solver,
            is_valid=is_valid,
            show_rotation=(solver.mode == 1),
        )

        if DEBUG:
            cv2.imshow("Pose Visualization", out_frame)

        if SAVE_OUTPUT:
            output_data_out.write(out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end_time = time.time()
        print(f"FPS: {1/(end_time - start_time):.2f}")

    # ser.write(msg.encode())
    if SAVE_OUTPUT:
        raw_data_out.release()
        output_data_out.release()
        if traj_file is not None:
            traj_file.close()
            print(f"Trajectory data saved to {SAVE_PATH}traj_data/traj_{traj_data_count}.csv")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
