import cv2
import numpy as np
import time
import sys

from detector import ObjectDetector
from solver import Solver
from tools import count_files_in_directory
from config import (
    DEBUG,
    MODEL_PATH,
    VIDEO_PATH,
    CONFIG_PATH,
    SAVE_PATH,
    SAVE_OUTPUT,
    FLIP,
    OBJ_WIDTH,
    OBJ_LENGTH,
    SHOW_ROTATION_FOR_4POINT,
    SERIAL_ENABLED,
    SERIAL_PORT,
    SERIAL_BAUD,
)


def visualize_frame(
    frame: np.ndarray,
    rvec: np.ndarray | None,
    tvec: np.ndarray | None,
    solver: Solver,
    is_valid: bool = True,
    show_rotation: bool = True,
) -> np.ndarray:
    """Visualizes 2D pose on a single frame.

    Args:
        frame: Input BGR image (H, W, 3).
        rvec: Rotation vector (3,) of the target in the camera frame.
        tvec: Translation vector (3,) of the target in the camera frame.
        solver: Solver instance for 2D pose visualization.
        is_valid: Whether the pose is valid (affects display color).
        show_rotation: Whether to show rotation info (axes + Yaw/Pitch/Roll).
            Recommended False for 4-point mode.

    Returns:
        Output frame with pose information drawn.
    """
    out_frame = frame.copy()

    if rvec is not None and tvec is not None:
        solver.rvec = rvec
        solver.tvec = tvec
        out_frame = solver.visualize_pose(
            out_frame, length=0.2, show_rotation=show_rotation
        )

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
                out_frame,
                text,
                (w - tw - 10, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
            )

    return out_frame


def main() -> None:
    """Main processing loop."""
    detector = ObjectDetector(model_path=MODEL_PATH, debug=DEBUG)
    solver = Solver(
        config_path=CONFIG_PATH, obj_width=OBJ_WIDTH, obj_length=OBJ_LENGTH
    )
    cap = cv2.VideoCapture(VIDEO_PATH)
    if sys.platform.startswith("linux"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        cap.set(cv2.CAP_PROP_FPS, 20)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, 100)

    raw_data_count, output_data_count, traj_data_count = (
        count_files_in_directory(SAVE_PATH)
    )

    raw_data_out = None
    output_data_out = None
    traj_file = None

    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        raw_data_out = cv2.VideoWriter(
            SAVE_PATH + "raw_data/" + f"raw_data_{raw_data_count}.avi",
            fourcc,
            20.0,
            (640, 480),
        )
        output_data_out = cv2.VideoWriter(
            SAVE_PATH + "output_data/" + f"output_{output_data_count}.avi",
            fourcc,
            20.0,
            (640, 480),
        )
        traj_file = open(
            SAVE_PATH + "traj_data/" + f"traj_{traj_data_count}.csv", "w"
        )
        traj_file.write("timestamp,frame_id,x,y,z,yaw,pitch,roll\n")

    if not cap.isOpened():
        print("Unable to open video:", VIDEO_PATH)
        return

    # Serial port initialization (referencing swarm_following pattern).
    ser = None
    if SERIAL_ENABLED:
        try:
            import serial
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
            time.sleep(2)  # Wait for serial port initialization
            print(f"Serial port opened: {SERIAL_PORT}")
        except Exception as e:
            print(f"Failed to open serial port: {e}")
            ser = None

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

        if SAVE_OUTPUT and raw_data_out is not None:
            raw_data_out.write(left)

        # Detection and point extraction.
        detector.detect(left)
        out_frame = left.copy()

        center_points, solver.mode, out_frame = detector.get_points(
            left, out_frame
        )

        # Pose in camera frame (target relative to camera) - used for 2D viz
        rvec, tvec = None, None
        # Pose in target frame (camera relative to target) - used for serial/log
        rvec_cam, tvec_cam = None, None
        is_valid = False

        if len(center_points) >= 4:
            success, rvec_raw, tvec_raw = solver.solver(center_points)
            if success and rvec_raw is not None and tvec_raw is not None:
                is_valid = True
                tvec = tvec_raw.flatten()
                rvec = rvec_raw.flatten()

                # Convert to camera pose in target frame for serial/log output
                rvec_cam, tvec_cam = solver.get_camera_pose_in_target_frame(
                    rvec, tvec
                )

                if tvec_cam is not None and rvec_cam is not None:
                    msg = (
                        f"[{tvec_cam[0]:.2f},{tvec_cam[1]:.2f},{tvec_cam[2]:.2f},"
                        f"{rvec_cam[0]:.2f},{rvec_cam[1]:.2f},{rvec_cam[2]:.2f},"
                        f"{solver.mode}]\r\n"
                    )
                    print("Pose (cam in target):", msg.strip())

                    # Serial output (camera pose in target frame).
                    if ser is not None:
                        try:
                            ser.write(msg.encode("utf-8"))
                        except Exception as e:
                            print(f"Serial write failed: {e}")
            else:
                # No valid pose: send zero values via serial.
                if ser is not None:
                    try:
                        zero_msg = f"[0.00,0.00,0.00,0.00,0.00,0.00,{solver.mode}]\r\n"
                        ser.write(zero_msg.encode("utf-8"))
                    except Exception as e:
                        print(f"Serial write failed: {e}")
        else:
            # Not enough points: send zero values via serial.
            if ser is not None:
                try:
                    zero_msg = f"[0.00,0.00,0.00,0.00,0.00,0.00,{solver.mode}]\r\n"
                    ser.write(zero_msg.encode("utf-8"))
                except Exception as e:
                    print(f"Serial write failed: {e}")

        # Save trajectory data (camera pose in target frame).
        if SAVE_OUTPUT and traj_file is not None:
            timestamp = time.time() - start_timestamp
            if rvec_cam is not None and tvec_cam is not None:
                traj_file.write(
                    f"{timestamp:.6f},{frame_id},"
                    f"{tvec_cam[0]:.6f},{tvec_cam[1]:.6f},{tvec_cam[2]:.6f},"
                    f"{rvec_cam[2]:.6f},{rvec_cam[1]:.6f},{rvec_cam[0]:.6f}\n"
                )
                traj_file.flush()
            else:
                traj_file.write(
                    f"{timestamp:.6f},{frame_id},0.0,0.0,0.0,0.0,0.0,0.0\n"
                )
                traj_file.flush()
            frame_id += 1

        # Visualization.
        out_frame = visualize_frame(
            frame=out_frame,
            rvec=rvec,
            tvec=tvec,
            solver=solver,
            is_valid=is_valid,
            show_rotation=(solver.mode == 1 or (solver.mode == 0 and SHOW_ROTATION_FOR_4POINT)),
        )

        if DEBUG:
            cv2.imshow("Pose Visualization", out_frame)

        if SAVE_OUTPUT and output_data_out is not None:
            output_data_out.write(out_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        end_time = time.time()
        print(f"FPS: {1 / (end_time - start_time):.2f}")

    if SAVE_OUTPUT:
        if raw_data_out is not None:
            raw_data_out.release()
        if output_data_out is not None:
            output_data_out.release()
        if traj_file is not None:
            traj_file.close()
            print(
                f"Trajectory data saved to "
                f"{SAVE_PATH}traj_data/traj_{traj_data_count}.csv"
            )

    # Close serial port.
    if ser is not None:
        ser.close()
        print("Serial port closed.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
