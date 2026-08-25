import cv2
import numpy as np
import time
import sys
import threading

from detector import ObjectDetector
from solver import Solver
from tools import count_files_in_directory
from pose_viz_live import LivePoseVisualizer
from traditional_detector import TraditionalFeatureDetector
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
    ENABLE_LIVE_VIZ,
    LIVE_VIZ_BACKEND,
    LIVE_VIZ_FPS,
    THRESH_FALLBACK_PARAMS,
    THRESH_FALLBACK_TARGET_PHYSICAL_AREA_M2,
    FALLBACK_ANGLE_FILL_VALUE,
    AUTO_SWITCH_TO_TRADITIONAL,
    CLOSE_DISTANCE_THRESHOLD_M,
    LOST_FRAMES_BEFORE_TRADITIONAL,
    TRACK_MODE,
    ENABLE_TRACK_MODE_COLUMN,
)


# Global auto-switch state.  Index 0: close-distance condition satisfied;
# index 1: enough consecutive lost frames -> switch to traditional tracking.
_switch_state = [False, False]
_lost_frame_count = 0


def visualize_frame(
    frame: np.ndarray,
    rvec: np.ndarray | None,
    tvec: np.ndarray | None,
    solver: Solver,
    is_valid: bool = True,
    show_rotation: bool = True,
    pose_text: tuple | None = None,
    mode: int = 0,
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
        pose_text: Optional ``(x, y, z, roll, pitch, yaw)`` in the docking
            output frame O (meters, degrees in ``(-180, 180]``). Passed through to
            :meth:`Solver.visualize_pose` so the on-image readout matches the
            serial/CSV output. When None, the legacy camera-frame decomposition
            is shown.
        mode: Point-mode hint for diagnostics, 0 for 4-point PnP and 1 for
            5-point PnP.  Does not affect drawing when ``rvec``/``tvec`` are
            None (fallback / track_mode=1 use text-only overlay).

    Returns:
        Output frame with pose information drawn.
    """
    out_frame = frame.copy()

    if rvec is not None and tvec is not None:
        solver.rvec = rvec
        solver.tvec = tvec
        out_frame = solver.visualize_pose(
            out_frame, length=0.2, show_rotation=show_rotation,
            pose_text=pose_text,
        )

        if DEBUG:
            h, w = out_frame.shape[:2]
            font_scale = 0.7
            thickness = 2

            # Status text (top-right)
            status_text = "Valid" if is_valid else "Invalid"
            status_color = (0, 255, 0) if is_valid else (0, 0, 255)
            text = f"Status: {status_text}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.putText(
                out_frame,
                text,
                (w - tw - 10, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                status_color,
                thickness,
            )

            # Bottom-right info block: mode + reprojection error.
            mode_text = "5-point" if mode == 1 else "4-point"
            err = getattr(solver, "_last_reproj_error", None)
            err_str = f"{err:.2f}px" if err is not None else "--"
            lines = [mode_text, f"Reproj: {err_str}"]
            line_height = 30
            for idx, line in enumerate(lines):
                (tw, th), _ = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 3
                )
                y = h - 10 - (len(lines) - 1 - idx) * line_height
                color = (0, 255, 255)
                # Draw a black outline for better readability (bold effect).
                cv2.putText(
                    out_frame,
                    line,
                    (w - tw - 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    5,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    out_frame,
                    line,
                    (w - tw - 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    3,
                    cv2.LINE_AA,
                )

    elif pose_text is not None:
        # Text-only overlay for fallback / track_mode=1 where no target-frame
        # rotation/translation is available (rvec/tvec are None).
        x_t, y_t, z_t, roll_deg, pitch_deg, yaw_deg = pose_text
        cv2.putText(out_frame, f"Yaw: {yaw_deg:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(out_frame, f"Pitch: {pitch_deg:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(out_frame, f"Roll: {roll_deg:.2f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(out_frame, f"X: {x_t:.2f}m", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(out_frame, f"Y: {y_t:.2f}m", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(out_frame, f"Z: {z_t:.2f}m", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return out_frame


def main() -> None:
    """Main processing loop."""
    global _switch_state, _lost_frame_count
    # Reset auto-switch state at the start of each run.
    _switch_state = [False, False]
    _lost_frame_count = 0

    # Only load the YOLO model when track_mode=0 (YOLO+PnP).
    detector: ObjectDetector | None = None
    if TRACK_MODE == 0:
        detector = ObjectDetector(model_path=MODEL_PATH, debug=DEBUG)
    traditional_detector = TraditionalFeatureDetector(params=THRESH_FALLBACK_PARAMS)
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
        header = "timestamp,frame_id,x,y,z,yaw,pitch,roll"
        if ENABLE_TRACK_MODE_COLUMN:
            header += ",mode"
        traj_file.write(header + "\n")
        # x,y,z in meters; yaw,pitch,roll in degrees (output frame O);
        # mode: effective_track_mode (0=YOLO+PnP, 1=traditional, may switch at runtime).

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

    # Live matplotlib visualization window
    viz = None
    if ENABLE_LIVE_VIZ:
        viz = LivePoseVisualizer(
            history_size=300, backend=LIVE_VIZ_BACKEND, fps=LIVE_VIZ_FPS
        )
        print(f"Live pose visualization enabled (backend={LIVE_VIZ_BACKEND}, fps={LIVE_VIZ_FPS}).")

    start_timestamp = time.time()
    frame_id = 0

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("VideoStream end or cannot fetch the frame.")
                break
            if frame.shape[1] == 1280:
                # Side-by-side input: use the right half as the mono view.
                # NOTE: Do NOT flip here — calibration assumes original image
                # coordinates. Flipping is only applied to the display/output.
                left = cv2.flip(frame[:, 640:1280, :], -1)
            else:
                left = frame

            if SAVE_OUTPUT and raw_data_out is not None:
                # Flip for display if configured.
                raw_data_out.write(cv2.flip(left, -1) if FLIP else left)

            # Pose state.
            out_frame = left.copy()
            rvec = tvec = None
            rvec_cam = tvec_cam = None
            roll = pitch = yaw = None
            is_valid = False
            trad_target = None
            pnp_point_mode = -1  # valid only in track_mode=0 (0=4p, 1=5p)
            effective_track_mode = (
                1
                if (AUTO_SWITCH_TO_TRADITIONAL and _switch_state[1])
                else TRACK_MODE
            )

            if effective_track_mode == 1:
                # ------------------------------------------------------------------
                # track_mode=1: traditional feature extraction (largest bright blob).
                # ------------------------------------------------------------------
                solver.mode = 1
                trad_target = traditional_detector.detect(left)
                if trad_target is not None:
                    out_frame = traditional_detector.visualize(
                        out_frame, trad_target
                    )
                    h, w = left.shape[:2]
                    ok, _, tvec_cam = solver.solve_fallback_from_centroid(
                        image_center=(w / 2.0, h / 2.0),
                        target_center=trad_target.center,
                        target_area_px=trad_target.area_px,
                        image_size=(w, h),
                        physical_area_m2=THRESH_FALLBACK_TARGET_PHYSICAL_AREA_M2,
                    )
                    if ok and tvec_cam is not None:
                        is_valid = True
                        roll = pitch = yaw = 0.0

                        msg = (
                            f"[{tvec_cam[0]:.2f},{tvec_cam[1]:.2f},{tvec_cam[2]:.2f},"
                            f"0.00,0.00,0.00,{effective_track_mode}]\r\n"
                        )
                        print("Pose (trad mode):", msg.strip())
                        if ser is not None:
                            try:
                                ser.write(msg.encode("utf-8"))
                            except Exception as e:
                                print(f"Serial write failed: {e}")
                if not is_valid and ser is not None:
                    try:
                        zero_msg = (
                            f"[0.00,0.00,0.00,0.00,0.00,0.00,{effective_track_mode}]\r\n"
                        )
                        ser.write(zero_msg.encode("utf-8"))
                    except Exception as e:
                        print(f"Serial write failed: {e}")
            else:
                # ------------------------------------------------------------------
                # track_mode=0: YOLO + PnP.
                # ------------------------------------------------------------------
                if detector is None:
                    # Should never happen because detector is created for track_mode=0.
                    raise RuntimeError(
                        "ObjectDetector is None but TRACK_MODE == 0."
                    )
                detector.detect(left)
                center_points, pnp_point_mode, out_frame = detector.get_points(
                    left, out_frame
                )

                if len(center_points) >= 4:
                    success, rvec_raw, tvec_raw = solver.solver(center_points)

                    # Diagnostic print of PnP candidate count and reprojection error.
                    err = getattr(solver, "_last_reproj_error", None)
                    err_str = f"{err:.2f}px" if err is not None else "None"
                    pnp_label = "5-point" if solver._pnp_num_points >= 5 else "4-point"
                    print(
                        f"Diag: {pnp_label}, "
                        f"candidates={solver._last_num_candidates}, "
                        f"reproj={err_str}"
                    )

                    if success and rvec_raw is not None and tvec_raw is not None:
                        tvec = tvec_raw.flatten()
                        rvec = rvec_raw.flatten()

                        # Abnormal value filter: target behind camera (z < 0)
                        if tvec[2] < 0:
                            is_valid = False
                            rvec, tvec = None, None
                            rvec_cam, tvec_cam = None, None
                        else:
                            is_valid = True
                            # Convert to camera pose in target frame + Euler angles
                            rvec_cam, tvec_cam, roll, pitch, yaw = (
                                solver.get_camera_pose_euler_in_target_frame(rvec, tvec)
                            )

                        if tvec_cam is not None and roll is not None:
                            # Angles to degrees, all in (-180,180].
                            roll_deg = np.degrees(roll)
                            pitch_deg = np.degrees(pitch)
                            yaw_deg = np.degrees(yaw)
                            msg = (
                                f"[{tvec_cam[0]:.2f},{tvec_cam[1]:.2f},{tvec_cam[2]:.2f},"
                                f"{roll_deg:.2f},{pitch_deg:.2f},{yaw_deg:.2f},"
                                f"{effective_track_mode}]\r\n"
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
                                zero_msg = f"[0.00,0.00,0.00,0.00,0.00,0.00,{effective_track_mode}]\r\n"
                                ser.write(zero_msg.encode("utf-8"))
                            except Exception as e:
                                print(f"Serial write failed: {e}")
                else:
                    # Not enough points: send zero values via serial.
                    if ser is not None:
                        try:
                            zero_msg = f"[0.00,0.00,0.00,0.00,0.00,0.00,{effective_track_mode}]\r\n"
                            ser.write(zero_msg.encode("utf-8"))
                        except Exception as e:
                            print(f"Serial write failed: {e}")

                # ------------------------------------------------------------------
                # Auto-switch to traditional tracking:
                #   condition1 = once a valid PnP pose is closer than
                #                CLOSE_DISTANCE_THRESHOLD_M.
                #   condition2 = after condition1, PnP is lost for
                #                LOST_FRAMES_BEFORE_TRADITIONAL consecutive frames.
                # When condition2 becomes True, subsequent frames use traditional
                # tracking (effective_track_mode == 1) and stay there.
                # ------------------------------------------------------------------
                if AUTO_SWITCH_TO_TRADITIONAL:
                    if (
                        is_valid
                        and tvec_cam is not None
                        and tvec_cam[2] < CLOSE_DISTANCE_THRESHOLD_M
                    ):
                        _switch_state[0] = True
                        _lost_frame_count = 0
                    elif not is_valid:
                        if _switch_state[0]:
                            _lost_frame_count += 1
                            if (
                                _lost_frame_count
                                >= LOST_FRAMES_BEFORE_TRADITIONAL
                            ):
                                _switch_state[1] = True
                                if DEBUG:
                                    print(
                                        f"Auto-switch: PnP lost for "
                                        f"{_lost_frame_count} frames, "
                                        f"switching to traditional tracking."
                                    )
                        else:
                            _lost_frame_count = 0
                    else:
                        _lost_frame_count = 0

            # Save trajectory data (camera pose in target frame).
            if SAVE_OUTPUT and traj_file is not None:
                timestamp = time.time() - start_timestamp
                if tvec_cam is not None:
                    # For track_mode=1 and fallback roll/pitch/yaw are filled with
                    # a placeholder; for PnP they are radians here.
                    yaw_deg = np.degrees(yaw) if yaw is not None else 0.0
                    pitch_deg = np.degrees(pitch) if pitch is not None else 0.0
                    roll_deg = np.degrees(roll) if roll is not None else 0.0
                    line = (
                        f"{timestamp:.6f},{frame_id},"
                        f"{tvec_cam[0]:.6f},{tvec_cam[1]:.6f},{tvec_cam[2]:.6f},"
                        f"{yaw_deg:.6f},{pitch_deg:.6f},{roll_deg:.6f}"
                    )
                    if ENABLE_TRACK_MODE_COLUMN:
                        line += f",{effective_track_mode}"
                    traj_file.write(line + "\n")
                    traj_file.flush()
                else:
                    zeros = "0.0,0.0,0.0,0.0,0.0,0.0"
                    if ENABLE_TRACK_MODE_COLUMN:
                        zeros += f",{effective_track_mode}"
                    traj_file.write(
                        f"{timestamp:.6f},{frame_id},{zeros}\n"
                    )
                    traj_file.flush()
                frame_id += 1

            # Visualization.
            # Build the on-image readout from the docking output frame O so it
            # matches the serial/CSV values exactly (angles in degrees, (-180,180]).
            pose_text = None
            if tvec_cam is not None and roll is not None:
                pose_text = (
                    tvec_cam[0], tvec_cam[1], tvec_cam[2],
                    np.degrees(roll), np.degrees(pitch), np.degrees(yaw),
                )
            out_frame = visualize_frame(
                frame=out_frame,
                rvec=rvec,
                tvec=tvec,
                solver=solver,
                is_valid=is_valid,
                show_rotation=(
                    solver._pnp_num_points >= 5
                    or (solver._pnp_num_points == 4 and SHOW_ROTATION_FOR_4POINT)
                ),
                pose_text=pose_text,
                mode=pnp_point_mode,
            )

            if DEBUG:
                cv2.imshow("Pose Visualization", out_frame)

            if SAVE_OUTPUT and output_data_out is not None:
                output_data_out.write(out_frame)

            # Update live visualization window
            if viz is not None:
                viz.update(
                    tvec_cam, rvec_cam, is_valid=is_valid, roll=roll, pitch=pitch,
                    yaw=yaw, mode=solver.mode,
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if viz is not None and viz.should_quit():
                print("Visualization window closed, exiting...")
                break

            end_time = time.time()
            print(f"FPS: {1 / (end_time - start_time):.2f}")

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C), shutting down...")
    finally:
        def _cleanup(name: str, fn) -> None:
            t0 = time.time()
            try:
                fn()
            except Exception as e:
                print(f"Cleanup error ({name}): {e}")
            print(f"Cleanup [{name}] took {(time.time() - t0) * 1000:.1f} ms")

        if SAVE_OUTPUT:
            if raw_data_out is not None:
                _cleanup("raw_data_out.release", raw_data_out.release)
            if output_data_out is not None:
                _cleanup("output_data_out.release", output_data_out.release)
            if traj_file is not None:
                _cleanup("traj_file.close", traj_file.close)
                print(
                    f"Trajectory data saved to "
                    f"{SAVE_PATH}traj_data/traj_{traj_data_count}.csv"
                )

        # Close serial port.
        if ser is not None:
            _cleanup("serial.close", ser.close)
            print("Serial port closed.")

        # Release video capture before tearing down GUI windows to reduce
        # cross-backend contention on Windows.
        _cleanup("cap.release", cap.release)

        # Destroy only the OpenCV window we created; destroyAllWindows() can
        # stall the Windows message queue if the Open3D/Filament context is
        # still active in the background GUI thread.
        if DEBUG:
            _cleanup("cv2.destroyWindow", lambda: cv2.destroyWindow("Pose Visualization"))

        # Close live visualization.
        if viz is not None:
            _cleanup("viz.close", viz.close)

        print("Shutdown complete.")
        print("Active threads:", threading.enumerate())


if __name__ == "__main__":
    main()
