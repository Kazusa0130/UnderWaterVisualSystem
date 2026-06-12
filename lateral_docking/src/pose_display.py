"""Real-time camera pose display utility (camera-in-target-frame).

This module provides plug-and-play functions/classes for visualizing
the camera's 6-DOF pose relative to the target plane in real-time.

Usage in main.py:
    from pose_display import draw_camera_pose_overlay, CameraPoseDisplay

    # Simple overlay (returns annotated frame)
    out_frame = draw_camera_pose_overlay(frame, tvec_cam, rvec_cam, is_valid)

    # Enhanced display with history plots (manages its own window)
    display = CameraPoseDisplay(history_size=300)
    display.update(tvec_cam, rvec_cam, is_valid)
    display.show()

Author: BJTU Underwater Robotics Team
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PANEL_BG_COLOR = (30, 30, 30)
_PANEL_BORDER_COLOR = (100, 100, 100)
_TEXT_COLOR_VALID = (0, 255, 0)
_TEXT_COLOR_INVALID = (0, 0, 255)
_TEXT_COLOR_LABEL = (200, 200, 200)
_AXIS_COLORS = {
    "x": (0, 0, 255),  # Red
    "y": (0, 255, 0),  # Green
    "z": (255, 0, 0),  # Blue
}


# ---------------------------------------------------------------------------
# Helper: draw a rounded-rectangle panel on the frame
# ---------------------------------------------------------------------------
def _draw_panel(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    bg_color: tuple[int, int, int] = _PANEL_BG_COLOR,
    border_color: tuple[int, int, int] = _PANEL_BORDER_COLOR,
    alpha: float = 0.6,
) -> None:
    """Draw a semi-transparent info panel."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), border_color, 1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Put text with a subtle black outline for readability."""
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


# ---------------------------------------------------------------------------
# Simple overlay function (plug-and-play)
# ---------------------------------------------------------------------------
def draw_camera_pose_overlay(
    frame: np.ndarray,
    tvec_cam: np.ndarray | None,
    rvec_cam: np.ndarray | None,
    is_valid: bool = True,
    mode: int = 0,
    fps: Optional[float] = None,
    roll: Optional[float] = None,
    pitch: Optional[float] = None,
    yaw: Optional[float] = None,
) -> np.ndarray:
    """Draw a compact HUD overlay showing camera pose in target frame.

    Args:
        frame: Input BGR image.
        tvec_cam: Camera position in target frame (3,) — [x, y, z].
        rvec_cam: Camera rotation in target frame (3,) — used for mini axes.
        is_valid: Whether the pose is valid (affects color).
        mode: PnP mode (0=4-point, 1=5-point) shown in corner.
        fps: Optional FPS value to display.
        roll: Camera roll in target frame (radians) for text display.
        pitch: Camera pitch in target frame (radians) for text display.
        yaw: Camera yaw in target frame (radians) for text display.

    Returns:
        Annotated frame (modified in-place; copy beforehand if needed).
    """
    h, w = frame.shape[:2]
    color = _TEXT_COLOR_VALID if is_valid else _TEXT_COLOR_INVALID

    # --- Top-left: pose data panel -----------------------------------------
    panel_w, panel_h = 280, 170 if fps is None else 195
    _draw_panel(frame, 10, 10, panel_w, panel_h)

    lines: list[tuple[str, tuple[int, int, int]]] = [
        ("Camera Pose in Target Frame", _TEXT_COLOR_LABEL),
    ]

    if is_valid and tvec_cam is not None and rvec_cam is not None and roll is not None:
        x, y, z = tvec_cam.flatten()
        lines.extend([
            (f"X:   {x:+.3f} m", _AXIS_COLORS["x"]),
            (f"Y:   {y:+.3f} m", _AXIS_COLORS["y"]),
            (f"Z:   {z:+.3f} m", _AXIS_COLORS["z"]),
            (f"Roll:  {np.degrees(roll):>+7.2f} deg", _AXIS_COLORS["x"]),
            (f"Pitch: {np.degrees(pitch):>+7.2f} deg", _AXIS_COLORS["y"]),
            (f"Yaw:   {np.degrees(yaw):>+7.2f} deg", _AXIS_COLORS["z"]),
        ])
    else:
        lines.append(("--- INVALID ---", _TEXT_COLOR_INVALID))

    if fps is not None:
        lines.append((f"FPS: {fps:.1f}", _TEXT_COLOR_LABEL))

    y_off = 35
    for txt, c in lines:
        _put_text(frame, txt, (20, y_off), c, font_scale=0.55, thickness=1)
        y_off += 22

    # --- Top-right: mini 3D axes (camera orientation hint) ----------------
    if is_valid and rvec_cam is not None:
        _draw_mini_axes(frame, w - 90, 60, rvec_cam, scale=35)

    # --- Bottom-right: mode badge ------------------------------------------
    mode_text = "MODE: 5P" if mode == 1 else "MODE: 4P"
    (tw, th), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    _draw_panel(frame, w - tw - 25, h - th - 25, tw + 15, th + 15, alpha=0.5)
    _put_text(frame, mode_text, (w - tw - 18, h - th - 5), _TEXT_COLOR_LABEL, font_scale=0.6, thickness=2)

    return frame


def _draw_mini_axes(
    img: np.ndarray,
    cx: int,
    cy: int,
    rvec: np.ndarray,
    scale: float = 30.0,
) -> None:
    """Draw a small 3-axis orientation widget at (cx, cy)."""
    R, _ = cv2.Rodrigues(np.array(rvec).flatten())
    # Project unit axes onto a virtual image plane (simple orthographic)
    axes = np.eye(3) * scale
    # Rotate axes by camera rotation in target frame
    axes_rot = (R @ axes.T).T

    # Draw Z axis (forward) with thicker line
    for i, (axis, color_name) in enumerate(zip(axes_rot, ["x", "y", "z"])):
        thickness = 2 if color_name == "z" else 1
        end = (int(cx + axis[0]), int(cy - axis[1]))  # flip Y for image coords
        cv2.line(img, (cx, cy), end, _AXIS_COLORS[color_name], thickness)
        # Small arrow head
        cv2.circle(img, end, 2, _AXIS_COLORS[color_name], -1)

    cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1)


# ---------------------------------------------------------------------------
# Enhanced real-time display with history plots
# ---------------------------------------------------------------------------
class CameraPoseDisplay:
    """Enhanced real-time display with history plots and 3D view.

    Manages its own OpenCV window and internal state. Call ``update()``
    every frame and ``show()`` to refresh the display.

    Example::

        display = CameraPoseDisplay(history_size=200)
        while True:
            # ... get pose ...
            display.update(tvec_cam, rvec_cam, is_valid)
            display.show()
            if display.should_quit():
                break
    """

    def __init__(
        self,
        history_size: int = 300,
        window_name: str = "Camera Pose in Target Frame",
        show_history: bool = True,
        show_3d_view: bool = True,
    ) -> None:
        self.history_size = history_size
        self.window_name = window_name
        self.show_history = show_history
        self.show_3d_view = show_3d_view

        # Circular buffers for history
        self.timestamps: deque[float] = deque(maxlen=history_size)
        self.x_hist: deque[float] = deque(maxlen=history_size)
        self.y_hist: deque[float] = deque(maxlen=history_size)
        self.z_hist: deque[float] = deque(maxlen=history_size)
        self.roll_hist: deque[float] = deque(maxlen=history_size)
        self.pitch_hist: deque[float] = deque(maxlen=history_size)
        self.yaw_hist: deque[float] = deque(maxlen=history_size)
        self.valid_hist: deque[bool] = deque(maxlen=history_size)

        self._start_time = time.time()
        self._last_key = -1
        self.current_rvec: Optional[np.ndarray] = None

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

    def update(
        self,
        tvec_cam: np.ndarray | None,
        rvec_cam: np.ndarray | None,
        is_valid: bool = True,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
    ) -> None:
        """Push a new pose sample into the history buffers."""
        t = time.time() - self._start_time
        self.timestamps.append(t)
        self.valid_hist.append(is_valid)

        if is_valid and tvec_cam is not None and rvec_cam is not None and roll is not None:
            x, y, z = tvec_cam.flatten()
            self.x_hist.append(x)
            self.y_hist.append(y)
            self.z_hist.append(z)
            self.roll_hist.append(np.degrees(roll))
            self.pitch_hist.append(np.degrees(pitch))
            self.yaw_hist.append(np.degrees(yaw))
            self.current_rvec = np.array(rvec_cam).flatten()
        else:
            # Repeat last valid value or NaN
            nan = float("nan")
            self.x_hist.append(nan)
            self.y_hist.append(nan)
            self.z_hist.append(nan)
            self.roll_hist.append(nan)
            self.pitch_hist.append(nan)
            self.yaw_hist.append(nan)
            self.current_rvec = None

    def should_quit(self) -> bool:
        """Return True if user pressed 'q' or ESC in the display window."""
        return self._last_key in (ord("q"), ord("Q"), 27)

    def _render_history_plot(
        self,
        width: int = 400,
        height: int = 200,
    ) -> np.ndarray:
        """Render position/angle history as a multi-line plot."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        if len(self.x_hist) < 2:
            _put_text(canvas, "Collecting data...", (10, 30), _TEXT_COLOR_LABEL)
            return canvas

        def _draw_channel(
            data: deque[float],
            color: tuple[int, int, int],
            y_offset: float,
            scale: float,
            label: str,
        ) -> None:
            arr = np.array(data, dtype=float)
            valid = ~np.isnan(arr)
            if np.sum(valid) < 2:
                return
            t = np.arange(len(arr))
            ys = y_offset - arr * scale
            # Clip to canvas
            ys = np.clip(ys, 5, height - 5)
            pts = np.column_stack([t * (width / len(arr)), ys]).astype(np.int32)
            # Only draw valid segments
            for i in range(len(pts) - 1):
                if valid[i] and valid[i + 1]:
                    cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), color, 1)
            _put_text(canvas, label, (5, int(y_offset - 10)), color, font_scale=0.4)

        # Position channels (top half)
        _draw_channel(self.x_hist, _AXIS_COLORS["x"], height * 0.25, 50.0, "X")
        _draw_channel(self.y_hist, _AXIS_COLORS["y"], height * 0.25, 50.0, "Y")
        _draw_channel(self.z_hist, _AXIS_COLORS["z"], height * 0.25, 50.0, "Z")

        # Angle channels (bottom half)
        _draw_channel(self.roll_hist, _AXIS_COLORS["x"], height * 0.75, 0.8, "R")
        _draw_channel(self.pitch_hist, _AXIS_COLORS["y"], height * 0.75, 0.8, "P")
        _draw_channel(self.yaw_hist, _AXIS_COLORS["z"], height * 0.75, 0.8, "Y")

        cv2.line(canvas, (0, height // 2), (width, height // 2), (60, 60, 60), 1)
        return canvas

    def _render_3d_view(
        self,
        width: int = 400,
        height: int = 400,
    ) -> np.ndarray:
        """Render a simple 3D perspective of camera relative to target plane."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        cx, cy = width // 2, height // 2 + 40

        # Draw target plane (horizontal line for simplicity)
        cv2.line(canvas, (50, cy), (width - 50, cy), (0, 255, 255), 2)
        _put_text(canvas, "Target Plane (Z=0)", (width // 2 - 60, cy - 10), (0, 255, 255), font_scale=0.4)

        # Draw target axes at origin
        axis_len = 40
        cv2.arrowedLine(canvas, (cx, cy), (cx + axis_len, cy), _AXIS_COLORS["x"], 2, tipLength=0.2)
        cv2.arrowedLine(canvas, (cx, cy), (cx, cy - axis_len), _AXIS_COLORS["y"], 2, tipLength=0.2)
        _put_text(canvas, "X", (cx + axis_len + 5, cy), _AXIS_COLORS["x"], font_scale=0.5)
        _put_text(canvas, "Y", (cx, cy - axis_len - 5), _AXIS_COLORS["y"], font_scale=0.5)

        if len(self.x_hist) == 0 or not self.valid_hist[-1] or self.current_rvec is None:
            _put_text(canvas, "No valid pose", (10, 30), _TEXT_COLOR_INVALID)
            return canvas

        # Get latest pose
        x = self.x_hist[-1]
        y = self.y_hist[-1]
        z = self.z_hist[-1]
        roll_deg = self.roll_hist[-1]
        pitch_deg = self.pitch_hist[-1]
        yaw_deg = self.yaw_hist[-1]

        # Simple perspective projection: X right, Y up, Z into screen
        # Scale: 1 meter = 100 pixels
        scale = 80.0
        cam_px = int(cx + x * scale)
        cam_py = int(cy - y * scale - z * scale * 0.3)  # Z gives slight vertical offset for depth cue

        # Draw camera frustum / body
        body_w, body_h = 20, 14
        pt1 = (cam_px - body_w // 2, cam_py - body_h // 2)
        pt2 = (cam_px + body_w // 2, cam_py + body_h // 2)
        cv2.rectangle(canvas, pt1, pt2, (255, 255, 255), 2)
        cv2.circle(canvas, (cam_px, cam_py), 3, (0, 255, 0), -1)

        # Draw line from target origin to camera
        cv2.line(canvas, (cx, cy), (cam_px, cam_py), (100, 100, 100), 1, cv2.LINE_AA)

        # Draw camera orientation axes (projected)
        R, _ = cv2.Rodrigues(self.current_rvec)
        axes = np.eye(3) * 25.0
        axes_img = (R @ axes.T).T
        for i, (axis, color_name) in enumerate(zip(axes_img, ["x", "y", "z"])):
            end = (int(cam_px + axis[0]), int(cam_py - axis[1]))
            cv2.line(canvas, (cam_px, cam_py), end, _AXIS_COLORS[color_name], 2)

        # Info text near camera icon
        info_lines = [
            f"x:{x:+.2f} y:{y:+.2f} z:{z:+.2f}",
            f"r:{roll_deg:+.1f} p:{pitch_deg:+.1f} y:{yaw_deg:+.1f}",
        ]
        tx = min(max(cam_px - 80, 5), width - 200)
        ty = max(cam_py - 40, 20)
        for line in info_lines:
            _put_text(canvas, line, (tx, ty), (255, 255, 255), font_scale=0.4)
            ty += 14

        return canvas

    def show(self) -> None:
        """Refresh the display window. Must be called after ``update()``."""
        h, w = 720, 1280
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        # Left side: current numeric readout
        panel_w = 450
        _draw_panel(canvas, 10, 10, panel_w, h - 20)
        _put_text(canvas, "Camera Pose in Target Frame", (20, 40), (255, 255, 255), font_scale=0.7, thickness=2)

        y = 80
        if len(self.x_hist) > 0 and self.valid_hist[-1]:
            x, y_pos, z = self.x_hist[-1], self.y_hist[-1], self.z_hist[-1]
            r, p, yaw = self.roll_hist[-1], self.pitch_hist[-1], self.yaw_hist[-1]
            lines = [
                ("Position", None),
                (f"  X:     {x:+.4f} m", _AXIS_COLORS["x"]),
                (f"  Y:     {y_pos:+.4f} m", _AXIS_COLORS["y"]),
                (f"  Z:     {z:+.4f} m", _AXIS_COLORS["z"]),
                ("", None),
                ("Rotation", None),
                (f"  Roll:  {r:+.2f} deg", _AXIS_COLORS["x"]),
                (f"  Pitch: {p:+.2f} deg", _AXIS_COLORS["y"]),
                (f"  Yaw:   {yaw:+.2f} deg", _AXIS_COLORS["z"]),
            ]
        else:
            lines = [("--- NO VALID POSE ---", _TEXT_COLOR_INVALID)]

        for txt, color in lines:
            c = color if color is not None else _TEXT_COLOR_LABEL
            _put_text(canvas, txt, (20, y), c, font_scale=0.6, thickness=1)
            y += 26

        # Right side: optional sub-views
        sub_x = panel_w + 30
        sub_y = 10

        if self.show_3d_view:
            view = self._render_3d_view(width=w - sub_x - 20, height=350)
            canvas[sub_y : sub_y + view.shape[0], sub_x : sub_x + view.shape[1]] = view
            sub_y += view.shape[0] + 10

        if self.show_history:
            hist = self._render_history_plot(width=w - sub_x - 20, height=320)
            canvas[sub_y : sub_y + hist.shape[0], sub_x : sub_x + hist.shape[1]] = hist

        # Footer help
        _put_text(canvas, "Press 'q' or ESC to quit", (20, h - 15), (120, 120, 120), font_scale=0.45)

        cv2.imshow(self.window_name, canvas)
        self._last_key = cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        """Destroy the display window."""
        cv2.destroyWindow(self.window_name)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Quick standalone test with synthetic pose data."""
    import math

    print("Standalone test: synthetic camera pose display")
    print("Press 'q' or ESC to quit")

    display = CameraPoseDisplay(history_size=200)
    t0 = time.time()

    while not display.should_quit():
        t = time.time() - t0
        # Synthetic orbital motion
        radius = 1.5
        tvec = np.array([
            radius * math.cos(t * 0.5),
            radius * math.sin(t * 0.3) * 0.3,
            2.0 + 0.5 * math.sin(t * 0.8),
        ])
        rvec = np.array([
            math.radians(5 * math.sin(t * 1.2)),
            math.radians(10 * math.sin(t * 0.7)),
            math.radians(30 * math.sin(t * 0.5)),
        ])
        is_valid = True

        # In this synthetic test rvec approximates small Euler angles
        display.update(tvec, rvec, is_valid, roll=rvec[0], pitch=rvec[1], yaw=rvec[2])
        display.show()
        time.sleep(0.033)  # ~30 FPS

    display.close()
    print("Test finished.")
