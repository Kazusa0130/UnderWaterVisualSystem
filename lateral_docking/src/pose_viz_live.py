"""Real-time pose visualization using matplotlib.

This module provides a live camera-pose window for ``lateral_docking``.
It uses matplotlib for both the 3D scene and the 2D history plots.

Public API::

    viz = LivePoseVisualizer(history_size=300)
    viz.update(tvec_cam, rvec_cam, is_valid=True, roll=..., pitch=..., yaw=..., mode=...)
    if viz.should_quit():
        break
    viz.close()

Author: BJTU Underwater Robotics Team
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Visual style constants
# ---------------------------------------------------------------------------
COLOR_X_AXIS = "red"
COLOR_Y_AXIS = "green"
COLOR_Z_AXIS = "blue"
COLOR_CAMERA_X_AXIS = "orange"
COLOR_CAMERA_Y_AXIS = "cyan"
COLOR_CAMERA_Z_AXIS = "purple"
# Trajectory color by solver mode
COLOR_TRAJECTORY_4POINT = "darkorange"
COLOR_TRAJECTORY_5POINT = "green"
COLOR_TRAJECTORY_INVALID = "gray"
COLOR_START_POINT = "green"
COLOR_END_POINT = "red"
COLOR_TARGET_PLANE = "cyan"
COLOR_CAMERA = "orange"

COLOR_MODE_4POINT = "darkorange"
COLOR_MODE_5POINT = "green"

# Target corners in physical target frame (Y-up, Z-toward-camera)
DEFAULT_OBJ_CORNERS = np.array(
    [
        [-0.100, 0.300, 0.000],  # 左上
        [0.100, 0.300, 0.000],  # 右上
        [0.245, -0.300, 0.000],  # 右下
        [-0.245, -0.300, 0.000],  # 左下
    ]
)

HISTORY_SIZE = 300
FRUSTUM_SCALE = 0.06
FRUSTUM_FOCAL = 0.10 * FRUSTUM_SCALE / 0.08
TARGET_AXIS_LENGTH = 0.3
CAMERA_AXIS_LENGTH = 0.12
CAMERA_AXIS_OFFSET = FRUSTUM_FOCAL

# Line widths for visual hierarchy (thicker = more important / less occluding)
LINE_WIDTH_TARGET_AXIS = 2.0
LINE_WIDTH_TARGET_EDGES = 1.5
LINE_WIDTH_TRAJECTORY = 1.5
LINE_WIDTH_CAMERA_AXIS = 1.2
LINE_WIDTH_CAMERA_FRUSTUM = 1.0

# Transparency for dynamic geometry to reduce occlusion
ALPHA_TRAJECTORY = 0.7
ALPHA_CAMERA_FRUSTUM = 0.6
ALPHA_TARGET_EDGES = 0.8


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------
class LivePoseVisualizer:
    """Live pose visualizer using matplotlib."""

    def __init__(
        self,
        history_size: int = HISTORY_SIZE,
        backend: Optional[str] = None,
        fps: float = 15.0,
    ) -> None:
        """Create the visualizer.

        Args:
            history_size: Maximum number of historical pose samples to keep.
            backend: Kept for backward compatibility; ignored. Matplotlib is
                always used.
            fps: Target refresh rate for the live window.
        """
        del backend  # matplotlib is the only backend
        self._impl = _MatplotlibLivePoseVisualizer(history_size=history_size, fps=fps)

    def update(
        self,
        tvec_cam: Optional[np.ndarray],
        rvec_cam: Optional[np.ndarray],
        is_valid: bool = True,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        mode: Optional[int] = None,
    ) -> None:
        """Push a new pose sample and refresh the window."""
        self._impl.update(
            tvec_cam=tvec_cam,
            rvec_cam=rvec_cam,
            is_valid=is_valid,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            mode=mode,
        )

    def should_quit(self) -> bool:
        """Return ``True`` if the user closed the window."""
        return self._impl.should_quit()

    def close(self) -> None:
        """Destroy the figure window."""
        self._impl.close()


# ---------------------------------------------------------------------------
# Matplotlib implementation
# ---------------------------------------------------------------------------
class _MatplotlibLivePoseVisualizer:
    """Live pose visualizer using an optimized matplotlib backend."""

    def __init__(self, history_size: int = HISTORY_SIZE, fps: float = 15.0) -> None:
        import matplotlib.pyplot as plt

        self.history_size = history_size
        self.fps = max(1.0, fps)
        self._min_interval = 1.0 / self.fps
        self._last_draw = 0.0
        self._closed = False

        # Per-mode history: separate trajectories and rotation/position series
        # for 4-point and 5-point PnP so they can be distinguished visually.
        self.x_hist_4: deque[float] = deque(maxlen=history_size)
        self.y_hist_4: deque[float] = deque(maxlen=history_size)
        self.z_hist_4: deque[float] = deque(maxlen=history_size)
        self.roll_hist_4: deque[float] = deque(maxlen=history_size)
        self.pitch_hist_4: deque[float] = deque(maxlen=history_size)
        self.yaw_hist_4: deque[float] = deque(maxlen=history_size)
        self.valid_hist_4: deque[bool] = deque(maxlen=history_size)

        self.x_hist_5: deque[float] = deque(maxlen=history_size)
        self.y_hist_5: deque[float] = deque(maxlen=history_size)
        self.z_hist_5: deque[float] = deque(maxlen=history_size)
        self.roll_hist_5: deque[float] = deque(maxlen=history_size)
        self.pitch_hist_5: deque[float] = deque(maxlen=history_size)
        self.yaw_hist_5: deque[float] = deque(maxlen=history_size)
        self.valid_hist_5: deque[bool] = deque(maxlen=history_size)

        self.current_rvec: Optional[np.ndarray] = None
        self._mode: Optional[int] = None

        self._history_update_interval = 10  # Update 2D history every N redraws
        self._redraw_count = 0
        self._render_times: deque[float] = deque(maxlen=30)

        plt.ion()
        self.fig = plt.figure(figsize=(9, 5), dpi=72)
        self.fig.canvas.manager.set_window_title("Live Camera Pose in Target Frame")
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.ax3d = self.fig.add_subplot(1, 2, 1, projection="3d")
        self.ax_pos = self.fig.add_subplot(2, 2, 2)
        self.ax_rot = self.fig.add_subplot(2, 2, 4)

        self._setup_3d_axes()
        self._init_3d_artists()
        self._init_history_artists()
        self.fig.tight_layout()

    def _on_close(self, event) -> None:
        self._closed = True

    def _setup_3d_axes(self) -> None:
        self.ax3d.set_xlim([-3, 3])
        self.ax3d.set_ylim([-3, 3])
        self.ax3d.set_zlim([-1, 5])
        self.ax3d.set_box_aspect([1, 1, 1])
        self.ax3d.set_xlabel("X (m)")
        self.ax3d.set_ylabel("Y (m)")
        self.ax3d.set_zlabel("Z (m)")
        self.ax3d.set_title("3D View (Target as Origin, Physical: Y-up, Z-toward-cam)")
        self.ax3d.view_init(elev=20, azim=-60)

    def _init_3d_artists(self) -> None:
        """Create 3D artists once and reuse them every frame."""
        origin = [0.0, 0.0, 0.0]
        axis_len = TARGET_AXIS_LENGTH

        # Static target axes at origin
        self._target_x = self.ax3d.plot3D(
            [origin[0], axis_len], [origin[1], origin[1]], [origin[2], origin[2]],
            color=COLOR_X_AXIS, linewidth=LINE_WIDTH_TARGET_AXIS,
        )[0]
        self._target_y = self.ax3d.plot3D(
            [origin[0], origin[0]], [origin[1], axis_len], [origin[2], origin[2]],
            color=COLOR_Y_AXIS, linewidth=LINE_WIDTH_TARGET_AXIS,
        )[0]
        self._target_z = self.ax3d.plot3D(
            [origin[0], origin[0]], [origin[1], origin[1]], [origin[2], axis_len],
            color=COLOR_Z_AXIS, linewidth=LINE_WIDTH_TARGET_AXIS,
        )[0]

        # Static target plane edges
        edge_x, edge_y, edge_z = self._build_edge_line(DEFAULT_OBJ_CORNERS, [[0, 1], [1, 2], [2, 3], [3, 0]])
        self._target_edges = self.ax3d.plot3D(
            edge_x, edge_y, edge_z,
            color=COLOR_TARGET_PLANE,
            linewidth=LINE_WIDTH_TARGET_EDGES,
            alpha=ALPHA_TARGET_EDGES,
        )[0]

        # Dynamic trajectory (one line per mode)
        self._trajectory_4 = self.ax3d.plot3D(
            [0, 0], [0, 0], [0, 0],
            color=COLOR_TRAJECTORY_4POINT,
            linewidth=LINE_WIDTH_TRAJECTORY,
            alpha=ALPHA_TRAJECTORY,
            label="4-point",
        )[0]
        self._trajectory_5 = self.ax3d.plot3D(
            [0, 0], [0, 0], [0, 0],
            color=COLOR_TRAJECTORY_5POINT,
            linewidth=LINE_WIDTH_TRAJECTORY,
            alpha=ALPHA_TRAJECTORY,
            label="5-point",
        )[0]

        # Mode legend text handles (created once, updated on mode switch)
        self._legend_text = self.ax3d.text2D(
            0.02, 0.90, "", transform=self.ax3d.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            color="black",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Dynamic camera marker (small cross, faster than scatter3D)
        self._camera_marker = self.ax3d.plot3D(
            [0, 0], [0, 0], [0, 0],
            color=COLOR_CAMERA, linewidth=2.0,
        )[0]

        # Dynamic camera axes
        self._camera_x = self.ax3d.plot3D([0, 0], [0, 0], [0, 0], color=COLOR_CAMERA_X_AXIS, linewidth=LINE_WIDTH_CAMERA_AXIS)[0]
        self._camera_y = self.ax3d.plot3D([0, 0], [0, 0], [0, 0], color=COLOR_CAMERA_Y_AXIS, linewidth=LINE_WIDTH_CAMERA_AXIS)[0]
        self._camera_z = self.ax3d.plot3D([0, 0], [0, 0], [0, 0], color=COLOR_CAMERA_Z_AXIS, linewidth=LINE_WIDTH_CAMERA_AXIS)[0]

        # Dynamic camera frustum
        self._camera_frustum = self.ax3d.plot3D(
            [0, 0], [0, 0], [0, 0],
            color="black",
            linewidth=LINE_WIDTH_CAMERA_FRUSTUM,
            alpha=ALPHA_CAMERA_FRUSTUM,
        )[0]

        # Dynamic info text (top-left) and mode indicator (bottom-right)
        self._info_text = self.ax3d.text2D(
            0.02, 0.98, "", transform=self.ax3d.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            color="black",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        self._mode_text = self.ax3d.text2D(
            0.98, 0.02, "Mode: --", transform=self.ax3d.transAxes, fontsize=11,
            horizontalalignment="right", verticalalignment="bottom", fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # self.ax3d.legend(loc="upper left", fontsize=8)

    def _init_history_artists(self) -> None:
        idx0 = [0]
        self._pos_lines = [
            self.ax_pos.plot(idx0, [0], color=COLOR_X_AXIS, label="X", linewidth=1.2)[0],
            self.ax_pos.plot(idx0, [0], color=COLOR_Y_AXIS, label="Y", linewidth=1.2)[0],
            self.ax_pos.plot(idx0, [0], color=COLOR_Z_AXIS, label="Z", linewidth=1.2)[0],
        ]
        self._rot_lines = [
            self.ax_rot.plot(idx0, [0], color=COLOR_X_AXIS, label="Roll", linewidth=1.2)[0],
            self.ax_rot.plot(idx0, [0], color=COLOR_Y_AXIS, label="Pitch", linewidth=1.2)[0],
            self.ax_rot.plot(idx0, [0], color=COLOR_Z_AXIS, label="Yaw", linewidth=1.2)[0],
        ]
        self.ax_pos.set_ylabel("Position (m)")
        self.ax_pos.set_title("Position History")
        self.ax_pos.grid(True)
        self.ax_pos.legend(loc="upper right", fontsize=8)
        self.ax_rot.set_xlabel("Frame")
        self.ax_rot.set_ylabel("Angle (deg)")
        self.ax_rot.set_title("Rotation History")
        self.ax_rot.grid(True)
        self.ax_rot.legend(loc="upper right", fontsize=8)

    def update(
        self,
        tvec_cam: Optional[np.ndarray],
        rvec_cam: Optional[np.ndarray],
        is_valid: bool = True,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        mode: Optional[int] = None,
    ) -> None:
        if self._closed:
            return

        if mode is not None:
            self._mode = int(mode)

        # Append to the appropriate per-mode history.
        if self._mode == 1:
            hist = (self.x_hist_5, self.y_hist_5, self.z_hist_5,
                    self.roll_hist_5, self.pitch_hist_5, self.yaw_hist_5,
                    self.valid_hist_5)
        else:
            hist = (self.x_hist_4, self.y_hist_4, self.z_hist_4,
                    self.roll_hist_4, self.pitch_hist_4, self.yaw_hist_4,
                    self.valid_hist_4)
        (x_h, y_h, z_h, r_h, p_h, y_h2, v_h) = hist

        v_h.append(is_valid)
        if (
            is_valid
            and tvec_cam is not None
            and roll is not None
        ):
            x, y, z = np.asarray(tvec_cam).flatten()
            x_h.append(float(x))
            y_h.append(float(y))
            z_h.append(float(z))
            r_h.append(float(np.degrees(roll)))
            p_h.append(float(np.degrees(pitch)))
            y_h2.append(float(np.degrees(yaw)))
            self.current_rvec = (
                np.array(rvec_cam).flatten() if rvec_cam is not None else None
            )
        else:
            nan = float("nan")
            x_h.append(nan)
            y_h.append(nan)
            z_h.append(nan)
            r_h.append(nan)
            p_h.append(nan)
            y_h2.append(nan)
            self.current_rvec = None

        now = time.time()
        if now - self._last_draw < self._min_interval:
            return
        self._last_draw = now
        self._redraw()

    def should_quit(self) -> bool:
        return self._closed

    def close(self) -> None:
        import matplotlib.pyplot as plt

        if not self._closed:
            plt.close(self.fig)
            self._closed = True

    def _redraw(self) -> None:
        t0 = time.time()
        self._draw_3d()
        self._redraw_count += 1
        if self._redraw_count % self._history_update_interval == 0:
            self._draw_history()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._render_times.append(time.time() - t0)

    def _draw_3d(self) -> None:
        # Trajectory: draw one line for each mode, omitting each mode's most
        # recent valid point so it does not overlap the camera icon. Display the
        # last 50 valid points per mode to keep rendering fast.
        TRAJ_DISPLAY_LEN = 50

        def _update_trajectory_line(line, x_hist, y_hist, z_hist):
            if len(x_hist) > 1:
                x_arr = np.array(x_hist, dtype=float)
                y_arr = np.array(y_hist, dtype=float)
                z_arr = np.array(z_hist, dtype=float)
                valid = ~(np.isnan(x_arr) | np.isnan(y_arr) | np.isnan(z_arr))
                valid_indices = np.nonzero(valid)[0]
                if valid_indices.size > 0:
                    valid[valid_indices[-1]] = False
                if np.any(valid):
                    x_show = x_arr[valid]
                    y_show = y_arr[valid]
                    z_show = z_arr[valid]
                    if len(x_show) > TRAJ_DISPLAY_LEN:
                        x_show = x_show[-TRAJ_DISPLAY_LEN:]
                        y_show = y_show[-TRAJ_DISPLAY_LEN:]
                        z_show = z_show[-TRAJ_DISPLAY_LEN:]
                    line.set_data_3d(x_show, y_show, z_show)
                    return
            line.set_data_3d([0, 0], [0, 0], [0, 0])

        _update_trajectory_line(
            self._trajectory_4, self.x_hist_4, self.y_hist_4, self.z_hist_4
        )
        _update_trajectory_line(
            self._trajectory_5, self.x_hist_5, self.y_hist_5, self.z_hist_5
        )

        # Active-mode legend / current sample source
        mode_label = "5-point" if self._mode == 1 else "4-point"
        self._legend_text.set_text(
            f"Trajectory:\n4-point {COLOR_TRAJECTORY_4POINT}\n"
            f"5-point {COLOR_TRAJECTORY_5POINT}\n\n"
            f"Current: {mode_label}"
        )

        # Current camera: sample from the active-mode history.
        if self._mode == 1:
            x_hist = self.x_hist_5
            y_hist = self.y_hist_5
            z_hist = self.z_hist_5
            roll_hist = self.roll_hist_5
            pitch_hist = self.pitch_hist_5
            yaw_hist = self.yaw_hist_5
        else:
            x_hist = self.x_hist_4
            y_hist = self.y_hist_4
            z_hist = self.z_hist_4
            roll_hist = self.roll_hist_4
            pitch_hist = self.pitch_hist_4
            yaw_hist = self.yaw_hist_4

        has_current = (
            len(x_hist) > 0
            and not np.isnan(x_hist[-1])
        )
        if has_current:
            x = float(x_hist[-1])
            y = float(y_hist[-1])
            z = float(z_hist[-1])
            t = np.array([x, y, z])
            m = 0.03  # cross half-length
            self._camera_marker.set_data_3d(
                [x - m, x + m, float("nan"), x, x, float("nan"), x, x],
                [y, y, float("nan"), y - m, y + m, float("nan"), y, y],
                [z, z, float("nan"), z, z, float("nan"), z - m, z + m],
            )

            if self.current_rvec is not None:
                R, _ = cv2.Rodrigues(self.current_rvec)
                axis_len = CAMERA_AXIS_LENGTH
                # Anchor axes at the frustum base so they do not pierce the frustum.
                axis_origin = t + R[:, 2] * CAMERA_AXIS_OFFSET
                ox, oy, oz = axis_origin

                self._camera_x.set_data_3d(
                    [ox, ox + R[0, 0] * axis_len],
                    [oy, oy + R[1, 0] * axis_len],
                    [oz, oz + R[2, 0] * axis_len],
                )
                self._camera_y.set_data_3d(
                    [ox, ox + R[0, 1] * axis_len],
                    [oy, oy + R[1, 1] * axis_len],
                    [oz, oz + R[2, 1] * axis_len],
                )
                self._camera_z.set_data_3d(
                    [ox, ox + R[0, 2] * axis_len],
                    [oy, oy + R[1, 2] * axis_len],
                    [oz, oz + R[2, 2] * axis_len],
                )

                frustum = self._build_frustum(R, t)
                fx, fy, fz = self._build_edge_line(frustum, [
                    (0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 2), (2, 3), (3, 4), (4, 1),
                ])
                self._camera_frustum.set_data_3d(fx, fy, fz)
            else:
                # Position-only (fallback / traditional mode): hide orientation.
                self._camera_x.set_data_3d([0, 0], [0, 0], [0, 0])
                self._camera_y.set_data_3d([0, 0], [0, 0], [0, 0])
                self._camera_z.set_data_3d([0, 0], [0, 0], [0, 0])
                self._camera_frustum.set_data_3d([0, 0], [0, 0], [0, 0])

            info = (
                f"X:{x:+.2f} Y:{y:+.2f} Z:{z:+.2f}\n"
                f"R:{roll_hist[-1]:+.1f} "
                f"P:{pitch_hist[-1]:+.1f} "
                f"Y:{yaw_hist[-1]:+.1f}"
            )
            self._info_text.set_text(info)

            if self._mode == 2:
                mode_label = "fallback"
            elif self._mode == 1:
                mode_label = "5-point"
            else:
                mode_label = "4-point"
            self._mode_text.set_text(f"Mode: {mode_label}")
            self._mode_text.set_color("#00cc00")
        else:
            self._camera_marker.set_data_3d([0, 0], [0, 0], [0, 0])
            self._camera_x.set_data_3d([0, 0], [0, 0], [0, 0])
            self._camera_y.set_data_3d([0, 0], [0, 0], [0, 0])
            self._camera_z.set_data_3d([0, 0], [0, 0], [0, 0])
            self._camera_frustum.set_data_3d([0, 0], [0, 0], [0, 0])
            self._info_text.set_text("")
            self._mode_text.set_text("Mode: --")
            self._mode_text.set_color("black")

    def _draw_history(self) -> None:
        # Combine both mode histories into unified frame index series for the
        # 2D plots; inactive-mode samples appear as gaps (NaN) in each line.
        n4 = len(self.x_hist_4)
        n5 = len(self.x_hist_5)
        n = max(n4, n5)
        if n == 0:
            return
        idx = np.arange(n)
        # Cap displayed history to reduce 2D redraw cost.
        display_n = min(n, 100)
        start = n - display_n
        idx = idx[start:]

        def _slice(hist, start, n, display_n):
            arr = np.full(n, float("nan"))
            arr[:len(hist)] = list(hist)
            return arr[start:]

        self._pos_lines[0].set_data(idx, _slice(self.x_hist_4, start, n, display_n))
        self._pos_lines[1].set_data(idx, _slice(self.y_hist_4, start, n, display_n))
        self._pos_lines[2].set_data(idx, _slice(self.z_hist_4, start, n, display_n))
        self._rot_lines[0].set_data(idx, _slice(self.roll_hist_4, start, n, display_n))
        self._rot_lines[1].set_data(idx, _slice(self.pitch_hist_4, start, n, display_n))
        self._rot_lines[2].set_data(idx, _slice(self.yaw_hist_4, start, n, display_n))

        # Second set of lines for 5-point mode, drawn with dashed style.
        if not hasattr(self, "_pos_lines_5"):
            self._pos_lines_5 = [
                self.ax_pos.plot([], [], color=COLOR_TRAJECTORY_5POINT, linestyle="--", linewidth=1.5)[0],
                self.ax_pos.plot([], [], color=COLOR_TRAJECTORY_5POINT, linestyle="--", linewidth=1.5)[0],
                self.ax_pos.plot([], [], color=COLOR_TRAJECTORY_5POINT, linestyle="--", linewidth=1.5)[0],
            ]
            self._rot_lines_5 = [
                self.ax_rot.plot([], [], color=COLOR_TRAJECTORY_5POINT, linestyle="--", linewidth=1.5)[0],
                self.ax_rot.plot([], [], color=COLOR_TRAJECTORY_5POINT, linestyle="--", linewidth=1.5)[0],
                self.ax_rot.plot([], [], color=COLOR_TRAJECTORY_5POINT, linestyle="--", linewidth=1.5)[0],
            ]

        self._pos_lines_5[0].set_data(idx, _slice(self.x_hist_5, start, n, display_n))
        self._pos_lines_5[1].set_data(idx, _slice(self.y_hist_5, start, n, display_n))
        self._pos_lines_5[2].set_data(idx, _slice(self.z_hist_5, start, n, display_n))
        self._rot_lines_5[0].set_data(idx, _slice(self.roll_hist_5, start, n, display_n))
        self._rot_lines_5[1].set_data(idx, _slice(self.pitch_hist_5, start, n, display_n))
        self._rot_lines_5[2].set_data(idx, _slice(self.yaw_hist_5, start, n, display_n))

        for ax, datasets_4, datasets_5 in (
            (self.ax_pos,
             [self.x_hist_4, self.y_hist_4, self.z_hist_4],
             [self.x_hist_5, self.y_hist_5, self.z_hist_5]),
            (self.ax_rot,
             [self.roll_hist_4, self.pitch_hist_4, self.yaw_hist_4],
             [self.roll_hist_5, self.pitch_hist_5, self.yaw_hist_5]),
        ):
            recent_4 = [np.array(list(d)[max(0, len(d) - display_n):], dtype=float) for d in datasets_4]
            recent_5 = [np.array(list(d)[max(0, len(d) - display_n):], dtype=float) for d in datasets_5]
            finite = np.concatenate([
                d[np.isfinite(d)] for d in recent_4
            ] + [
                d[np.isfinite(d)] for d in recent_5
            ])
            if finite.size > 0:
                lo, hi = finite.min(), finite.max()
                margin = (hi - lo) * 0.1 + 1e-6
                ax.set_ylim(lo - margin, hi + margin)
            ax.set_xlim(max(0, n - display_n), max(1, n - 1))

    def _build_frustum(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        scale = FRUSTUM_SCALE
        focal = 0.10 * scale / 0.08
        img_w = 0.08 * scale / 0.08
        img_h = 0.06 * scale / 0.08
        frustum_cam = np.array(
            [
                [0, 0, 0],
                [-img_w / 2, -img_h / 2, focal],
                [img_w / 2, -img_h / 2, focal],
                [img_w / 2, img_h / 2, focal],
                [-img_w / 2, img_h / 2, focal],
            ]
        )
        return (R @ frustum_cam.T).T + np.asarray(t).flatten()

    @staticmethod
    def _build_edge_line(pts: np.ndarray, edges: list) -> tuple[list[float], list[float], list[float]]:
        """Build a single line with NaN separators from a list of edges."""
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for edge in edges:
            a, b = pts[edge[0]], pts[edge[1]]
            xs.extend([float(a[0]), float(b[0]), float("nan")])
            ys.extend([float(a[1]), float(b[1]), float("nan")])
            zs.extend([float(a[2]), float(b[2]), float("nan")])
        return xs, ys, zs


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(line_buffering=True)
    print("Standalone test: synthetic orbital motion")
    print("Close the window or press Ctrl+C to quit")

    viz = LivePoseVisualizer(history_size=200, fps=15)
    t0 = time.time()
    frame = 0
    test_duration = 5.0  # seconds; set to None to run until window closed

    try:
        while not viz.should_quit():
            t = time.time() - t0
            if test_duration is not None and t >= test_duration:
                break
            radius = 1.5
            tvec = np.array(
                [
                    radius * math.cos(t * 0.5),
                    radius * math.sin(t * 0.3) * 0.3,
                    2.0 + 0.5 * math.sin(t * 0.8),
                ]
            )
            rvec = np.array(
                [
                    math.radians(5 * math.sin(t * 1.2)),
                    math.radians(10 * math.sin(t * 0.7)),
                    math.radians(30 * math.sin(t * 0.5)),
                ]
            )
            viz.update(
                tvec, rvec, is_valid=True, roll=rvec[0], pitch=rvec[1], yaw=rvec[2],
                mode=1 if math.sin(t) > 0 else 0,
            )
            frame += 1
            if frame % 30 == 0:
                avg_render = (
                    sum(viz._impl._render_times) / len(viz._impl._render_times)
                    if viz._impl._render_times
                    else 0.0
                )
                print(
                    f"Push FPS: {frame / (time.time() - t0):.1f}, "
                    f"Render FPS: {1.0 / avg_render:.1f} "
                    f"(avg {avg_render * 1000:.1f} ms/frame)",
                    flush=True,
                )
            time.sleep(0.033)
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()
        print("Test finished.", flush=True)
