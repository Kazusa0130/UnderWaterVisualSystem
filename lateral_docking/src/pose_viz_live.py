"""Real-time matplotlib pose visualization (camera-in-target-frame).

A standalone matplotlib window showing live camera pose relative to the
target plane. Visual style matches ``visualize_traj.py``.

Usage in main.py::

    from pose_viz_live import LivePoseVisualizer

    viz = LivePoseVisualizer(history_size=300)
    while True:
        # ... get pose ...
        viz.update(tvec_cam, rvec_cam, is_valid)
        if viz.should_quit():
            break
    viz.close()

Author: BJTU Underwater Robotics Team
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Visual style constants (matching visualize_traj.py)
# ---------------------------------------------------------------------------
COLOR_X_AXIS = "red"
COLOR_Y_AXIS = "green"
COLOR_Z_AXIS = "blue"
COLOR_TRAJECTORY = "blue"
COLOR_TRAJECTORY_INVALID = "gray"
COLOR_START_POINT = "green"
COLOR_END_POINT = "red"
COLOR_TARGET_PLANE = "cyan"
COLOR_CAMERA = "orange"

# Target corners in physical target frame (Y-up, Z-toward-camera)
DEFAULT_OBJ_CORNERS = np.array(
    [
        [-0.100, 0.300, 0.000],  # 左上
        [0.100, 0.300, 0.000],  # 右上
        [0.245, -0.300, 0.000],  # 右下
        [-0.245, -0.300, 0.000],  # 左下
    ]
)


class LivePoseVisualizer:
    """Real-time matplotlib pose visualization window.

    Opens a standalone figure with three subplots:

    * **Left (large)**: 3D view with target plane, camera position,
      orientation axes, frustum, and trajectory trail.
    * **Top-right**: Position history (X/Y/Z vs. frame).
    * **Bottom-right**: Rotation history (Roll/Pitch/Yaw vs. frame).

    Call :meth:`update` every frame and :meth:`should_quit` to check whether
    the user closed the window.
    """

    def __init__(
        self,
        history_size: int = 300,
        show_trail: bool = True,
        frustum_scale: float = 0.08,
    ) -> None:
        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title(
            "Live Camera Pose in Target Frame"
        )
        self._closed = False
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        # Layout: left 2/3 for 3D, right 1/3 split into two history plots
        self.ax3d = self.fig.add_subplot(1, 2, 1, projection="3d")
        self.ax_pos = self.fig.add_subplot(2, 2, 2)
        self.ax_rot = self.fig.add_subplot(2, 2, 4)

        self.history_size = history_size
        self.show_trail = show_trail
        self.frustum_scale = frustum_scale

        # Data buffers
        self.x_hist: deque[float] = deque(maxlen=history_size)
        self.y_hist: deque[float] = deque(maxlen=history_size)
        self.z_hist: deque[float] = deque(maxlen=history_size)
        self.roll_hist: deque[float] = deque(maxlen=history_size)
        self.pitch_hist: deque[float] = deque(maxlen=history_size)
        self.yaw_hist: deque[float] = deque(maxlen=history_size)
        self.valid_hist: deque[bool] = deque(maxlen=history_size)

        self._start_time = time.time()
        self.current_rvec: np.ndarray | None = None

        # Static 3D elements drawn once via fixed view limits
        self._setup_3d_axes()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _setup_3d_axes(self) -> None:
        """Configure 3D axis limits and aspect (static)."""
        self.ax3d.set_xlim([-3, 3])
        self.ax3d.set_ylim([-3, 3])
        self.ax3d.set_zlim([-1, 5])
        self.ax3d.set_box_aspect([1, 1, 1])
        self.ax3d.set_xlabel("X (m)")
        self.ax3d.set_ylabel("Y (m)")
        self.ax3d.set_zlabel("Z (m)")
        self.ax3d.set_title("3D View (Target as Origin, Physical: Y-up, Z-toward-cam)")
        self.ax3d.view_init(elev=20, azim=-60)

    def _on_close(self, event) -> None:
        self._closed = True

    def should_quit(self) -> bool:
        """Return ``True`` if the user closed the matplotlib window."""
        return self._closed

    def close(self) -> None:
        """Destroy the figure window."""
        if not self._closed:
            plt.close(self.fig)
            self._closed = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        tvec_cam: np.ndarray | None,
        rvec_cam: np.ndarray | None,
        is_valid: bool = True,
        roll: float | None = None,
        pitch: float | None = None,
        yaw: float | None = None,
    ) -> None:
        """Push a new pose sample and refresh the window.

        Args:
            tvec_cam: Camera position in target frame (3,).
            rvec_cam: Camera rotation vector in target frame (3,).
            is_valid: Whether the pose is valid.
            roll: Camera roll in target frame (radians) for history/text.
            pitch: Camera pitch in target frame (radians) for history/text.
            yaw: Camera yaw in target frame (radians) for history/text.
        """
        if self._closed:
            return

        self.valid_hist.append(is_valid)
        if is_valid and tvec_cam is not None and rvec_cam is not None and roll is not None:
            x, y, z = tvec_cam.flatten()
            self.x_hist.append(float(x))
            self.y_hist.append(float(y))
            self.z_hist.append(float(z))
            self.roll_hist.append(float(np.degrees(roll)))
            self.pitch_hist.append(float(np.degrees(pitch)))
            self.yaw_hist.append(float(np.degrees(yaw)))
            self.current_rvec = np.array(rvec_cam).flatten()
        else:
            nan = float("nan")
            self.x_hist.append(nan)
            self.y_hist.append(nan)
            self.z_hist.append(nan)
            self.roll_hist.append(nan)
            self.pitch_hist.append(nan)
            self.yaw_hist.append(nan)
            self.current_rvec = None

        self._redraw()

    # ------------------------------------------------------------------
    # Internal render
    # ------------------------------------------------------------------
    def _redraw(self) -> None:
        """Redraw all dynamic artists."""
        self._draw_3d()
        self._draw_history()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _draw_3d(self) -> None:
        """Clear and redraw the 3D scene."""
        ax = self.ax3d
        ax.clear()
        self._setup_3d_axes()

        # --- Static: target coordinate system ---
        axis_len = 0.3
        ax.quiver(
            0,
            0,
            0,
            axis_len,
            0,
            0,
            color=COLOR_X_AXIS,
            arrow_length_ratio=0.2,
            linewidth=2,
        )
        ax.quiver(
            0,
            0,
            0,
            0,
            axis_len,
            0,
            color=COLOR_Y_AXIS,
            arrow_length_ratio=0.2,
            linewidth=2,
        )
        ax.quiver(
            0,
            0,
            0,
            0,
            0,
            axis_len,
            color=COLOR_Z_AXIS,
            arrow_length_ratio=0.2,
            linewidth=2,
        )

        # --- Static: target plane (non-symmetric corners) ---
        corners = DEFAULT_OBJ_CORNERS
        edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        for edge in edges:
            pts = corners[edge]
            ax.plot3D(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color=COLOR_TARGET_PLANE,
                linewidth=2,
                alpha=0.8,
            )

        # --- Dynamic: trajectory trail ---
        if self.show_trail and len(self.x_hist) > 1:
            x_arr = np.array(self.x_hist, dtype=float)
            y_arr = np.array(self.y_hist, dtype=float)
            z_arr = np.array(self.z_hist, dtype=float)
            valid = ~(np.isnan(x_arr) | np.isnan(y_arr) | np.isnan(z_arr))
            if np.any(valid):
                ax.plot3D(
                    x_arr[valid],
                    y_arr[valid],
                    z_arr[valid],
                    color=COLOR_TRAJECTORY,
                    linewidth=1.5,
                    alpha=0.7,
                    label="Trajectory",
                )

        # --- Dynamic: current camera pose ---
        has_valid_current = (
            len(self.x_hist) > 0
            and not np.isnan(self.x_hist[-1])
            and self.current_rvec is not None
        )
        if has_valid_current:
            x = float(self.x_hist[-1])
            y = float(self.y_hist[-1])
            z = float(self.z_hist[-1])

            # Camera position
            ax.scatter(
                [x],
                [y],
                [z],
                c=COLOR_CAMERA,
                marker="o",
                s=80,
                label="Camera",
                depthshade=False,
            )

            # Camera orientation axes (rvec is axis-angle from PnP)
            R, _ = cv2.Rodrigues(self.current_rvec)
            axis_len = 0.2
            colors = [COLOR_X_AXIS, COLOR_Y_AXIS, COLOR_Z_AXIS]
            for i, color in enumerate(colors):
                axis_dir = R[:, i] * axis_len
                ax.quiver(
                    x,
                    y,
                    z,
                    axis_dir[0],
                    axis_dir[1],
                    axis_dir[2],
                    color=color,
                    arrow_length_ratio=0.3,
                    linewidth=1.5,
                )

            # Camera frustum (ORB-SLAM style)
            frustum = self._build_frustum(R, np.array([x, y, z]))
            for edge in [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 1),
            ]:
                pts = frustum[list(edge)]
                ax.plot3D(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color="black",
                    linewidth=0.8,
                    alpha=0.5,
                )

            # Numeric readout inside 3D axes
            info = (
                f"X:{x:+.2f} Y:{y:+.2f} Z:{z:+.2f}\n"
                f"R:{self.roll_hist[-1]:+.1f} "
                f"P:{self.pitch_hist[-1]:+.1f} "
                f"Y:{self.yaw_hist[-1]:+.1f}"
            )
            ax.text2D(
                0.02,
                0.98,
                info,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            ax.legend(loc="upper left", fontsize=8)

    def _draw_history(self) -> None:
        """Redraw position and rotation history subplots."""
        # Position history
        self.ax_pos.clear()
        if len(self.x_hist) > 0:
            idx = np.arange(len(self.x_hist))
            self.ax_pos.plot(
                idx, list(self.x_hist), color=COLOR_X_AXIS, label="X", linewidth=1.2
            )
            self.ax_pos.plot(
                idx, list(self.y_hist), color=COLOR_Y_AXIS, label="Y", linewidth=1.2
            )
            self.ax_pos.plot(
                idx, list(self.z_hist), color=COLOR_Z_AXIS, label="Z", linewidth=1.2
            )
        self.ax_pos.set_ylabel("Position (m)")
        self.ax_pos.set_title("Position History")
        self.ax_pos.grid(True)
        self.ax_pos.legend(loc="upper right", fontsize=8)

        # Rotation history
        self.ax_rot.clear()
        if len(self.roll_hist) > 0:
            idx = np.arange(len(self.roll_hist))
            self.ax_rot.plot(
                idx,
                list(self.roll_hist),
                color=COLOR_X_AXIS,
                label="Roll",
                linewidth=1.2,
            )
            self.ax_rot.plot(
                idx,
                list(self.pitch_hist),
                color=COLOR_Y_AXIS,
                label="Pitch",
                linewidth=1.2,
            )
            self.ax_rot.plot(
                idx,
                list(self.yaw_hist),
                color=COLOR_Z_AXIS,
                label="Yaw",
                linewidth=1.2,
            )
        self.ax_rot.set_xlabel("Frame")
        self.ax_rot.set_ylabel("Angle (deg)")
        self.ax_rot.set_title("Rotation History")
        self.ax_rot.grid(True)
        self.ax_rot.legend(loc="upper right", fontsize=8)

    def _build_frustum(
        self,
        R: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """Build camera frustum vertices in target frame.

        Args:
            R: Camera rotation matrix in target frame (3, 3).
            t: Camera position in target frame (3,).

        Returns:
            Frustum vertices (5, 3) in target frame.
        """
        scale = self.frustum_scale
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
        return (R @ frustum_cam.T).T + t


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    print("Standalone test: synthetic orbital motion")
    print("Close the matplotlib window or press Ctrl+C to quit")

    viz = LivePoseVisualizer(history_size=200)
    t0 = time.time()

    try:
        while not viz.should_quit():
            t = time.time() - t0
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
            # Synthetic test: treat rvec as small Euler angles for display
            viz.update(tvec, rvec, is_valid=True, roll=rvec[0], pitch=rvec[1], yaw=rvec[2])
            time.sleep(0.033)
    except KeyboardInterrupt:
        pass
    finally:
        viz.close()
        print("Test finished.")
