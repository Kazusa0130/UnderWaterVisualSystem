"""3D pose visualization module using matplotlib.

This module provides real-time 3D visualization of camera and target poses
for underwater robotics applications. It supports two visualization modes:
- Target as origin: Shows camera pose relative to the target
- Camera as origin: Shows target pose relative to the camera

Coordinate System:
    Uses OpenCV camera coordinate convention throughout:
    - X axis: points to the right
    - Y axis: points downward
    - Z axis: points forward (into the scene, along the optical axis)

Typical usage example:
    visualizer = PoseVisualizer3D(
        axis_length=0.3,
        view_angle=(20, -45),
        origin_mode="target"
    )
    visualizer.update(rvec, tvec, obj_width=0.46, obj_length=0.60)
    visualizer.close()

Author: BJTU Underwater Robotics Team
"""

from typing import Optional, Sequence, Tuple, Union
import time

import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


__all__ = ['PoseVisualizer3D']


# Constants
DEFAULT_AXIS_LENGTH = 0.5
DEFAULT_VIEW_ANGLE = (30, -60)  # (elev, azim)
DEFAULT_FOV_H = 60  # Horizontal field of view in degrees
DEFAULT_FOV_V = 45  # Vertical field of view in degrees
DEFAULT_FRUSTUM_DEPTH = 0.3
MAX_HISTORY_LENGTH = 100

# Color schemes
COLOR_X_AXIS = 'red'
COLOR_Y_AXIS = 'green'
COLOR_Z_AXIS = 'blue'
COLOR_TARGET_X = 'orange'
COLOR_TARGET_Y = 'lime'
COLOR_TARGET_Z = 'purple'
COLOR_CAMERA_VALID = 'blue'
COLOR_CAMERA_INVALID = 'red'
COLOR_TRAJECTORY = 'yellow'
COLOR_START_POINT = 'green'
COLOR_END_POINT = 'red'
COLOR_TARGET_PLANE = 'cyan'


class PoseVisualizer3D:
    """3D pose visualizer supporting target-as-origin or camera-as-origin modes.

    This class creates an interactive matplotlib 3D plot showing the relative
    pose between camera and target. It supports real-time updates and can
    display motion trajectories.

    Attributes:
        axis_length: Length of coordinate axes in meters.
        view_angle: Initial viewing angle as (elevation, azimuth) in degrees.
        origin_mode: Visualization mode - "target" or "camera" as origin.
        fig: Matplotlib figure instance.
        ax: Matplotlib 3D axes instance.
    """

    def __init__(
        self,
        axis_length: float = DEFAULT_AXIS_LENGTH,
        view_angle: Tuple[float, float] = DEFAULT_VIEW_ANGLE,
        origin_mode: str = "target"
    ) -> None:
        """Initialize the 3D pose visualizer.

        Args:
            axis_length: Length of coordinate axes in meters.
            view_angle: Initial viewing angle as (elevation, azimuth) tuple.
            origin_mode: "target" for target-as-origin, "camera" for
                camera-as-origin.

        Raises:
            ValueError: If origin_mode is not "target" or "camera".
        """
        if origin_mode not in ("target", "camera"):
            raise ValueError(
                f"Invalid origin_mode: {origin_mode}. "
                "Must be 'target' or 'camera'."
            )

        self.axis_length = axis_length
        self.view_angle = view_angle
        self.origin_mode = origin_mode
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[Axes3D] = None
        self._init_figure()

    def _init_figure(self) -> None:
        """Initialize the matplotlib figure with interactive mode."""
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._setup_axes()
        plt.show(block=False)

    def _setup_axes(self) -> None:
        """Configure axis labels, limits, and title."""
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_zlabel('Z (m)', fontsize=12)

        title = (
            '3D Pose Visualization (Target as Origin)'
            if self.origin_mode == "target"
            else '3D Pose Visualization (Camera as Origin)'
        )
        self.ax.set_title(title, fontsize=14)

        # Set axis limits
        limit = self.axis_length * 4
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([-limit, limit])

        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])

    def _rotation_matrix_from_rvec(
        self,
        rvec: Optional[Union[np.ndarray, Sequence[float]]]
    ) -> np.ndarray:
        """Convert rotation vector to rotation matrix.

        Args:
            rvec: Rotation vector with shape (3,) or (3, 1).

        Returns:
            3x3 rotation matrix. Returns identity if rvec is None or invalid.
        """
        if rvec is None:
            return np.eye(3)

        rvec_arr = np.asarray(rvec).flatten()
        if rvec_arr.shape != (3,):
            return np.eye(3)

        rotation_matrix, _ = cv2.Rodrigues(rvec_arr)
        return rotation_matrix

    def _draw_coordinate_frame(
        self,
        origin: Union[np.ndarray, Sequence[float]],
        rotation_matrix: np.ndarray,
        label: str,
        colors: Optional[Tuple[str, str, str]] = None,
        alpha: float = 1.0,
        linewidth: float = 2.0
    ) -> None:
        """Draw a 3D coordinate frame at specified origin.

        Args:
            origin: Origin position with shape (3,).
            rotation_matrix: 3x3 rotation matrix.
            label: Label text for the coordinate frame.
            colors: Tuple of (x_color, y_color, z_color).
            alpha: Transparency value [0, 1].
            linewidth: Width of axis lines.
        """
        if colors is None:
            colors = (COLOR_X_AXIS, COLOR_Y_AXIS, COLOR_Z_AXIS)

        origin_arr = np.asarray(origin).flatten()

        # Draw axes
        axes = np.eye(3) * self.axis_length
        for i, (axis, color) in enumerate(zip(axes, colors)):
            rotated_axis = rotation_matrix @ axis
            self.ax.quiver(
                origin_arr[0], origin_arr[1], origin_arr[2],
                rotated_axis[0], rotated_axis[1], rotated_axis[2],
                color=color,
                arrow_length_ratio=0.2,
                linewidth=linewidth,
                alpha=alpha
            )

        # Mark origin
        self.ax.scatter(
            *origin_arr,
            color='black',
            s=50,
            marker='o',
            alpha=alpha
        )
        self.ax.text(
            origin_arr[0],
            origin_arr[1],
            origin_arr[2],
            label,
            fontsize=10
        )

    def _draw_camera_frustum(
        self,
        position: Union[np.ndarray, Sequence[float]],
        rotation_matrix: np.ndarray,
        color: str = COLOR_CAMERA_VALID,
        alpha: float = 0.3
    ) -> None:
        """Draw camera frustum (viewing pyramid).

        Uses OpenCV camera coordinate convention: X right, Y down, Z forward.
        The frustum extends along the positive Z axis (camera viewing direction).

        Args:
            position: Camera position in world coordinates (3,).
            rotation_matrix: Camera rotation matrix (3, 3) from camera to world.
            color: Color for the frustum.
            alpha: Transparency value [0, 1].
        """
        position_arr = np.asarray(position).flatten()

        # Camera frustum parameters
        fov_h = np.radians(DEFAULT_FOV_H)
        fov_v = np.radians(DEFAULT_FOV_V)
        depth = DEFAULT_FRUSTUM_DEPTH

        # Frustum corners in camera coordinates (OpenCV convention)
        # OpenCV camera: X right, Y down, Z forward (into the scene)
        # Camera looks along positive Z axis
        corners_cam = np.array([
            [0, 0, 0],  # Camera center
            [-depth * np.tan(fov_h / 2),
             -depth * np.tan(fov_v / 2),
             depth],
            [depth * np.tan(fov_h / 2),
             -depth * np.tan(fov_v / 2),
             depth],
            [depth * np.tan(fov_h / 2),
             depth * np.tan(fov_v / 2),
             depth],
            [-depth * np.tan(fov_h / 2),
             depth * np.tan(fov_v / 2),
             depth],
        ])

        # Transform from camera coordinates to world coordinates
        # rotation_matrix is camera-to-world rotation
        corners_world = np.zeros_like(corners_cam)
        for i, corner in enumerate(corners_cam):
            corners_world[i] = rotation_matrix @ corner + position_arr

        # Draw frustum edges
        edges = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # Center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # Corner connections
        ]
        for edge in edges:
            points = corners_world[edge]
            self.ax.plot3D(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                alpha=alpha,
                linewidth=1
            )

        # Fill frustum faces
        faces = [
            [corners_world[0], corners_world[1], corners_world[2]],
            [corners_world[0], corners_world[2], corners_world[3]],
            [corners_world[0], corners_world[3], corners_world[4]],
            [corners_world[0], corners_world[4], corners_world[1]],
        ]
        poly3d = Poly3DCollection(
            faces,
            alpha=alpha * 0.5,
            facecolor=color,
            edgecolor='none'
        )
        self.ax.add_collection3d(poly3d)

    def _draw_target_plane(
        self,
        rotation_world: np.ndarray,
        translation: Union[np.ndarray, Sequence[float]],
        width: float,
        length: float,
        color: str = COLOR_TARGET_PLANE,
        alpha: float = 0.3
    ) -> None:
        """Draw the target object plane.

        Args:
            rotation_world: Target-to-world rotation matrix (3, 3).
            translation: Translation vector (3,) in world coordinates.
            width: Object width in meters.
            length: Object length in meters.
            color: Color for the target plane.
            alpha: Transparency value [0, 1].
        """
        # Target corners in target coordinates
        corners_target = np.array([
            [-width / 2, -length / 2, 0],
            [-width / 2, length / 2, 0],
            [width / 2, length / 2, 0],
            [width / 2, -length / 2, 0]
        ])

        # Transform to world coordinates
        if self.origin_mode == "target":
            corners_world = corners_target
        else:
            tvec = np.asarray(translation).flatten()
            corners_world = (rotation_world @ corners_target.T).T + tvec

        # Draw rectangle edges
        edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        for edge in edges:
            points = corners_world[edge]
            self.ax.plot3D(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                linewidth=2,
                alpha=0.8
            )

        # Fill plane
        verts = [corners_world]
        poly3d = Poly3DCollection(
            verts,
            alpha=alpha,
            facecolor=color,
            edgecolor='none'
        )
        self.ax.add_collection3d(poly3d)

    def _calculate_camera_pose_in_target_frame(
        self,
        tvec: np.ndarray,
        rvec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate camera pose in target coordinate frame.

        OpenCV PnP returns rvec, tvec representing target pose in camera frame.
        This method converts to camera pose in target frame.

        Args:
            tvec: Target translation in camera frame (3,).
            rvec: Target rotation in camera frame (3,).

        Returns:
            Tuple of (camera_position, camera_rotation) in target frame.
            Uses OpenCV convention: X right, Y down, Z forward.
        """
        # rvec describes rotation from target to camera coordinates
        r_target_to_cam = self._rotation_matrix_from_rvec(rvec)
        # Inverse rotation: from camera to target coordinates
        r_cam_to_target = r_target_to_cam.T
        # Camera position in target frame: -R^T * tvec
        camera_pos = -r_cam_to_target @ tvec
        return camera_pos, r_cam_to_target

    def _draw_target_as_origin_mode(
        self,
        rvec: Optional[np.ndarray],
        tvec: Optional[np.ndarray],
        obj_width: Optional[float],
        obj_length: Optional[float],
        history: Optional[Sequence[np.ndarray]],
        is_valid: bool
    ) -> None:
        """Render scene with target as coordinate origin.

        Args:
            rvec: Target rotation in camera frame (3,).
            tvec: Target translation in camera frame (3,).
            obj_width: Object width in meters.
            obj_length: Object length in meters.
            history: Sequence of historical camera positions.
            is_valid: Whether current pose is valid.
        """
        # Calculate camera pose in target frame
        if tvec is not None and rvec is not None:
            camera_pos, camera_rot = self._calculate_camera_pose_in_target_frame(
                tvec, rvec
            )
        else:
            camera_pos, camera_rot = None, None

        # Draw target coordinate system at origin
        self._draw_coordinate_frame(
            [0, 0, 0],
            np.eye(3),
            "Target",
            colors=(COLOR_TARGET_X, COLOR_TARGET_Y, COLOR_TARGET_Z)
        )

        # Draw target plane
        if obj_width is not None and obj_length is not None:
            self._draw_target_plane(
                np.eye(3),
                [0, 0, 0],
                obj_width,
                obj_length,
                color=COLOR_TARGET_PLANE,
                alpha=0.3
            )

        # Draw camera
        if camera_pos is not None and camera_rot is not None:
            cam_color = COLOR_CAMERA_VALID if is_valid else COLOR_CAMERA_INVALID
            self._draw_coordinate_frame(
                camera_pos,
                camera_rot,
                "Camera",
                colors=(COLOR_X_AXIS, COLOR_Y_AXIS, COLOR_Z_AXIS)
            )
            self._draw_camera_frustum(
                camera_pos,
                camera_rot,
                color=cam_color,
                alpha=0.2
            )

            # Draw line to origin
            self.ax.plot3D(
                [camera_pos[0], 0],
                [camera_pos[1], 0],
                [camera_pos[2], 0],
                color='gray',
                linestyle='--',
                alpha=0.5
            )

        # Draw trajectory
        if history is not None and len(history) > 0:
            history_arr = np.array(history)
            self.ax.plot3D(
                history_arr[:, 0],
                history_arr[:, 1],
                history_arr[:, 2],
                color=COLOR_TRAJECTORY,
                linestyle='-',
                alpha=0.6,
                linewidth=2,
                label='Camera Trajectory'
            )
            if len(history_arr) > 1:
                self.ax.scatter(
                    *history_arr[0],
                    c=COLOR_START_POINT,
                    marker='o',
                    s=100,
                    label='Start'
                )
                self.ax.scatter(
                    *history_arr[-1],
                    c=COLOR_END_POINT,
                    marker='s',
                    s=100,
                    label='End'
                )

        # Display pose info
        if camera_pos is not None:
            distance = np.linalg.norm(camera_pos)
            status_text = 'Valid' if is_valid else 'Rejected (Outlier)'
            facecolor = 'lightblue' if is_valid else 'lightcoral'
            info_text = (
                f"Camera Position (in Target Frame):\n"
                f"  X: {camera_pos[0]:.3f} m\n"
                f"  Y: {camera_pos[1]:.3f} m\n"
                f"  Z: {camera_pos[2]:.3f} m\n"
                f"  Distance: {distance:.3f} m\n"
                f"Status: {status_text}"
            )
            self.ax.text2D(
                0.02,
                0.98,
                info_text,
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(
                    boxstyle='round',
                    facecolor=facecolor,
                    alpha=0.8
                )
            )

    def _draw_camera_as_origin_mode(
        self,
        rvec: Optional[np.ndarray],
        tvec: Optional[np.ndarray],
        obj_width: Optional[float],
        obj_length: Optional[float],
        history: Optional[Sequence[np.ndarray]],
        is_valid: bool
    ) -> None:
        """Render scene with camera as coordinate origin.

        Args:
            rvec: Target rotation in camera frame (3,).
            tvec: Target translation in camera frame (3,).
            obj_width: Object width in meters.
            obj_length: Object length in meters.
            history: Sequence of historical target positions.
            is_valid: Whether current pose is valid.
        """
        # Draw camera coordinate system at origin
        self._draw_coordinate_frame(
            [0, 0, 0],
            np.eye(3),
            "Camera",
            colors=(COLOR_X_AXIS, COLOR_Y_AXIS, COLOR_Z_AXIS)
        )

        # Draw target
        if rvec is not None and tvec is not None:
            r_target_to_cam = self._rotation_matrix_from_rvec(rvec)
            target_color = COLOR_TARGET_PLANE if is_valid else 'orange'

            self._draw_coordinate_frame(
                tvec,
                r_target_to_cam,
                "Target",
                colors=(COLOR_TARGET_X, COLOR_TARGET_Y, COLOR_TARGET_Z)
            )

            # Draw target plane
            if obj_width is not None and obj_length is not None:
                self._draw_target_plane(
                    r_target_to_cam,
                    tvec,
                    obj_width,
                    obj_length,
                    color=target_color,
                    alpha=0.3
                )

            # Draw line from camera to target
            self.ax.plot3D(
                [0, tvec[0]],
                [0, tvec[1]],
                [0, tvec[2]],
                color='gray',
                linestyle='--',
                alpha=0.5
            )

        # Draw trajectory
        if history is not None and len(history) > 0:
            history_arr = np.array(history)
            self.ax.plot3D(
                history_arr[:, 0],
                history_arr[:, 1],
                history_arr[:, 2],
                color=COLOR_TRAJECTORY,
                linestyle='-',
                alpha=0.6,
                linewidth=2,
                label='Target Trajectory'
            )
            if len(history_arr) > 1:
                self.ax.scatter(
                    *history_arr[0],
                    c=COLOR_START_POINT,
                    marker='o',
                    s=100
                )
                self.ax.scatter(
                    *history_arr[-1],
                    c=COLOR_END_POINT,
                    marker='s',
                    s=100
                )

        # Display pose info
        if tvec is not None:
            distance = np.linalg.norm(tvec)
            status_text = 'Valid' if is_valid else 'Rejected (Outlier)'
            facecolor = 'lightblue' if is_valid else 'lightcoral'
            info_text = (
                f"Target Position (in Camera Frame):\n"
                f"  X: {tvec[0]:.3f} m\n"
                f"  Y: {tvec[1]:.3f} m\n"
                f"  Z: {tvec[2]:.3f} m\n"
                f"  Distance: {distance:.3f} m\n"
                f"Status: {status_text}"
            )
            self.ax.text2D(
                0.02,
                0.98,
                info_text,
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(
                    boxstyle='round',
                    facecolor=facecolor,
                    alpha=0.8
                )
            )

    def update(
        self,
        rvec: Optional[Union[np.ndarray, Sequence[float]]] = None,
        tvec: Optional[Union[np.ndarray, Sequence[float]]] = None,
        obj_width: Optional[float] = None,
        obj_length: Optional[float] = None,
        history_camera_positions: Optional[Sequence[np.ndarray]] = None,
        is_valid: bool = True
    ) -> None:
        """Update the 3D visualization.

        Args:
            rvec: Target rotation vector in camera frame (3,).
            tvec: Target translation vector in camera frame (3,).
            obj_width: Object width in meters.
            obj_length: Object length in meters.
            history_camera_positions: Historical positions for trajectory display.
            is_valid: Whether current pose is valid (affects color coding).
        """
        self.ax.clear()
        self._setup_axes()

        # Convert inputs to arrays
        rvec_arr = np.asarray(rvec).flatten() if rvec is not None else None
        tvec_arr = np.asarray(tvec).flatten() if tvec is not None else None

        # Render based on origin mode
        if self.origin_mode == "target":
            self._draw_target_as_origin_mode(
                rvec_arr,
                tvec_arr,
                obj_width,
                obj_length,
                history_camera_positions,
                is_valid
            )
        else:
            self._draw_camera_as_origin_mode(
                rvec_arr,
                tvec_arr,
                obj_width,
                obj_length,
                history_camera_positions,
                is_valid
            )

        # Set view angle and legend
        self.ax.view_init(
            elev=self.view_angle[0],
            azim=self.view_angle[1]
        )
        self.ax.legend(loc='upper right')

        # Update canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, filename: str) -> None:
        """Save current view to image file.

        Args:
            filename: Output file path.
        """
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"3D visualization saved to {filename}")

    def close(self) -> None:
        """Close the visualization window."""
        plt.close(self.fig)
        plt.ioff()


def _test_visualizer() -> None:
    """Test the visualizer with simulated circular motion."""
    print("Testing 3D Pose Visualizer...")
    print("Mode: Target as Origin")

    visualizer = PoseVisualizer3D(
        axis_length=0.3,
        view_angle=(20, -45),
        origin_mode="target"
    )

    history = []
    radius = 1.5
    height = 0.5

    for i in range(100):
        # Simulate circular motion around target
        angle = i * 0.1
        camera_pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            height + 0.3 * np.sin(angle * 2)
        ])

        # Calculate camera orientation looking at origin
        z_axis = -camera_pos / np.linalg.norm(camera_pos)
        up = np.array([0, 0, 1])
        if np.abs(np.dot(z_axis, up)) > 0.99:
            up = np.array([0, 1, 0])
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        r_camera = np.column_stack([x_axis, y_axis, z_axis])
        r_target_to_cam = r_camera.T
        rvec, _ = cv2.Rodrigues(r_target_to_cam)
        tvec = -r_target_to_cam @ camera_pos

        history.append(camera_pos)
        if len(history) > MAX_HISTORY_LENGTH:
            history.pop(0)

        visualizer.update(
            rvec=rvec,
            tvec=tvec,
            obj_width=0.46,
            obj_length=0.60,
            history_camera_positions=history
        )
        time.sleep(0.05)

    visualizer.close()
    print("Test completed!")


if __name__ == "__main__":
    _test_visualizer()
