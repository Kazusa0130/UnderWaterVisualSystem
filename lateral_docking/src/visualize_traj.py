"""Trajectory visualization script.

This script loads and visualizes 6-DOF pose trajectory data from the latest
trajectory file in the traj_data folder.

Usage:
    python visualize_traj.py [--file FILE] [--mode MODE] [--save SAVE]

Arguments:
    --file:  Path to specific trajectory file (default: use latest)
    --mode:  Visualization mode - "3d" or "2d" (default: 3d)
    --save:  Save visualization to file instead of displaying

Author: BJTU Underwater Robotics Team
"""

import argparse
import os
import sys
from typing import Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import get_latest_traj_file, parse_traj_file
from solver import Solver
from config import SAVE_PATH, OBJ_WIDTH, OBJ_LENGTH


# Color schemes
COLOR_X_AXIS = 'red'
COLOR_Y_AXIS = 'green'
COLOR_Z_AXIS = 'blue'
COLOR_TRAJECTORY_VALID = 'blue'
COLOR_TRAJECTORY_INVALID = 'gray'
COLOR_START_POINT = 'green'
COLOR_END_POINT = 'red'
COLOR_TARGET_PLANE = 'cyan'

# Abnormal-pose handling: frames whose camera-Z in the output frame O is
# negative mean the target solved to behind the camera (a bad PnP solution).
# These are kept but de-emphasized (drawn faded gray) rather than dropped.
COLOR_TRAJECTORY_ABNORMAL = 'gray'
ABNORMAL_ALPHA = 0.25


def compute_abnormal_mask(tvecs: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Flag valid frames whose output-frame camera-Z is negative.

    In the output frame O, Z points toward the camera and a normal docking
    pose has Z > 0. A negative Z means the solver placed the target behind the
    camera, which is physically impossible for a real observation and is
    treated as an abnormal (faded) frame.

    Args:
        tvecs: Camera positions in the output frame (N, 3).
        valid_mask: Boolean array marking frames that carry pose data.

    Returns:
        Boolean array (N,) that is True for valid-but-abnormal frames.
    """
    tvecs = np.asarray(tvecs, dtype=float)
    return valid_mask & (tvecs[:, 2] < 0.0)


def visualize_3d_trajectory(
    timestamps: np.ndarray,
    tvecs: np.ndarray,
    rvecs: np.ndarray,
    valid_mask: np.ndarray,
    origin_mode: str = "target",
    save_path: Optional[str] = None,
    pose_interval: int = 10,
    frustum_scale: float = 0.08,
    pose_distance_threshold: Optional[float] = 0.5,
) -> None:
    """Visualize trajectory in 3D.

    Args:
        timestamps: Array of timestamps.
        tvecs: Array of translation vectors (N, 3).
        rvecs: Array of rotation vectors (N, 3).
        valid_mask: Boolean array indicating valid poses.
        origin_mode: "target" for target-as-origin, "camera" for camera-as-origin.
        save_path: Optional path to save the figure.
        pose_interval: Interval (in frames) for drawing camera pose (used when pose_distance_threshold is None).
        pose_distance_threshold: Distance threshold (in meters) for drawing camera pose.
                                 Set to None to use frame-based interval instead.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    if origin_mode == "target":
        # Trajectory data is already in camera-pose-in-target-frame format
        # (tvecs = camera position in target frame, rvecs = camera rotation in target frame)
        camera_positions = np.array(tvecs, dtype=float)
        # Mark invalid positions as NaN so they don't get plotted
        camera_positions[~valid_mask] = np.nan

        # Split valid frames into normal vs abnormal (faded) sets.
        abnormal_mask = compute_abnormal_mask(tvecs, valid_mask)
        normal_mask = valid_mask & ~abnormal_mask

        # Draw target coordinate system at origin
        axis_length = 0.3
        axes = np.eye(3) * axis_length
        colors = (COLOR_X_AXIS, COLOR_Y_AXIS, COLOR_Z_AXIS)
        for i, (axis, color) in enumerate(zip(axes, colors)):
            ax.quiver(0, 0, 0, axis[0], axis[1], axis[2],
                     color=color, arrow_length_ratio=0.2, linewidth=2)

        # Draw target plane (physical frame: Y-up, Z-toward-camera)
        corners = np.array([
            [-0.100,  0.300, 0.000],   # 左上
            [ 0.100,  0.300, 0.000],   # 右上
            [ 0.245, -0.300, 0.000],   # 右下
            [-0.245, -0.300, 0.000],   # 左下
        ])
        edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        for edge in edges:
            points = corners[edge]
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                     color=COLOR_TARGET_PLANE, linewidth=2, alpha=0.8)

        # Draw camera trajectory (normal frames in color, abnormal faded gray)
        normal_positions = camera_positions[normal_mask]
        if len(normal_positions) > 0:
            ax.plot3D(normal_positions[:, 0], normal_positions[:, 1], normal_positions[:, 2],
                     color=COLOR_TRAJECTORY_VALID, linewidth=2, label='Camera Trajectory')
            ax.scatter(*normal_positions[0], c=COLOR_START_POINT, marker='o', s=100, label='Start')
            ax.scatter(*normal_positions[-1], c=COLOR_END_POINT, marker='s', s=100, label='End')

        # Abnormal frames (output-frame Z < 0): faded gray scatter, no solid line.
        abnormal_positions = np.array(tvecs, dtype=float)[abnormal_mask]
        if len(abnormal_positions) > 0:
            ax.scatter(abnormal_positions[:, 0], abnormal_positions[:, 1],
                       abnormal_positions[:, 2], c=COLOR_TRAJECTORY_ABNORMAL,
                       marker='x', s=30, alpha=ABNORMAL_ALPHA,
                       label='Abnormal (Z<0)')

        # Draw camera poses along trajectory (ORB-SLAM style frustum)
        if pose_interval > 0:
            focal_len = 0.10 * frustum_scale / 0.08
            img_w = 0.08 * frustum_scale / 0.08
            img_h = 0.06 * frustum_scale / 0.08

            # Frustum vertices in camera coordinates
            frustum_cam = np.array([
                [0, 0, 0],
                [-img_w / 2, -img_h / 2, focal_len],
                [img_w / 2, -img_h / 2, focal_len],
                [img_w / 2, img_h / 2, focal_len],
                [-img_w / 2, img_h / 2, focal_len],
            ])

            # Only draw frustums for normal frames; abnormal ones are shown
            # as faded scatter only.
            valid_indices = np.where(normal_mask)[0]

            # Select indices based on distance threshold or frame interval
            if pose_distance_threshold is not None and pose_distance_threshold > 0:
                # Distance-based sampling: draw pose when distance from last saved point exceeds threshold
                selected_indices = []
                last_pos = None
                for idx in valid_indices:
                    pos = camera_positions[idx]
                    if last_pos is None or np.linalg.norm(pos - last_pos) >= pose_distance_threshold:
                        selected_indices.append(idx)
                        last_pos = pos
                selected_indices = np.array(selected_indices)
            else:
                # Frame-based sampling (original behavior)
                selected_indices = valid_indices[::pose_interval] if pose_interval > 0 else []

            for idx in selected_indices:
                # rvecs[idx] stores docking [roll, pitch, yaw] (rad), output frame
                rvec = rvecs[idx]
                R_cam_to_target = Solver.docking_euler_to_rotation_output(
                    rvec[0], rvec[1], rvec[2]
                )
                pos = camera_positions[idx]

                # Transform frustum vertices to target frame
                frustum_world = (R_cam_to_target @ frustum_cam.T).T + pos

                # Draw edges: center to 4 corners + rectangle edges
                edges = [
                    (0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 2), (2, 3), (3, 4), (4, 1)
                ]
                for e in edges:
                    pts = frustum_world[list(e)]
                    ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2],
                             color='black', linewidth=1.0, alpha=0.6)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory Visualization (Target as Origin, Physical Frame)')

        # Set initial view for physical target frame (X right, Y up, Z toward camera)
        ax.view_init(elev=20, azim=-60)

    else:
        # Camera as origin mode
        # Trajectory data is camera-pose-in-target-frame, need to convert to
        # target-pose-in-camera-frame for display
        target_positions = []
        for i in range(len(tvecs)):
            if valid_mask[i]:
                t_tc = tvecs[i]
                # rvecs[i] = docking [roll, pitch, yaw] in output frame
                R_tc = Solver.docking_euler_to_rotation_output(
                    rvecs[i][0], rvecs[i][1], rvecs[i][2]
                )
                # target position in camera frame = -R_tc.T @ t_tc
                t_ct = -R_tc.T @ t_tc
                target_positions.append(t_ct)
            else:
                target_positions.append([np.nan, np.nan, np.nan])
        target_positions = np.array(target_positions)
        abnormal_mask = compute_abnormal_mask(tvecs, valid_mask)
        normal_mask = valid_mask & ~abnormal_mask
        normal_positions = target_positions[normal_mask]
        abnormal_positions = target_positions[abnormal_mask]

        # Draw camera coordinate system at origin
        axis_length = 0.3
        axes = np.eye(3) * axis_length
        colors = (COLOR_X_AXIS, COLOR_Y_AXIS, COLOR_Z_AXIS)
        for i, (axis, color) in enumerate(zip(axes, colors)):
            ax.quiver(0, 0, 0, axis[0], axis[1], axis[2],
                     color=color, arrow_length_ratio=0.2, linewidth=2)

        # Draw target trajectory (normal in color, abnormal faded gray)
        if len(normal_positions) > 0:
            ax.plot3D(normal_positions[:, 0], normal_positions[:, 1], normal_positions[:, 2],
                     color=COLOR_TRAJECTORY_VALID, linewidth=2, label='Target Trajectory')
            ax.scatter(*normal_positions[0], c=COLOR_START_POINT, marker='o', s=100, label='Start')
            ax.scatter(*normal_positions[-1], c=COLOR_END_POINT, marker='s', s=100, label='End')
        if len(abnormal_positions) > 0:
            ax.scatter(abnormal_positions[:, 0], abnormal_positions[:, 1],
                       abnormal_positions[:, 2], c=COLOR_TRAJECTORY_ABNORMAL,
                       marker='x', s=30, alpha=ABNORMAL_ALPHA,
                       label='Abnormal (Z<0)')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory Visualization (Camera as Origin)')

        # Set initial view for physical target frame
        ax.view_init(elev=20, azim=-60)

    # Set equal aspect ratio
    max_range = 2.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_box_aspect([1, 1, 1])

    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D visualization saved to {save_path}")
    else:
        plt.show()


def visualize_2d_trajectory(
    timestamps: np.ndarray,
    tvecs: np.ndarray,
    rvecs: np.ndarray,
    valid_mask: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """Visualize trajectory in 2D plots.

    Args:
        timestamps: Array of timestamps.
        tvecs: Array of translation vectors (N, 3).
        rvecs: Array of rotation vectors (N, 3).
        valid_mask: Boolean array indicating valid poses.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    abnormal_mask = compute_abnormal_mask(tvecs, valid_mask)
    normal_mask = valid_mask & ~abnormal_mask
    rvecs_deg = np.degrees(rvecs)

    def _plot_channel(ax, data, color, title, ylabel):
        """Plot normal frames as a colored line and abnormal ones faded gray."""
        ax.plot(timestamps[normal_mask], data[normal_mask], color=color,
                linestyle='-')
        if np.any(abnormal_mask):
            ax.scatter(timestamps[abnormal_mask], data[abnormal_mask],
                       c=COLOR_TRAJECTORY_ABNORMAL, marker='x', s=20,
                       alpha=ABNORMAL_ALPHA, label='Abnormal (Z<0)')
            ax.legend(fontsize=8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

    # Position plots (camera position in target frame)
    _plot_channel(axes[0, 0], tvecs[:, 0], 'b', 'Camera X in Target Frame', 'Position (m)')
    _plot_channel(axes[0, 1], tvecs[:, 1], 'g', 'Camera Y in Target Frame', 'Position (m)')
    _plot_channel(axes[0, 2], tvecs[:, 2], 'r', 'Camera Z in Target Frame', 'Position (m)')

    # Rotation plots (camera rotation in target frame, in degrees)
    _plot_channel(axes[1, 0], rvecs_deg[:, 0], 'b', 'Cam Roll in Target Frame', 'Angle (deg)')
    _plot_channel(axes[1, 1], rvecs_deg[:, 1], 'g', 'Cam Pitch in Target Frame', 'Angle (deg)')
    _plot_channel(axes[1, 2], rvecs_deg[:, 2], 'r', 'Cam Yaw in Target Frame', 'Angle (deg)')

    fig.suptitle('6-DOF Camera Pose in Target Frame Over Time', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"2D visualization saved to {save_path}")
    else:
        plt.show()


def print_trajectory_summary(
    timestamps: np.ndarray,
    tvecs: np.ndarray,
    rvecs: np.ndarray,
    valid_mask: np.ndarray,
    filepath: str
) -> None:
    """Print summary statistics of the trajectory.

    Args:
        timestamps: Array of timestamps.
        tvecs: Array of translation vectors (N, 3).
        rvecs: Array of rotation vectors (N, 3).
        valid_mask: Boolean array indicating valid poses.
        filepath: Path to the trajectory file.
    """
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"File: {filepath}")
    print(f"Total frames: {len(timestamps)}")
    print(f"Valid poses: {np.sum(valid_mask)} ({100*np.sum(valid_mask)/len(valid_mask):.1f}%)")
    abnormal_mask = compute_abnormal_mask(tvecs, valid_mask)
    n_abnormal = int(np.sum(abnormal_mask))
    if n_abnormal > 0:
        print(f"Abnormal (Z<0, faded): {n_abnormal} "
              f"({100*n_abnormal/max(int(np.sum(valid_mask)),1):.1f}% of valid)")
    print(f"Duration: {timestamps[-1]:.2f} seconds")
    print(f"Average FPS: {len(timestamps)/timestamps[-1]:.2f}")

    if np.sum(valid_mask) > 0:
        valid_tvecs = tvecs[valid_mask]
        valid_rvecs = rvecs[valid_mask]

        print("\nPosition Statistics (valid frames only, camera in target frame):")
        print(f"  X: min={valid_tvecs[:, 0].min():.3f}, max={valid_tvecs[:, 0].max():.3f}, "
              f"mean={valid_tvecs[:, 0].mean():.3f}, std={valid_tvecs[:, 0].std():.3f}")
        print(f"  Y: min={valid_tvecs[:, 1].min():.3f}, max={valid_tvecs[:, 1].max():.3f}, "
              f"mean={valid_tvecs[:, 1].mean():.3f}, std={valid_tvecs[:, 1].std():.3f}")
        print(f"  Z: min={valid_tvecs[:, 2].min():.3f}, max={valid_tvecs[:, 2].max():.3f}, "
              f"mean={valid_tvecs[:, 2].mean():.3f}, std={valid_tvecs[:, 2].std():.3f}")

        rvecs_deg = np.degrees(valid_rvecs)
        print("\nRotation Statistics (valid frames only, camera in target frame, in degrees):")
        print(f"  Roll: min={rvecs_deg[:, 0].min():.2f}, max={rvecs_deg[:, 0].max():.2f}, "
              f"mean={rvecs_deg[:, 0].mean():.2f}, std={rvecs_deg[:, 0].std():.2f}")
        print(f"  Pitch: min={rvecs_deg[:, 1].min():.2f}, max={rvecs_deg[:, 1].max():.2f}, "
              f"mean={rvecs_deg[:, 1].mean():.2f}, std={rvecs_deg[:, 1].std():.2f}")
        print(f"  Yaw: min={rvecs_deg[:, 2].min():.2f}, max={rvecs_deg[:, 2].max():.2f}, "
              f"mean={rvecs_deg[:, 2].mean():.2f}, std={rvecs_deg[:, 2].std():.2f}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize 6-DOF pose trajectory data.'
    )
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Path to trajectory file (default: use latest)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['3d', '2d', 'both'],
        default='3d',
        help='Visualization mode (default: 3d)'
    )
    parser.add_argument(
        '--origin',
        type=str,
        choices=['target', 'camera'],
        default='target',
        help='Origin mode for 3D visualization (default: target)'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save visualization to file instead of displaying'
    )
    parser.add_argument(
        '--pose-interval',
        type=int,
        default=30,
        help='Interval (in frames) for drawing camera pose axes along trajectory (default: 30, 0 to disable)'
    )
    parser.add_argument(
        '--frustum-scale',
        type=float,
        default=0.16,
        help='Scale factor for ORB-SLAM style camera frustum (default: 0.16)'
    )
    parser.add_argument(
        '--pose-distance-threshold',
        type=float,
        default=0.5,
        help='Distance threshold in meters for drawing camera poses along trajectory (default: 0.5, set to 0 to use frame interval instead)'
    )
    args = parser.parse_args()

    # Get trajectory file
    if args.file:
        filepath = args.file
    else:
        filepath = get_latest_traj_file(SAVE_PATH)

    if filepath is None:
        print("Error: No trajectory file found.")
        sys.exit(1)

    print(f"Loading trajectory file: {filepath}")

    # Parse trajectory data
    result = parse_traj_file(filepath)
    if result is None:
        print("Error: Failed to parse trajectory file.")
        sys.exit(1)

    timestamps, tvecs, rvecs, valid_mask = result

    # Print summary
    print_trajectory_summary(timestamps, tvecs, rvecs, valid_mask, filepath)

    # Visualize
    if args.mode in ('3d', 'both'):
        save_path = args.save.replace('.png', '_3d.png') if args.save and args.mode == 'both' else args.save
        # Determine distance threshold (0 means use frame interval)
        distance_threshold = args.pose_distance_threshold if args.pose_distance_threshold > 0 else None
        visualize_3d_trajectory(timestamps, tvecs, rvecs, valid_mask,
                               origin_mode=args.origin, save_path=save_path,
                               pose_interval=args.pose_interval,
                               frustum_scale=args.frustum_scale,
                               pose_distance_threshold=distance_threshold)

    if args.mode in ('2d', 'both'):
        save_path = args.save.replace('.png', '_2d.png') if args.save and args.mode == 'both' else args.save
        visualize_2d_trajectory(timestamps, tvecs, rvecs, valid_mask, save_path=save_path)


if __name__ == "__main__":
    main()
