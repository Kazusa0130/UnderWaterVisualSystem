import cv2
import yaml
import numpy as np
from itertools import combinations
from config import *


class Solver:
    def __init__(self, config_path, obj_width=0.05, obj_length=0.05) -> None:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        self.intrinsic_matrix = np.array(config_data['Left']['CameraMatrix']['data']).reshape(3,3)
        self.dist_coeffs = np.array(config_data['Left']['distortion_coefficients']['data'])
        self.obj_width = obj_width
        self.obj_length = obj_length
        self.obj_points = np.array([
            (-0.100, -0.300, 0.000),   # 左上
            ( 0.100, -0.300, 0.000),   # 右上
            ( 0.245,  0.300, 0.000),   # 右下
            (-0.245,  0.300, 0.000),   # 左下
        ])
        self.obj_points_5 = np.array([
            (-0.100, -0.300, 0.000),   # 左上
            ( 0.100, -0.300, 0.000),   # 右上
            ( 0.245,  0.300, 0.000),   # 右下
            (-0.245,  0.300, 0.000),   # 左下
            (0.000,  0.000,  0.980)    # 中心点，位于目标平面更远离相机一侧
        ])
        self.tvec = None
        self.rvec = None
        # Mode semantics (public, mainly for visualization/CSV):
        #   0 = PnP (4-point or 5-point)
        #   1 = traditional feature tracking (track_mode=1)
        self.mode = 0
        self._pnp_num_points = 0  # internal: 4 or 5, for diagnostics/viz only
        self._last_reproj_error = None
        self._last_num_candidates = 0  # diagnostic: survivors after gate

    def _is_pose_physically_valid(
        self,
        tvec: np.ndarray,
        min_depth: float | None = None,
        max_range: float | None = None,
    ) -> bool:
        """Check whether a recovered translation is physically plausible.

        Uses configurable depth/range gates from ``config.py``.

        Args:
            tvec: Translation vector (3,) in the camera frame.
            min_depth: Optional override for ``PNP_MIN_DEPTH_M``.
            max_range: Optional override for ``PNP_MAX_RANGE_M``.

        Returns:
            True if the pose passes the depth and range checks.
        """
        t = np.asarray(tvec, dtype=float).flatten()
        if t.shape[0] < 3:
            return False
        if min_depth is None:
            min_depth = float(PNP_MIN_DEPTH_M)
        if max_range is None:
            max_range = float(PNP_MAX_RANGE_M)
        if t[2] < min_depth:
            return False
        if np.linalg.norm(t) >= max_range:
            return False
        return True

    def solver(self, points):
        """Solves PnP with the given image points.

        Automatically selects 4-point or 5-point mode based on the number
        of detected points.

        Args:
            points: List of (x, y) pixel coordinates. For 5-point mode the
                expected order is ``[corner0, corner1, corner2, corner3, center]``.

        Returns:
            Tuple of (success, rvec, tvec).
        """
        target_points = []
        self.mode = 0  # PnP mode
        if len(points) == 4:
            self._pnp_num_points = 4
            target_points = points
            success, rvec, tvec = self.solve_pnp(target_points)
        elif len(points) == 5:
            self._pnp_num_points = 5
            target_points = points
            success, rvec, tvec = self._solve_pnp_5p(target_points)
        elif len(points) > 5:
            # Defensive: reduce to the best 4 corners + 1 center.
            self._pnp_num_points = 5
            target_points = self._select_best_five(points)
            if target_points is None or len(target_points) != 5:
                print("Unable to reduce >5 points to a valid 5-point set.")
                return False, None, None
            success, rvec, tvec = self._solve_pnp_5p(target_points)
        else:
            print("Not enough points for PnP (need at least 4).")
            return False, None, None

        if not success:
            print("PnP solving failed.")
        return success, rvec, tvec

    def _select_best_five(
        self, points: list[tuple[float, float]]
    ) -> list[tuple[float, float]] | None:
        """Reduce >5 input points to 4 corners + 1 center.

        Uses the convex hull to identify corner candidates and selects the
        interior point closest to the hull centroid as the center.

        Args:
            points: List of (x, y) pixel coordinates.

        Returns:
            A list of 5 points ``[corner0..corner3, center]`` or ``None`` if
            no valid reduction is found.
        """
        pts = np.array(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 5:
            return None

        hull = cv2.convexHull(pts, returnPoints=True).reshape(-1, 2)
        if len(hull) != 4:
            # Hull is not a quadrilateral: fall back to the largest-area
            # 4-subset plus the centroid of the remaining points.
            best_area = -1.0
            best_quad = None
            for quad_idx in combinations(range(len(pts)), 4):
                quad = pts[list(quad_idx)]
                area = cv2.contourArea(quad.reshape(-1, 1, 2))
                if area > best_area:
                    best_area = area
                    best_quad = quad_idx
            if best_quad is None:
                return None
            corner_mask = np.zeros(len(pts), dtype=bool)
            corner_mask[list(best_quad)] = True
            corners = pts[corner_mask]
            center = np.mean(pts[~corner_mask], axis=0)
        else:
            corners = hull
            # Interior points are those not on the hull.
            hull_set = {tuple(p) for p in np.round(hull, 6)}
            interior = [p for p in pts if tuple(np.round(p, 6)) not in hull_set]
            if not interior:
                return None
            hull_centroid = np.mean(hull, axis=0)
            center = min(interior, key=lambda p: np.linalg.norm(p - hull_centroid))

        ordered_corners = self._order_corners_clockwise(corners)
        result = [tuple(p) for p in ordered_corners] + [tuple(center)]
        return result

    @staticmethod
    def _order_corners_clockwise(pts: np.ndarray) -> np.ndarray:
        """Order four corner points clockwise starting from top-left.

        Args:
            pts: Array of shape (4, 2).

        Returns:
            Clockwise ordered corners (top-left, top-right, bottom-right,
            bottom-left).
        """
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if len(pts) != 4:
            return pts
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        # Sort by angle; image y increases downward, so standard atan2 already
        # gives a clockwise ordering when plotted in image coordinates.
        ordered = pts[np.argsort(angles)]
        return ordered

    def _solve_pnp_5p(self, target_points):
        points = self.sort_points_(target_points)
        points = np.array([
            [points[0][0], points[0][1]],
            [points[1][0], points[1][1]],
            [points[2][0], points[2][1]],
            [points[3][0], points[3][1]],
            [points[4][0], points[4][1]]],
            dtype=np.double
        )
        success, self.rvec, self.tvec = cv2.solvePnP(
            self.obj_points_5,
            points,
            self.intrinsic_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )
        if success and self.rvec is not None and self.tvec is not None and self._is_pose_physically_valid(self.tvec):
            self.tvec = self.tvec.flatten()
            self.rvec = self.rvec.flatten()
            self._last_reproj_error = self._compute_reprojection_error(
                self.obj_points_5, points, self.rvec, self.tvec
            )
            # Reprojection-error gate (Ye Li et al. outlier filtering).
            if (
                REPROJ_ERROR_THRESHOLD is not None
                and self._last_reproj_error > REPROJ_ERROR_THRESHOLD
            ):
                success = False
            if success:
                return success, self.rvec, self.tvec
            return False, None, None
        else:
            return False, None, None

    def _compute_reprojection_error(
        self, obj_points: np.ndarray, img_points: np.ndarray,
        rvec: np.ndarray, tvec: np.ndarray
    ) -> float:
        """Mean pixel reprojection error for the given pose."""
        projected, _ = cv2.projectPoints(
            obj_points, rvec, tvec,
            self.intrinsic_matrix, self.dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        img_points = np.asarray(img_points).reshape(-1, 2)
        if projected.shape[0] != img_points.shape[0]:
            return float("inf")
        errors = np.linalg.norm(projected - img_points, axis=1)
        return float(np.mean(errors))

    def solve_pnp(self, target_points):
        points = self.sort_points_(target_points)
        points = np.array([
            [points[0][0], points[0][1]],
            [points[1][0], points[1][1]],
            [points[2][0], points[2][1]],
            [points[3][0], points[3][1]]],
            dtype=np.double
        )

        self._last_reproj_error = None
        self._last_num_candidates = 0

        if MULTIHYPOTHESIS_4P:
            ok, rvec, tvec = self._solve_pnp_4p_multi(points)
        else:
            ok, rvec, tvec, err = self._solve_pnp_iterative(points)
            if (
                ok
                and rvec is not None
                and tvec is not None
                and self._is_pose_physically_valid(tvec)
            ):
                if REPROJ_ERROR_THRESHOLD is None or err <= REPROJ_ERROR_THRESHOLD:
                    self.rvec = rvec
                    self.tvec = tvec
                    self._last_reproj_error = err
                    return True, self.rvec, self.tvec
            self.rvec = None
            self.tvec = None
            self._last_reproj_error = None
            return False, None, None

        return ok, (self.rvec if ok else None), (self.tvec if ok else None)

    def _solve_pnp_iterative(self, points):
        """4-point PnP using the iterative Levenberg-Marquardt solver.

        Returns:
            Tuple of (success, rvec, tvec, reproj_error). ``reproj_error`` is
            ``float("inf")`` when ``success`` is False.
        """
        success, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            points,
            self.intrinsic_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success and rvec is not None and tvec is not None:
            rvec = rvec.flatten()
            tvec = tvec.flatten()
            err = self._compute_reprojection_error(
                self.obj_points, points, rvec, tvec
            )
            return True, rvec, tvec, err
        return False, None, None, float("inf")

    def _solve_pnp_4p_multi(self, points):
        """Multi-hypothesis 4-point PnP (Ye Li et al., Ocean Engineering 2015).

        Tries all C(4,3)=4 three-point subsets via cv2.solveP3P (each yields up
        to 4 solutions), plus the full 4-point ITERATIVE solution. Candidates
        are gated by depth/range/reprojection; the lowest-error survivor wins.

        The 3-point subsets provide robustness when one detection is an
        outlier: the subset excluding it yields a clean solution that
        out-ranks the contaminated full-4-point solve.

        Returns:
            Tuple of (success, rvec, tvec). ``self._last_reproj_error`` and
            ``self._last_num_candidates`` are set as diagnostics.
        """
        threshold = (
            float(self._last_reproj_error)
            if REPROJ_ERROR_THRESHOLD is None
            else float(REPROJ_ERROR_THRESHOLD)
        )
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

        # --- 4 three-point subsets via dedicated P3P solver --------------
        # NOTE: cv2.solvePnP(..., SOLVEPNP_P3P) asserts npoints >= 4, so the
        # dedicated cv2.solveP3P (which takes exactly 3) must be used here.
        for subset in combinations(range(4), 3):
            idx = list(subset)
            num, rvecs, tvecs = cv2.solveP3P(
                self.obj_points[idx],
                points[idx],
                self.intrinsic_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_P3P,
            )
            for k in range(num):
                r_k = np.asarray(rvecs[k], dtype=float).flatten()
                t_k = np.asarray(tvecs[k], dtype=float).flatten()
                if not self._is_pose_physically_valid(t_k):
                    continue
                err = self._compute_reprojection_error(
                    self.obj_points, points, r_k, t_k
                )
                if err <= threshold:
                    candidates.append((err, r_k, t_k))

        # --- Full 4-point ITERATIVE as an additional candidate -----------
        ok, rvec_it, tvec_it, err_it = self._solve_pnp_iterative(points)
        if ok and err_it <= threshold:
            candidates.append((err_it, rvec_it, tvec_it))

        self._last_num_candidates = len(candidates)
        if not candidates:
            self._last_reproj_error = None
            return False, None, None

        candidates.sort(key=lambda c: c[0])
        best_err, best_rvec, best_tvec = candidates[0]
        self.rvec = best_rvec.copy()
        self.tvec = best_tvec.copy()
        self._last_reproj_error = best_err
        return True, self.rvec, self.tvec

    @staticmethod
    def rotation_matrix_to_euler_angles(R: np.ndarray) -> tuple[float, float, float]:
        """Extract ZYX Euler angles (roll, pitch, yaw) from a rotation matrix.

        Decomposition order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

        Args:
            R: 3x3 rotation matrix.

        Returns:
            Tuple of (roll, pitch, yaw) in radians.
        """
        R = np.array(R).reshape(3, 3)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0
        return roll, pitch, yaw

    @staticmethod
    def euler_angles_to_rotation_matrix(
        roll: float, pitch: float, yaw: float
    ) -> np.ndarray:
        """Construct rotation matrix from ZYX Euler angles (roll, pitch, yaw).

        Rotation order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

        Args:
            roll: Rotation around X axis (radians).
            pitch: Rotation around Y axis (radians).
            yaw: Rotation around Z axis (radians).

        Returns:
            3x3 rotation matrix.
        """
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )
        Ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )
        Rz = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        return Rz @ Ry @ Rx

    # Permutation mapping the docking output-frame axes to the standard ZYX
    # axes used by the Euler helpers:
    #   output Z (line of sight) -> roll-about-X
    #   output X (lateral / left) -> pitch-about-Y
    #   output Y (vertical / down) -> yaw-about-Z
    _DOCK_AXIS_PERM = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    # Native solvePnP target frame (X-right, Y-down, Z-away-from-camera) ->
    # docking output frame O (X-left, Y-down, Z-toward-camera).
    # This is a 180 deg rotation about the Y (down) axis: diag(-1, 1, -1).
    _M_NATIVE_TO_OUT = np.diag([-1.0, 1.0, -1.0])
    # Orientation of an ideal head-on, upright camera expressed in the output
    # frame O.  Equal to the frame map itself: the head-on camera looks toward
    # the target (camera +Z = -Z_o) with image-up aligned to target-up.
    _R_HEADON_OUT = np.diag([-1.0, 1.0, -1.0])

    @staticmethod
    def docking_euler_to_rotation_output(
        roll: float, pitch: float, yaw: float
    ) -> np.ndarray:
        """Reconstruct camera rotation in the docking output frame from RPY.

        Inverse of the decomposition performed by
        :meth:`get_camera_pose_euler_in_target_frame`. Given the docking
        ``(roll, pitch, yaw)`` (radians) it returns the camera's rotation
        matrix in the output frame O (X-left, Y-down, Z-toward-camera),
        suitable for drawing camera axes/frustums.

        Args:
            roll: Bank about the line-of-sight axis (radians).
            pitch: Tilt about the lateral axis (radians).
            yaw: Heading about the vertical (down) axis (radians).

        Returns:
            3x3 rotation matrix of the camera in the output frame O.
        """
        R_std = Solver.euler_angles_to_rotation_matrix(roll, pitch, yaw)
        P = Solver._DOCK_AXIS_PERM
        R_delta = P.T @ R_std @ P
        return R_delta @ Solver._R_HEADON_OUT

    def visualize_pose(self, image, length=0.1, show_rotation=True,
                       pose_text=None):
        """Draw the projected pose axes and a numeric readout on the image.

        Args:
            image: BGR image to draw on (modified in place).
            length: Axis length in meters for the reprojected axes.
            show_rotation: If True, draw the rotation axes and the
                Yaw/Pitch/Roll lines; if False, only the X/Y/Z translation.
            pose_text: Optional ``(x, y, z, roll, pitch, yaw)`` in the docking
                output frame O (position in meters, angles in **degrees**,
                ``(-180, 180]``). When provided, the on-image text uses these
                values so the overlay matches the serial/CSV output exactly.
                When None, falls back to decomposing ``self.rvec``/``self.tvec``
                in the raw camera frame (legacy behavior).
        """
        if pose_text is not None:
            x_t, y_t, z_t, roll_deg, pitch_deg, yaw_deg = pose_text
        if show_rotation:
            # Reproject 3D axes to image plane
            axis_points_4 = np.float32([
                [0, 0, 0],
                [length, 0, 0],
                [0, length, 0],
                [0, 0, length]
            ]).reshape(-1, 3)
            img_points, _ = cv2.projectPoints(
                axis_points_4, self.rvec, self.tvec,
                self.intrinsic_matrix, self.dist_coeffs
            )
            img_points = img_points.reshape(-1, 2).astype(np.int32)
            origin = (int(img_points[0, 0]), int(img_points[0, 1]))
            x_axis = (int(img_points[1, 0]), int(img_points[1, 1]))
            y_axis = (int(img_points[2, 0]), int(img_points[2, 1]))
            z_axis = (int(img_points[3, 0]), int(img_points[3, 1]))

            if pose_text is None:
                # Legacy: decompose the raw camera-frame rotation/translation.
                rotation_matrix, _ = cv2.Rodrigues(self.rvec)
                roll, pitch, yaw = Solver.rotation_matrix_to_euler_angles(
                    rotation_matrix
                )
                yaw_deg = np.degrees(yaw)
                pitch_deg = np.degrees(pitch)
                roll_deg = np.degrees(roll)
                x_t, y_t, z_t = self.tvec[0], self.tvec[1], self.tvec[2]

            cv2.arrowedLine(image, origin, x_axis, (0, 0, 255), 3)
            cv2.arrowedLine(image, origin, y_axis, (0, 255, 0), 3)
            cv2.arrowedLine(image, origin, z_axis, (255, 0, 0), 3)
            cv2.putText(image, f"Yaw: {yaw_deg:.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Pitch: {pitch_deg:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Roll: {roll_deg:.2f}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, f"X: {x_t:.2f}m", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Y: {y_t:.2f}m", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Z: {z_t:.2f}m", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            if pose_text is None:
                x_t, y_t, z_t = self.tvec[0], self.tvec[1], self.tvec[2]
            cv2.putText(image, f"X: {x_t:.2f}m", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Y: {y_t:.2f}m", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Z: {z_t:.2f}m", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return image

    def point_selector(self, points):
        best_area = -1.0
        best_points = []
        remain_point = None

        for idx in range(5):
            candidate = [points[j] for j in range(5) if j != idx]
            center_x = sum(point[0] for point in candidate) / 4.0
            center_y = sum(point[1] for point in candidate) / 4.0
            ordered_candidate = sorted(
                candidate,
                key=lambda point: np.arctan2(point[1] - center_y, point[0] - center_x)
            )

            contour = np.array(ordered_candidate, dtype=np.float32).reshape(-1, 1, 2)
            area = cv2.contourArea(contour)

            if area > best_area:
                best_area = area
                best_points = ordered_candidate
                remain_point = points[idx]

        best_points.append(remain_point)
        return best_points

    def sort_points_(self, points) -> np.ndarray:
        """Order 2D image points for PnP.

        - 4 points: order the four corners clockwise starting near top-left.
        - 5 points: treat the last point as the centre and order the first four
          corners clockwise; the centre is appended unchanged.
        - More than 5 points: should already have been reduced by
          :meth:`_select_best_five`; this method orders the first four corners
          and appends any remaining points as-is.

        Args:
            points: List/array of ``(x, y)`` image coordinates.

        Returns:
            ``np.ndarray`` of shape ``(N, 2)`` with the same number of points.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        n = len(pts)
        if n == 4:
            return self._order_corners_clockwise(pts)

        if n == 5:
            selected = self.point_selector(pts.tolist())
            return np.asarray(selected, dtype=float).reshape(-1, 2)

        # Defensive fallback (should not happen in normal flow).
        corners = self._order_corners_clockwise(pts[:4])
        if n > 4:
            corners = np.vstack([corners, pts[4:]])
        return corners

    def get_camera_pose_in_target_frame(
        self,
        rvec: np.ndarray | None = None,
        tvec: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Calculate camera pose in target coordinate frame.

        OpenCV solvePnP returns rvec, tvec representing target pose in camera
        frame. This method converts to camera pose in target frame, consistent
        with the ``target-as-origin`` visualization mode in
        :class:`visualizer_3d.PoseVisualizer3D`.

        Args:
            rvec: Target rotation vector in camera frame (3,). Uses
                :attr:`self.rvec` if None.
            tvec: Target translation vector in camera frame (3,). Uses
                :attr:`self.tvec` if None.

        Returns:
            Tuple of ``(rvec_cam, tvec_cam)`` where:
                - ``rvec_cam``: Rotation vector of camera in target frame.
                - ``tvec_cam``: Position of camera in target frame.
            Returns ``(None, None)`` if inputs are invalid.
        """
        if rvec is None:
            rvec = self.rvec
        if tvec is None:
            tvec = self.tvec
        if rvec is None or tvec is None:
            return None, None

        R_target_to_cam, _ = cv2.Rodrigues(np.array(rvec).flatten())
        R_cam_to_target = R_target_to_cam.T
        rvec_cam, _ = cv2.Rodrigues(R_cam_to_target)
        tvec_cam = -R_cam_to_target @ np.array(tvec).flatten()
        return rvec_cam.flatten(), tvec_cam.flatten()

    def get_camera_pose_euler_in_target_frame(
        self,
        rvec: np.ndarray | None = None,
        tvec: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None, float | None, float | None]:
        """Calculate camera pose + docking RPY in the output target frame.

        Output frame O (origin at target, axes defined w.r.t. the camera
        frame directions):
            - X: left  (positive toward image left)
            - Y: down  (positive toward the ground)
            - Z: toward camera (normal to target plane, positive in front)

        This frame is obtained from the native OpenCV solvePnP target frame
        (X-right, Y-down, Z-away-from-camera) by a 180 deg rotation about the
        Y (down) axis: ``M = diag(-1, 1, -1)``.

        ``tvec_cam`` is the camera position in this output frame.

        The returned ``(roll, pitch, yaw)`` describe the camera's orientation
        relative to a head-on, upright pose (camera looking straight at the
        target with image-up aligned to target-up), following the right-hand
        rule about the output-frame axes. All three are wrapped to
        ``(-pi, pi]`` (i.e. -180..180 deg):

            - ``yaw``:   heading about the vertical (down) axis. Target
              centered = 0; when the camera pans **right** (target moves to the
              image **left**) yaw is **positive**; panning left makes it
              negative.
            - ``pitch``: tilt about the lateral axis. Parallel = 0; right-hand
              rotation is positive, the opposite is negative.
            - ``roll``:  bank about the line-of-sight axis. Parallel = 0;
              right-hand rotation is positive, the opposite is negative.

        A perfect head-on, upright camera therefore yields
        ``roll = pitch = yaw = 0``.

        Args:
            rvec: Target rotation vector in camera frame (3,). Uses
                :attr:`self.rvec` if None.
            tvec: Target translation vector in camera frame (3,). Uses
                :attr:`self.tvec` if None.

        Returns:
            Tuple of ``(rvec_cam, tvec_cam, roll, pitch, yaw)``. All three
            angles are in radians, wrapped to ``(-pi, pi]``. ``rvec_cam`` and
            ``tvec_cam`` are expressed in the output frame O. Returns
            ``(None, None, None, None, None)`` if inputs are invalid.
        """
        rvec_cam, tvec_cam = self.get_camera_pose_in_target_frame(rvec, tvec)
        if rvec_cam is None:
            return None, None, None, None, None

        # Camera rotation in the native OpenCV target frame
        # (X=right, Y=down, Z=away-from-camera).
        R_cam_to_target_native, _ = cv2.Rodrigues(np.array(rvec_cam).flatten())

        # Transform to the output frame O (X-left, Y-down, Z-toward-camera):
        # a 180 deg rotation about Y (down), i.e. diag(-1, 1, -1).
        M = Solver._M_NATIVE_TO_OUT
        R_cam_out = M @ R_cam_to_target_native
        tvec_out = M @ np.array(tvec_cam).flatten()

        # Orientation of an ideal head-on, upright camera in the output frame.
        R_ideal_out = Solver._R_HEADON_OUT

        # Deviation of the actual camera from the head-on pose.
        R_delta = R_cam_out @ R_ideal_out.T

        # Re-express the deviation so that docking axes line up with the
        # standard ZYX (roll-about-X, pitch-about-Y, yaw-about-Z) used by
        # :meth:`rotation_matrix_to_euler_angles`:
        #   output Z (line of sight)  -> roll
        #   output X (lateral / left) -> pitch
        #   output Y (vertical / down) -> yaw
        P = Solver._DOCK_AXIS_PERM
        R_std = P @ R_delta @ P.T

        roll, pitch, yaw = Solver.rotation_matrix_to_euler_angles(R_std)

        # Wrap all three angles to (-pi, pi] (i.e. -180..180 deg): head-on /
        # parallel = 0, right-hand rotation about each output axis is positive,
        # the opposite direction is negative. rotation_matrix_to_euler_angles
        # already returns atan2 values, but normalize explicitly to guarantee
        # the half-open (-pi, pi] range.
        def _wrap(angle: float) -> float:
            angle = (angle + np.pi) % (2.0 * np.pi) - np.pi
            # Map -pi to +pi so the range is (-pi, pi].
            if angle <= -np.pi + 1e-12:
                angle = np.pi
            return angle

        roll = _wrap(roll)
        pitch = _wrap(pitch)
        yaw = _wrap(yaw)

        rvec_out, _ = cv2.Rodrigues(R_cam_out)

        return rvec_out.flatten(), tvec_out.flatten(), roll, pitch, yaw

    # ------------------------------------------------------------------
    # Threshold-fallback pose recovery (close-range PnP failure).
    # ------------------------------------------------------------------
    def solve_fallback_from_centroid(
        self,
        image_center: tuple[float, float],
        target_center: tuple[float, float],
        target_area_px: float,
        image_size: tuple[int, int],
        physical_area_m2: float,
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """Estimate camera position from a single blob centroid + area.

        Used when YOLO/PnP fails at close range but a bright target blob is
        still visible.  The method assumes the target is roughly fronto-parallel
        and uses the pinhole projection of a known physical area to estimate
        depth, then backs out lateral/vertical offsets from the centroid shift.

        Args:
            image_center: ``(cx, cy)`` optical center in pixels.
            target_center: ``(tx, ty)`` target centroid in pixels.
            target_area_px: Projected area of the target blob in pixels².
            image_size: ``(width, height)`` of the image.
            physical_area_m2: Known physical area of the target in m².

        Returns:
            ``(success, tvec_cam, rvec_cam)``.  ``tvec_cam`` is the camera
            position in the docking output frame O.  ``rvec_cam`` is returned
            as ``None`` because orientation cannot be estimated from a single
            blob.  ``success`` is False when the area is non-positive or the
            recovered depth is outside the configured gate.
        """
        if target_area_px <= 0 or physical_area_m2 <= 0:
            return False, None, None

        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        f_avg = (fx + fy) * 0.5

        # Depth from area ratio (fronto-parallel approximation).
        z = f_avg * np.sqrt(physical_area_m2 / target_area_px)
        if not self._is_pose_physically_valid(np.array([0.0, 0.0, z])):
            return False, None, None

        cx, cy = image_center
        tx, ty = target_center
        dx_px = tx - cx
        dy_px = ty - cy

        # Output frame O: X-left, Y-down, Z-toward-camera.
        x_out = -(dx_px / fx) * z
        y_out = (dy_px / fy) * z
        tvec_out = np.array([x_out, y_out, z], dtype=float)

        return True, None, tvec_out
