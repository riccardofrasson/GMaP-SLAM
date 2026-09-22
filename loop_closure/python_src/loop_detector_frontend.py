import numpy as np

try:
    import loop_detector

    LOOP_DETECTOR_AVAILABLE = True
except ImportError:
    print(
        "WARNING: C++ loop_detector module not found (.so/.pyd). Loop detection will return 0 loops."
    )
    LOOP_DETECTOR_AVAILABLE = False

import math
import open3d as o3d  # Import Open3D
from scipy.spatial.transform import Rotation as R  # For Odometry Check
import scipy.spatial.distance  # For Cosine Camera
import time


# --- Helper Function to Calculate Pose Difference ---
def calculate_pose_difference(pose_mat_i, pose_mat_i_minus_1):
    """
    Calculates the relative transformation T_{i, i-1} = T_{i-1}^-1 * T_i
    and returns the magnitude of translation and the rotation angle.
    """
    try:
        pose_mat_i_f64 = pose_mat_i.astype(np.float64)
        pose_i_minus_1_f64 = pose_mat_i_minus_1.astype(np.float64)
        pose_i_minus_1_inv = np.linalg.inv(pose_i_minus_1_f64)
        T_relative = pose_i_minus_1_inv @ pose_mat_i_f64
        translation_diff_m = np.linalg.norm(T_relative[:3, 3])
        trace = np.trace(T_relative[:3, :3])
        # Clamp value to avoid potential domain errors with arccos
        cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        rotation_diff_rad = np.arccos(cos_angle)
        return float(translation_diff_m), float(rotation_diff_rad)
    except np.linalg.LinAlgError:
        return float("inf"), float("inf")
    except Exception as e:
        return float("inf"), float("inf")


# --- Modified PythonLoopDetector Class ---
class PythonLoopDetector:
    """
    Python class for multi-modal loop closure (Radar+Camera) with
    GPS ENU pre-filtering, GICP, Robust PCC, and OR-Logic.
    """

    def __init__(
        self,
        sc_dist_thresh=0.3,
        accum_dist_thresh=8.0,
        icp_fitness_thresh=0.6,
        icp_max_correspondence=1.0,
        odom_check_trans_thresh_m_per_m=0.15,
        odom_check_rot_thresh_rad_per_m=0.05,
        stationary_trans_thresh_m=0.05,
        stationary_rot_thresh_deg=0.5,
        post_stationary_cooldown_frames=10,
        cam_sim_thresh=0.95,
        use_gps_filter=False,
        gps_search_radius_m=25.0,
    ):
        try:
            self.sc_manager = loop_detector.SCManager()
        except AttributeError:
            print("\nCRITICAL ERROR: Could not create SCManager instance.")
            exit()

        self.sc_manager.set_sc_dist_thresh(sc_dist_thresh)
        self.sc_manager.set_azimuth_range(56.5)
        print(f"Loop Detector C++ Backend OK (SC Thresh: {sc_dist_thresh}).")

        # Data lists
        self.keyframe_poses = []
        self.keyframe_pointclouds = []
        self.keyframe_cam_features = []
        self.keyframe_gps_enu = []
        self.relative_odometry = []

        print(f"  Submapping: DISABLED (using single scans).")

        # Loop Detection Parameters
        self.MIN_FRAMES_AFTER_LOOP = 500
        self.MIN_FRAMES_INTERVAL = 500

        # Geometric Verification Parameters
        self.ICP_FITNESS_THRESH = icp_fitness_thresh
        self.ICP_MAX_CORRESPONDENCE = icp_max_correspondence
        self.ODOM_TRANS_THRESH_M_PER_M = odom_check_trans_thresh_m_per_m
        self.ODOM_ROT_THRESH_RAD_PER_M = odom_check_rot_thresh_rad_per_m
        print(
            f"  Strict Radar GICP Verification: Fitness >= {self.ICP_FITNESS_THRESH:.2f}, Max Corr <= {self.ICP_MAX_CORRESPONDENCE:.1f}m"
        )
        print(
            f"  Strict Odom Verification: Trans Err <= {self.ODOM_TRANS_THRESH_M_PER_M:.3f} m/m, Rot Err <= {self.ODOM_ROT_THRESH_RAD_PER_M:.3f} rad/m"
        )

        # --- GPS SETTINGS ---
        self.USE_GPS_FILTER = use_gps_filter
        self.GPS_SEARCH_RADIUS_M = gps_search_radius_m
        self.ACCUM_DISTANCE_THRESH = accum_dist_thresh
        if self.USE_GPS_FILTER:
            print(
                f"  Spatial Filter: GPS ENU (Radius: {self.GPS_SEARCH_RADIUS_M:.1f}m)"
            )
        else:
            print(
                f"  Spatial Filter: Odometry (Radius: {self.ACCUM_DISTANCE_THRESH:.1f}m)"
            )
        # ---

        # Stationary Filter Parameters
        self.STATIONARY_TRANS_THRESH_M = stationary_trans_thresh_m
        self.STATIONARY_ROT_THRESH_RAD = math.radians(stationary_rot_thresh_deg)
        print(
            f"  Stationary Filter (Pose): Trans < {self.STATIONARY_TRANS_THRESH_M:.3f}m AND Rot < {stationary_rot_thresh_deg:.2f}deg."
        )
        self.POST_STATIONARY_COOLDOWN_FRAMES = post_stationary_cooldown_frames
        self.last_frame_was_stationary = False
        self.frames_since_stationary_end = float("inf")
        print(
            f"  Post-Stationary Cooldown: {self.POST_STATIONARY_COOLDOWN_FRAMES} frames."
        )

        # Camera Parameters
        self.CAM_SIM_THRESH = cam_sim_thresh
        if self.CAM_SIM_THRESH < 1.0:
            print(
                f"  Camera Verification: ENABLED (Sim Thresh: {self.CAM_SIM_THRESH:.3f})"
            )
        else:
            print(f"  Camera Verification: DISABLED (Thresh >= 1.0)")

        # Internal State
        self.last_loop_kf_index = -1
        self.detected_loops = []

        # --- GPS/Fallback Debug Counters ---
        self.gps_filter_successes = 0
        self.gps_filter_fallbacks = 0
        # ---

    def add_keyframe(
        self,
        point_cloud_np,
        pose_np,
        relative_odom_np=None,
        camera_features_np=None,
        gps_enu_np=None,
    ):
        """Adds a keyframe with radar, camera, and optional GPS ENU data."""
        current_index = len(self.keyframe_poses)

        valid_gps_data = None
        if (
            gps_enu_np is not None
            and isinstance(gps_enu_np, np.ndarray)
            and gps_enu_np.shape == (2,)
            and not np.isnan(gps_enu_np).any()
        ):
            valid_gps_data = gps_enu_np.astype(np.float32)

        # Type and shape checks
        if (
            not isinstance(point_cloud_np, np.ndarray)
            or point_cloud_np.dtype != np.float32
        ):
            if isinstance(point_cloud_np, np.ndarray):
                point_cloud_np = point_cloud_np.astype(np.float32)
            else:
                raise TypeError(f"KF {current_index}: point_cloud_np is not NumPy!")
        if point_cloud_np.ndim != 2 or point_cloud_np.shape[1] != 4:
            raise ValueError(
                f"KF {current_index}: point_cloud_np has wrong shape {point_cloud_np.shape}"
            )
        if not isinstance(pose_np, np.ndarray) or pose_np.shape != (4, 4):
            raise TypeError(f"KF {current_index}: pose_np is not 4x4")
        pose_np = pose_np.astype(np.float32)

        # Ensure relative odometry
        final_relative_odom = None
        if current_index > 0:
            if relative_odom_np is not None:
                if isinstance(
                    relative_odom_np, np.ndarray
                ) and relative_odom_np.shape == (4, 4):
                    final_relative_odom = relative_odom_np.astype(np.float32)
            if final_relative_odom is None:  # Fallback: calculate from absolute poses
                try:
                    if current_index - 1 < len(self.keyframe_poses):
                        prev_pose_inv = np.linalg.inv(
                            self.keyframe_poses[current_index - 1]
                        )
                        final_relative_odom = (prev_pose_inv @ pose_np).astype(
                            np.float32
                        )
                    else:
                        print(f"ERROR KF {current_index}: Previous pose index invalid.")
                except (np.linalg.LinAlgError, IndexError):
                    final_relative_odom = None
            if final_relative_odom is None:  # Final fallback
                final_relative_odom = np.eye(4, dtype=np.float32)

        scan_to_use = point_cloud_np
        self.keyframe_poses.append(pose_np)
        self.keyframe_pointclouds.append(scan_to_use)
        self.keyframe_cam_features.append(camera_features_np)
        self.keyframe_gps_enu.append(valid_gps_data)
        if final_relative_odom is not None:
            self.relative_odometry.append(final_relative_odom)
        elif current_index > 0:
            print(f"Critical ERROR KF {current_index}: Missing relative odom!")
            self.relative_odometry.append(np.eye(4, dtype=np.float32))

        try:
            scan_data = (
                scan_to_use
                if scan_to_use.shape[0] > 0
                else np.zeros((0, 4), dtype=np.float32)
            )
            self.sc_manager.add_scan(scan_data)
        except Exception as e:
            print(f"C++ add_scan ERROR KF {current_index}: {e}")
            self.keyframe_poses.pop()
            self.keyframe_pointclouds.pop()
            self.keyframe_cam_features.pop()
            self.keyframe_gps_enu.pop()
            if self.relative_odometry:
                self.relative_odometry.pop()
            return

        self.detect_loop_for_keyframe(current_index)

    def find_candidates(self, current_index):
        """Finds candidates with Stationary, Cooldown, Interval, Spatial (Hybrid GPS ENU/Odom), and Odom Yaw filters."""
        candidates = []
        if (
            not self.keyframe_poses
            or current_index < 1
            or current_index < self.MIN_FRAMES_INTERVAL
        ):
            return candidates
        if current_index >= len(self.keyframe_poses):
            return candidates
        if (
            self.last_loop_kf_index != -1
            and (current_index - self.last_loop_kf_index) < self.MIN_FRAMES_AFTER_LOOP
        ):
            return []

        current_pose = self.keyframe_poses[current_index]
        prev_pose = self.keyframe_poses[current_index - 1]
        current_gps_enu = None
        try:
            current_gps_enu = self.keyframe_gps_enu[current_index]
        except IndexError:
            pass

        # Stationary and Cooldown filter for the current frame
        trans_diff_curr, rot_diff_curr = calculate_pose_difference(
            current_pose, prev_pose
        )
        is_calculation_valid = not (
            math.isinf(trans_diff_curr) or math.isinf(rot_diff_curr)
        )
        is_current_stationary = False
        if is_calculation_valid:
            is_current_stationary = (
                trans_diff_curr < self.STATIONARY_TRANS_THRESH_M
                and rot_diff_curr < self.STATIONARY_ROT_THRESH_RAD
            )

        if is_current_stationary:
            self.last_frame_was_stationary = True
            self.frames_since_stationary_end = 0
            return []
        else:
            if self.last_frame_was_stationary:
                self.frames_since_stationary_end = 1
            elif self.frames_since_stationary_end != float("inf"):
                self.frames_since_stationary_end += 1
            self.last_frame_was_stationary = False
            if self.frames_since_stationary_end <= self.POST_STATIONARY_COOLDOWN_FRAMES:
                return []

        MAX_YAW_DIFF_ODOM_RAD = np.radians(20.0)

        # Candidate Search
        for i in range(len(self.keyframe_poses)):
            # 1. Frame Interval Filter
            if (current_index - i) < self.MIN_FRAMES_INTERVAL:
                continue

            pose_i = self.keyframe_poses[i]
            pose_i_gps_enu = None
            try:
                pose_i_gps_enu = self.keyframe_gps_enu[i]
            except IndexError:
                pass

            # 2. Stationary Filter for 'i'
            if i > 0:
                if i >= len(self.keyframe_poses) or i - 1 < 0:
                    continue
                pose_i_minus_1 = self.keyframe_poses[i - 1]
                trans_diff_i, rot_diff_i = calculate_pose_difference(
                    pose_i, pose_i_minus_1
                )
                if math.isinf(trans_diff_i) or math.isinf(rot_diff_i):
                    continue
                if (
                    trans_diff_i < self.STATIONARY_TRANS_THRESH_M
                    and rot_diff_i < self.STATIONARY_ROT_THRESH_RAD
                ):
                    continue

            # --- HYBRID SPATIAL FILTER (WITH COUNTERS) ---
            use_gps_for_this_pair = (
                self.USE_GPS_FILTER
                and current_gps_enu is not None
                and pose_i_gps_enu is not None
            )

            if use_gps_for_this_pair:
                self.gps_filter_successes += 1
                # 3a. USE GPS ENU FILTER
                dist_spatial_enu = np.linalg.norm(current_gps_enu - pose_i_gps_enu)
                if dist_spatial_enu > self.GPS_SEARCH_RADIUS_M:
                    continue
            else:
                self.gps_filter_fallbacks += 1
                # 3b. USE ODOMETRY FILTER (Fallback)
                dist_spatial_xy_odom = np.linalg.norm(
                    current_pose[:2, 3] - pose_i[:2, 3]
                )
                if dist_spatial_xy_odom > self.ACCUM_DISTANCE_THRESH:
                    continue
            # --- END SPATIAL FILTER ---

            # 4. Relative Odometry Yaw Filter
            # Run this filter ONLY IF we are in the ODOMETRY FALLBACK case.
            # If GPS has already validated the position (use_gps_for_this_pair == True),
            # we trust the GPS and SKIP this odometry filter.
            if not use_gps_for_this_pair:
                try:
                    T_relative_odom = np.linalg.inv(
                        pose_i.astype(np.float64)
                    ) @ current_pose.astype(np.float64)
                    r = R.from_matrix(T_relative_odom[:3, :3])
                    yaw_odom_rad = r.as_euler("zyx", degrees=False)[0]
                    if abs(yaw_odom_rad) > MAX_YAW_DIFF_ODOM_RAD:
                        continue
                except (np.linalg.LinAlgError, ValueError):
                    continue

            # If all filters passed
            candidates.append(i)

        return candidates

    def calculate_camera_similarity(self, idx_j, idx_i):
        """Calculates cosine similarity between two camera features."""
        if self.CAM_SIM_THRESH >= 1.0:
            return None
        try:
            feat_j = self.keyframe_cam_features[idx_j]
            feat_i = self.keyframe_cam_features[idx_i]
            if feat_j is None or feat_i is None:
                return None
            if feat_j.ndim > 1:
                feat_j = feat_j.flatten()
            if feat_i.ndim > 1:
                feat_i = feat_i.flatten()
            if feat_j.shape[0] == 0 or feat_i.shape[0] == 0:
                return None
            norm_j = np.linalg.norm(feat_j)
            norm_i = np.linalg.norm(feat_i)
            if norm_j < 1e-6 or norm_i < 1e-6:
                return 0.0
            sim = np.dot(feat_j, feat_i) / (norm_j * norm_i)
            sim = np.clip(sim, -1.0, 1.0)
            if math.isnan(sim):
                return None
            return float(sim)
        except (IndexError, ValueError, TypeError) as e:
            return None

    def _run_odometry_check(self, idx_i, idx_j, T_icp_loop_curr):
        """Runs a strict odometry consistency check."""
        print(f"  Running Odometry Check...")
        num_required_odos = idx_j - idx_i
        start_odo_index = idx_i
        end_odo_index = idx_j - 1
        if num_required_odos <= 0:
            print(f"  FAILED [Odom Check]: Invalid index interval.")
            return False
        if (
            not self.relative_odometry
            or end_odo_index >= len(self.relative_odometry)
            or start_odo_index < 0
        ):
            print(
                f"  FAILED [Odom Check]: Odom indices out of range (Req: {start_odo_index}-{end_odo_index}, Len: {len(self.relative_odometry)})."
            )
            return False

        try:
            T_odom_ji_correct = np.eye(4, dtype=np.float64)
            accumulated_distance = 0.0
            for k in range(start_odo_index, end_odo_index + 1):
                if k >= len(self.relative_odometry):
                    raise IndexError(f"Odom index {k} out of range")
                T_k1_k = self.relative_odometry[k].astype(np.float64)
                if np.isnan(T_k1_k).any() or np.isinf(T_k1_k).any():
                    raise ValueError(f"Invalid relative odom {k}")
                T_odom_ji_correct = T_odom_ji_correct @ T_k1_k
                accumulated_distance += np.linalg.norm(T_k1_k[:3, 3])

            T_err = T_icp_loop_curr.astype(np.float64) @ T_odom_ji_correct
            err_trans = np.linalg.norm(T_err[:3, 3])
            _, err_rot_rad = calculate_pose_difference(T_err, np.eye(4))

            ABSOLUTE_ROT_THRESH_RAD = np.radians(30.0)
            if err_rot_rad > ABSOLUTE_ROT_THRESH_RAD:
                print(
                    f"  FAILED [Odom Check]: ABSOLUTE rotation error ({math.degrees(err_rot_rad):.1f} deg) > threshold."
                )
                return False

            if accumulated_distance < 0.1:
                err_trans_norm = err_trans
                err_rot_norm = err_rot_rad
            else:
                err_trans_norm = err_trans / accumulated_distance
                err_rot_norm = err_rot_rad / accumulated_distance

            if err_trans_norm > self.ODOM_TRANS_THRESH_M_PER_M:
                print(
                    f"  FAILED [Odom Check]: Normalized translation error ({err_trans_norm:.4f}) > threshold."
                )
                return False
            if err_rot_norm > self.ODOM_ROT_THRESH_RAD_PER_M:
                print(
                    f"  FAILED [Odom Check]: Normalized rotation error ({err_rot_norm:.4f}) > threshold."
                )
                return False

            print(f"  -> Normalized Odom Check Passed.")
            return True
        except (IndexError, ValueError, np.linalg.LinAlgError) as e:
            print(f"  INTERNAL ERROR during Odometry Check: {e}.")
            return False

    def _run_pcc_check(self, idx_i, idx_j, T_icp_loop_curr, gicp_fitness_score):
        """Runs a robust, weighted Pairwise Consistency Check (PCC)."""
        print(f"  Running Weighted Pairwise Consistency Check (PCC)...")
        if not self.detected_loops:
            print(f"  [PCC Check]: First loop, accepting.")
            return True
        total_consistent_weight = gicp_fitness_score
        total_inconsistent_weight = 0.0

        for loop_idx, (
            idx_k,
            idx_l,
            T_icp_kl,
            loop_fitness_kl,
            cam_flag_kl,
        ) in enumerate(self.detected_loops):
            try:
                T_lc_ij = T_icp_loop_curr.astype(np.float64)

                if idx_l >= len(self.keyframe_poses) or idx_i >= len(
                    self.keyframe_poses
                ):
                    continue
                pose_l = self.keyframe_poses[idx_l].astype(np.float64)
                pose_i = self.keyframe_poses[idx_i].astype(np.float64)
                T_odom_li = np.linalg.inv(pose_l) @ pose_i

                T_lc_kl_inv = np.linalg.inv(T_icp_kl.astype(np.float64))

                if idx_k >= len(self.keyframe_poses) or idx_j >= len(
                    self.keyframe_poses
                ):
                    continue
                pose_j = self.keyframe_poses[idx_j].astype(np.float64)
                pose_k = self.keyframe_poses[idx_k].astype(np.float64)
                T_odom_kj = np.linalg.inv(pose_k) @ pose_j

                T_err = T_lc_ij @ T_odom_li @ T_lc_kl_inv @ T_odom_kj

                err_trans_pcc = np.linalg.norm(T_err[:3, 3])
                _, err_rot_pcc_rad = calculate_pose_difference(T_err, np.eye(4))
                PCC_TRANS_THRESH = 1.0
                PCC_ROT_THRESH_RAD = np.radians(10.0)

                if (
                    err_trans_pcc < PCC_TRANS_THRESH
                    and err_rot_pcc_rad < PCC_ROT_THRESH_RAD
                ):
                    total_consistent_weight += loop_fitness_kl
                else:
                    total_inconsistent_weight += loop_fitness_kl
            except (IndexError, ValueError, np.linalg.LinAlgError) as e:
                pass

        print(
            f"    PCC Result: Consistency={total_consistent_weight:.3f} vs Inconsistency={total_inconsistent_weight:.3f}"
        )
        if total_consistent_weight >= total_inconsistent_weight:
            print(f"  -> PCC Test (Weighted Consensus) Passed.")
            return True
        else:
            print(f"  FAILED [PCC Check]: Loop is inconsistent with the majority.")
            return False

    def _get_odometry_yaw_guess(self, idx_i, idx_j):
        """Calculates relative yaw T_{i,j} (from i -> j) using odometry poses."""
        try:
            if idx_i >= len(self.keyframe_poses) or idx_j >= len(self.keyframe_poses):
                return 0.0
            pose_i = self.keyframe_poses[idx_i].astype(np.float64)
            pose_j = self.keyframe_poses[idx_j].astype(np.float64)
            T_relative_odom = np.linalg.inv(pose_i) @ pose_j
            r = R.from_matrix(T_relative_odom[:3, :3])
            yaw_odom_rad = r.as_euler("zyx", degrees=False)[0]
            return yaw_odom_rad
        except (IndexError, np.linalg.LinAlgError, ValueError) as e:
            return 0.0

    def verify_loop_geometrically(self, idx_curr, idx_loop, yaw_diff_rad_guess):
        """
        Performs GICP, Camera Check (Sanity) AND Radar Check (Strict),
        and accepts if EITHER ONE is valid (OR-Logic).
        Returns: is_valid, T_reg, gicp_fitness_score, cam_override_flag
        """
        idx_j = idx_curr
        idx_i = idx_loop  # j=current, i=loop

        # 1. PRE-CHECK SCANS
        try:
            scan_j_np = self.keyframe_pointclouds[idx_j]
            scan_i_np = self.keyframe_pointclouds[idx_i]
            if scan_j_np.shape[0] < 100 or scan_i_np.shape[0] < 100:
                print(
                    f"  FAILED [PreCheck]: Scan {idx_j} ({scan_j_np.shape[0]}pts) or {idx_i} ({scan_i_np.shape[0]}pts) is too small."
                )
                return False, None, 0.0, False
            cloud_j_xyz = scan_j_np[:, :3].astype(np.float64)
            cloud_i_xyz = scan_i_np[:, :3].astype(np.float64)
            pcd_j = o3d.geometry.PointCloud()
            pcd_j.points = o3d.utility.Vector3dVector(cloud_j_xyz)
            pcd_i = o3d.geometry.PointCloud()
            pcd_i.points = o3d.utility.Vector3dVector(cloud_i_xyz)
        except IndexError:
            print(f"  INTERNAL ERROR: Indices {idx_j}/{idx_i} out of range for Scans!")
            return False, None, 0.0, False

        # 2. GICP (COARSE + FINE)
        try:
            radius_normal = 10.0
            pcd_j.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius_normal, max_nn=30
                )
            )
            pcd_i.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius_normal, max_nn=30
                )
            )
            pcd_j.orient_normals_towards_camera_location()
            pcd_i.orient_normals_towards_camera_location()

            initial_guess_rot_only = np.eye(4)
            initial_guess_rot_only[:3, :3] = R.from_euler(
                "z", yaw_diff_rad_guess
            ).as_matrix()

            # --- MODIFICATION 1: ROBUST GICP COARSE ---
            ICP_COARSE_THRESH_METERS = 20.0  # Was 10.0

            reg_gicp_coarse = o3d.pipelines.registration.registration_generalized_icp(
                pcd_j,
                pcd_i,
                ICP_COARSE_THRESH_METERS,
                initial_guess_rot_only,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
                ),
            )  # Was 30

            if reg_gicp_coarse.fitness < 0.001:
                print(
                    f"  FAILED [GICP Coarse]: Fitness near-zero ({reg_gicp_coarse.fitness:.4f})."
                )
                return False, None, 0.0, False

            # --- MODIFICATION 2: TOLERANT GICP FINE ---
            ICP_FINE_THRESH_METERS = 5.0  # Replaces self.ICP_MAX_CORRESPONDENCE (0.5m)

            reg_gicp_fine = o3d.pipelines.registration.registration_generalized_icp(
                pcd_j,
                pcd_i,
                ICP_FINE_THRESH_METERS,
                reg_gicp_coarse.transformation,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100
                ),
            )
            # --- END GICP MODIFICATIONS ---

        except Exception as e:
            print(f"  CRITICAL ERROR during GICP: {e}")
            return False, None, 0.0, False

        T_icp_loop_curr = reg_gicp_fine.transformation
        gicp_fitness_score = reg_gicp_fine.fitness
        print(f"  GICP [FINE] complete: Fitness={gicp_fitness_score:.4f}")

        # 3. PARALLEL CHECKS
        # Path 1: Camera (Sanity GICP)
        is_cam_ok = False
        cam_sim = self.calculate_camera_similarity(idx_curr, idx_loop)
        if cam_sim is not None:
            print(f"  Camera Check: Similarity = {cam_sim:.4f}")
            if cam_sim >= self.CAM_SIM_THRESH:

                # --- MODIFICATION: Sanity check raised to avoid false positives ---
                MIN_FITNESS_CAM_OVERRIDE = self.ICP_FITNESS_THRESH  # Was 0.3

                if gicp_fitness_score >= MIN_FITNESS_CAM_OVERRIDE:
                    print(
                        f"  -> [CAM Check]: TRUE (Sim>=Thresh AND Fitness {gicp_fitness_score:.4f} >= {MIN_FITNESS_CAM_OVERRIDE:.2f})"
                    )
                    is_cam_ok = True
                else:
                    print(
                        f"  [CAM Check]: Sim OK, but GICP Sanity Check FAILED (Fitness {gicp_fitness_score:.4f} < {MIN_FITNESS_CAM_OVERRIDE:.2f})"
                    )

        # Path 2: Radar (Strict)
        is_radar_strict_ok = False
        print("\n  RADAR Validation (Strict)...")
        _, icp_rot_rad = calculate_pose_difference(T_icp_loop_curr, np.eye(4))
        MAX_ICP_ROTATION_RAD = np.radians(25.0)

        passes_rot = icp_rot_rad <= MAX_ICP_ROTATION_RAD
        if not passes_rot:
            print(
                f"  FAILED [Radar Check]: GICP Rotation ({math.degrees(icp_rot_rad):.1f} deg) > Threshold."
            )

        passes_fitness = gicp_fitness_score >= self.ICP_FITNESS_THRESH
        if passes_rot and not passes_fitness:
            print(
                f"  FAILED [Radar Check]: Strict GICP Fitness ({gicp_fitness_score:.4f}) < Threshold ({self.ICP_FITNESS_THRESH:.3f})."
            )

        passes_odom = (
            passes_rot
            and passes_fitness
            and self._run_odometry_check(idx_i, idx_j, T_icp_loop_curr)
        )

        passes_pcc = passes_odom and self._run_pcc_check(
            idx_i, idx_j, T_icp_loop_curr, gicp_fitness_score
        )

        if passes_pcc:
            is_radar_strict_ok = True
            print(f"  -> [Radar Check]: TRUE (All checks passed)")
        elif passes_rot and passes_fitness and passes_odom:
            print(f"  -> [Radar Check]: FALSE (Only PCC failed)")
        elif passes_rot and passes_fitness:
            print(f"  -> [Radar Check]: FALSE (Odom failed)")

        # 4. FINAL DECISION (OR-Logic)
        print("\n  --- Final Decision (OR-Logic) ---")
        is_super_loop = is_cam_ok and is_radar_strict_ok
        is_loop_valid = is_cam_ok or is_radar_strict_ok
        cam_override_flag = is_cam_ok

        if is_loop_valid:
            loop_type_str = (
                "SUPER LOOP"
                if is_super_loop
                else ("CAM Override" if is_cam_ok else "Radar Strict")
            )
            print(f"--- Loop {idx_j} -> {idx_i} VALIDATED ({loop_type_str}) ---")
            return True, T_icp_loop_curr, gicp_fitness_score, cam_override_flag
        else:
            print(f"  FAILED: No validation path passed.")
            return False, None, 0.0, False

    def detect_loop_for_keyframe(self, current_index):
        """
        Orchestrates pre-filtering (GPS/Odom), parallel voting (SC vs Camera),
        and sends the merged candidates to geometric verification.
        """
        # 1. Find candidates
        candidate_indices_spatial = self.find_candidates(current_index)
        if not candidate_indices_spatial:
            return

        candidates_to_verify = {}  # Dictionary: {loop_idx -> yaw_guess_rad}

        # --- Channel 1: RADAR Voting (Scan Context) ---
        try:
            j_sc, yaw_sc = self.sc_manager.detect_loop(
                current_index, candidate_indices_spatial
            )
        except Exception as e:
            print(f"C++ detect_loop ERROR KF {current_index}: {e}")
            j_sc = -1

        if j_sc != -1:
            MAX_YAW_DIFF_RAD_SC = np.radians(20.0)
            if abs(yaw_sc) <= MAX_YAW_DIFF_RAD_SC:
                print(
                    f"  -> Candidate from ScanContext: {current_index} -> {j_sc} (Yaw SC: {math.degrees(yaw_sc):.2f} deg)"
                )
                candidates_to_verify[j_sc] = yaw_sc

        # --- Channel 2: CAMERA Voting (Cosine Similarity) ---
        for j_cam in candidate_indices_spatial:
            if j_cam in candidates_to_verify:
                continue

            sim = self.calculate_camera_similarity(current_index, j_cam)

            if sim is not None and sim >= self.CAM_SIM_THRESH:
                # Use a neutral guess (0.0 rad) and rely on a robust GICP Coarse.
                yaw_odom_guess = 0.0
                print(
                    f"  -> Candidate from Camera: {current_index} -> {j_cam} (Sim: {sim:.3f}, Odom Yaw: DISABLED/NEUTRAL)"
                )
                candidates_to_verify[j_cam] = yaw_odom_guess

        # --- Verification Phase ---
        if not candidates_to_verify:
            return

        print(
            f"--- Found {len(candidates_to_verify)} unique candidates (SC+Cam). Starting GICP verification..."
        )

        has_found_loop_this_frame = False

        for loop_idx_final, yaw_guess in sorted(candidates_to_verify.items()):

            if has_found_loop_this_frame:
                break

            print(
                f"\n--- Verifying (GICP+OR-Logic) Candidate {current_index} -> {loop_idx_final} ---"
            )

            is_valid, T_reg, gicp_fitness, cam_override_flag = (
                self.verify_loop_geometrically(current_index, loop_idx_final, yaw_guess)
            )

            if is_valid:
                self.detected_loops.append(
                    (
                        current_index,
                        loop_idx_final,
                        T_reg.astype(np.float32),
                        gicp_fitness,
                        cam_override_flag,
                    )
                )
                self.last_loop_kf_index = current_index
                has_found_loop_this_frame = True

    def get_detected_loops(self):
        """Returns list of CONFIRMED loops (idx_curr, idx_loop, T_reg, fitness, cam_flag)."""
        return self.detected_loops
