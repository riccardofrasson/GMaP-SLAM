import pandas as pd
import numpy as np
from pathlib import Path
from pyproj import Proj
from scipy.spatial.transform import Rotation
import argparse
import gtsam


# ==============================================================================
# FUNCTION 1: Synchronization and Conversion
# ==============================================================================
def synchronize_and_convert_to_enu(camera_ts_path, rtk_gt_path):
    """
    Synchronizes RTK data and converts it to a local ENU system,
    using the first point of the sequence as the origin.
    """
    print("1. Loading and synchronizing data...")
    cam_df = pd.read_csv(camera_ts_path, header=None, names=["timestamp"])
    rtk_df = pd.read_csv(rtk_gt_path, sep=" ")

    cam_df.rename(columns={"timestamp": "lidar_ts"}, inplace=True)
    rtk_df.rename(columns={"timestamp": "rtk_ts"}, inplace=True)

    cam_df.sort_values("lidar_ts", inplace=True)
    rtk_df.sort_values("rtk_ts", inplace=True)

    if "rtk_ts" in rtk_df.columns:
        rtk_df["rtk_ts_original"] = rtk_df["rtk_ts"]
    else:
        print("Error: 'timestamp' column not found in RTK file.")
        return None

    synced_df = pd.merge_asof(
        cam_df,
        rtk_df,
        left_on="lidar_ts",
        right_on="rtk_ts",
        direction="nearest",
        tolerance=0.1,
    )
    synced_df.dropna(inplace=True)

    if synced_df.empty:
        print("Synchronization failed: no matching frames found.")
        return None

    synced_df["time_diff_seconds"] = (
        synced_df["lidar_ts"] - synced_df["rtk_ts_original"]
    ).abs()

    print(f"\n✅ Found {len(synced_df)} synchronized frames.")
    print(
        f"   -> Maximum time difference: {synced_df['time_diff_seconds'].max():.6f} seconds\n"
    )

    print("2. Converting to ENU coordinates...")
    origin_lat, origin_lon, origin_alt = synced_df.iloc[0][["lat", "lon", "alt"]]

    proj = Proj(proj="tmerc", lat_0=origin_lat, lon_0=origin_lon, ellps="WGS84")
    x, y = proj(synced_df["lon"].values, synced_df["lat"].values)
    z = synced_df["alt"].values - origin_alt

    synced_df["x_enu"] = x
    synced_df["y_enu"] = y
    synced_df["z_enu"] = z

    return synced_df


# ==============================================================================
# FUNCTION 2: The GTSAM Smoother
# ==============================================================================
def smooth_trajectory_with_gtsam(enu_points, timestamps):
    """
    Uses GTSAM, balancing weights between GPS measurements and a motion model,
    to obtain a smooth trajectory.
    """
    print("   -> 🚀 Starting smoothing with noise rebalancing...")
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # --- Key Tuning Parameters ---
    STATIONARY_THRESHOLD = 0.05
    SMOOTHING_WINDOW_SIZE = 7
    # -----------------------------

    gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))

    # DECREASE MOTION NOISE (dx component): Tell it to trust our smoothed velocity estimate MORE.
    # The array represents [roll, pitch, yaw, x, y, z]. We modify the 4th value (x).
    moving_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.deg2rad(2.0), np.deg2rad(1.0), np.deg2rad(0.5), 0.03, 0.02, 0.1])
    )
    # ==========================

    stationary_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.deg2rad(0.1), np.deg2rad(0.1), np.deg2rad(0.1), 0.01, 0.01, 0.01])
    )

    # --- STAGE 1: Pre-calculation and filtering of displacements ---
    displacements = np.linalg.norm(np.diff(enu_points, axis=0), axis=1)
    window = np.ones(SMOOTHING_WINDOW_SIZE) / SMOOTHING_WINDOW_SIZE
    smoothed_displacements_valid = np.convolve(displacements, window, mode="valid")
    pad_size = SMOOTHING_WINDOW_SIZE // 2
    smoothed_displacements = np.pad(
        smoothed_displacements_valid, (pad_size, pad_size), "edge"
    )
    # -----------------------------------------------------------

    last_valid_yaw = 0.0
    for i in range(len(enu_points)):
        key = gtsam.symbol("x", i)
        current_pos = enu_points[i]

        yaw_estimate = last_valid_yaw
        if i < len(enu_points) - 1:
            next_pos = enu_points[i + 1]
            if np.linalg.norm(next_pos - current_pos) > STATIONARY_THRESHOLD / 2:
                yaw_estimate = np.arctan2(
                    next_pos[1] - current_pos[1], next_pos[0] - current_pos[0]
                )
                last_valid_yaw = yaw_estimate

        pose_estimate = gtsam.Pose3(
            gtsam.Rot3.Ypr(yaw_estimate, 0, 0),
            gtsam.Point3(current_pos[0], current_pos[1], 0),
        )
        initial_estimate.insert(key, pose_estimate)
        graph.add(
            gtsam.GPSFactor(
                key, gtsam.Point3(current_pos[0], current_pos[1], 0), gps_noise
            )
        )

        if i > 0:
            prev_key = gtsam.symbol("x", i - 1)
            raw_displacement_this_step = np.linalg.norm(current_pos - enu_points[i - 1])

            if raw_displacement_this_step < STATIONARY_THRESHOLD:
                # Stationary factor
                pose_between = gtsam.Pose3()
                graph.add(
                    gtsam.BetweenFactorPose3(
                        prev_key, key, pose_between, stationary_noise
                    )
                )
            else:
                # Motion factor (using smoothed displacement)
                dx_smoothed = smoothed_displacements[i - 1]
                pose_between = gtsam.Pose3(
                    gtsam.Rot3(), gtsam.Point3(dx_smoothed, 0, 0)
                )
                graph.add(
                    gtsam.BetweenFactorPose3(prev_key, key, pose_between, moving_noise)
                )

    print("   -> ⚙️  Optimizing graph...")
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print("   -> ✅ Optimization complete!")

    optimized_poses = []
    for i in range(len(enu_points)):
        pose = result.atPose3(gtsam.symbol("x", i))
        optimized_poses.append(
            [pose.translation()[0], pose.translation()[1], pose.rotation().yaw()]
        )

    return np.array(optimized_poses)


# ==============================================================================
# FUNCTION 3: Main function orchestrating the GTSAM process
# ==============================================================================
def process_with_gtsam(df):
    """
    Takes the synchronized DataFrame, runs GTSAM smoothing,
    and calculates the final relative poses.
    """
    print("3. Processing trajectory with GTSAM...")
    enu_points = df[["x_enu", "y_enu"]].values
    timestamps = df["rtk_ts_original"].values

    # Run smoothing to get clean absolute poses [x, y, yaw]
    optimized_absolute_poses = smooth_trajectory_with_gtsam(enu_points, timestamps)

    print("4. Calculating relative poses from the optimized trajectory...")
    relative_poses_data = []
    for i in range(len(optimized_absolute_poses) - 1):
        # Reconstruct 4x4 transformation matrices from absolute poses
        # Pose i
        pos_i = np.array(
            [
                optimized_absolute_poses[i, 0],
                optimized_absolute_poses[i, 1],
                df.iloc[i]["z_enu"],
            ]
        )
        rot_i = Rotation.from_euler("z", optimized_absolute_poses[i, 2])
        T_world_to_i = np.eye(4)
        T_world_to_i[:3, :3] = rot_i.as_matrix()
        T_world_to_i[:3, 3] = pos_i

        # Pose i+1
        pos_i1 = np.array(
            [
                optimized_absolute_poses[i + 1, 0],
                optimized_absolute_poses[i + 1, 1],
                df.iloc[i + 1]["z_enu"],
            ]
        )
        rot_i1 = Rotation.from_euler("z", optimized_absolute_poses[i + 1, 2])
        T_world_to_i1 = np.eye(4)
        T_world_to_i1[:3, :3] = rot_i1.as_matrix()
        T_world_to_i1[:3, 3] = pos_i1

        # Calculate the relative pose T_i -> T_{i+1}
        T_relative = np.linalg.inv(T_world_to_i) @ T_world_to_i1

        translation = T_relative[:3, 3]
        quat = Rotation.from_matrix(T_relative[:3, :3]).as_quat()  # [x, y, z, w]

        row_data = {
            "timestamp": df.iloc[i]["lidar_ts"],
            "dx": translation[0],
            "dy": translation[1],
            "dz": translation[2],
            "qx": quat[0],
            "qy": quat[1],
            "qz": quat[2],
            "qw": quat[3],
        }
        relative_poses_data.append(row_data)

    return pd.DataFrame(relative_poses_data)


# ==============================================================================
# MAIN FUNCTION: Script execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Calculate relative poses from RTK data using GTSAM."
    )
    parser.add_argument("sequence_dir", type=str, help="Sequence folder to process.")
    args = parser.parse_args()

    sequence_path = Path(args.sequence_dir)
    cam_ts_path = sequence_path / "lidar" / "timestamps.txt"
    rtk_path = sequence_path / "rtk_ground_truth.txt"
    output_path = sequence_path / "relative_poses.csv"

    if not (cam_ts_path.exists() and rtk_path.exists()):
        print(f"Error: Files not found. Check paths:\n- {cam_ts_path}\n- {rtk_path}")
        return

    # Stage 1 & 2: Synchronization and ENU conversion
    synced_data = synchronize_and_convert_to_enu(cam_ts_path, rtk_path)

    if synced_data is not None and not synced_data.empty:
        # Stage 3 & 4: GTSAM smoothing and relative pose calculation
        relative_poses_df = process_with_gtsam(synced_data)

        relative_poses_df.to_csv(output_path, index=False, float_format="%.8f")
        print(f"\nOperation completed successfully! ✅")
        print(f"Optimized relative poses saved to: {output_path}")


if __name__ == "__main__":
    main()
