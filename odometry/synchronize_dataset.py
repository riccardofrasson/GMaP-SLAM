import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def sync_sequence(sequence_dir: Path):
    """
    Synchronizes data for each sequence, distinguishing between "Loop" sequences
    (3 sensors + IMU) and others (2 channels + IMU).
    """
    print(f"--- Processing Sequence: {sequence_dir.name} ---")

    # --- COMMON SETUP ---
    gt_path = sequence_dir / "relative_poses.csv"
    lidar_ts_path = sequence_dir / "lidar" / "timestamps.txt"
    imu_data_path = sequence_dir / "imu" / "IMU.txt"
    output_path = sequence_dir / "synchronized_indices.csv"

    if not all([gt_path.exists(), lidar_ts_path.exists(), imu_data_path.exists()]):
        print(
            f"  -> ⚠️ Warning: Base files (GT, LiDAR, IMU) missing. Skipping Sequence."
        )
        return

    try:
        lidar_df = pd.read_csv(lidar_ts_path, header=None, names=["lidar_ts"])
        lidar_df["lidar_index"] = lidar_df.index

        temp_imu_df = pd.read_csv(
            imu_data_path,
            sep=" ",
            header=None,
            usecols=[1, 2],
            names=["ts_seconds", "ts_nanoseconds"],
        )
        temp_imu_df["imu_ts"] = (
            temp_imu_df["ts_seconds"] + temp_imu_df["ts_nanoseconds"] / 1e9
        )
        imu_df = temp_imu_df[["imu_ts"]].copy()
        imu_df["imu_index"] = imu_df.index

        gt_df = pd.read_csv(gt_path)
        master_lidar_ts = gt_df[["timestamp"]].rename(columns={"timestamp": "lidar_ts"})
    except Exception as e:
        print(f"  -> ⚠️ ERROR while loading files: {e}")
        return

    # Initial synchronization between GT and LiDAR (common to both branches)
    master_lidar_ts.sort_values("lidar_ts", inplace=True)
    lidar_df.sort_values("lidar_ts", inplace=True)
    imu_df.sort_values("imu_ts", inplace=True)

    master_df = pd.merge_asof(
        master_lidar_ts, lidar_df, on="lidar_ts", direction="nearest", tolerance=0.02
    )
    master_df.dropna(inplace=True)

    # --- TWO-BRANCH LOGIC ---
    is_loop_sequence = "loop" in sequence_dir.name.lower()

    if is_loop_sequence:
        print("  -> Found 'Loop' sequence. Synchronizing 3 sensors + IMU.")

        # 1. Load Loop-specific files (separate Camera and Radar)
        cam_ts_path = sequence_dir / "image_left" / "timestamps.txt"
        radar_ts_path = sequence_dir / "radar" / "timestamps.txt"
        if not all([cam_ts_path.exists(), radar_ts_path.exists()]):
            print(
                "  -> ⚠️ Warning: Camera or Radar timestamps missing. Skipping Sequence."
            )
            return

        cam_df = pd.read_csv(cam_ts_path, header=None, names=["cam_ts"])
        cam_df["camera_index"] = cam_df.index
        radar_df = pd.read_csv(radar_ts_path, header=None, names=["radar_ts"])
        radar_df["radar_index"] = radar_df.index
        cam_df.sort_values("cam_ts", inplace=True)
        radar_df.sort_values("radar_ts", inplace=True)

        # 2. Synchronize LiDAR-Camera and LiDAR-Radar
        synced_cam = pd.merge_asof(
            master_df,
            cam_df,
            left_on="lidar_ts",
            right_on="cam_ts",
            direction="nearest",
            tolerance=0.05,
        )
        synced_radar = pd.merge_asof(
            master_df,
            radar_df,
            left_on="lidar_ts",
            right_on="radar_ts",
            direction="nearest",
            tolerance=0.05,
        )

        # 3. Merge results to get the triplet
        base_synced_df = pd.merge(
            synced_cam.dropna()[["lidar_index", "camera_index", "lidar_ts"]],
            synced_radar.dropna()[["lidar_index", "radar_index"]],
            on="lidar_index",
            how="inner",
        )
        print(
            f"[DEBUG] Found {len(base_synced_df)} synchronized triplets (LiDAR, Camera, Radar)."
        )
        output_cols = [
            "lidar_index",
            "camera_index",
            "radar_index",
            "imu_start_index",
            "imu_end_index",
        ]

    else:
        print("  -> Non-Loop sequence. Synchronizing 2 channels + IMU.")

        # 1. Load specific files (unified Camera/Radar)
        cam_radar_ts_path = sequence_dir / "image_left" / "timestamps.txt"
        if not cam_radar_ts_path.exists():
            print("  -> ⚠️ Warning: Camera/Radar timestamps missing. Skipping Sequence.")
            return

        cam_radar_df = pd.read_csv(cam_radar_ts_path, header=None, names=["cam_ts"])
        cam_radar_df["camera_radar_index"] = cam_radar_df.index
        cam_radar_df.sort_values("cam_ts", inplace=True)

        # 2. Synchronize LiDAR-Camera/Radar
        base_synced_df = pd.merge_asof(
            master_df,
            cam_radar_df,
            left_on="lidar_ts",
            right_on="cam_ts",
            direction="nearest",
            tolerance=0.05,
        )
        base_synced_df.dropna(inplace=True)
        print(
            f"[DEBUG] Found {len(base_synced_df)} synchronized pairs (LiDAR, Camera/Radar)."
        )
        output_cols = [
            "lidar_index",
            "camera_radar_index",
            "imu_start_index",
            "imu_end_index",
        ]

    # --- IMU SYNCHRONIZATION (common to both branches) ---
    print("[DEBUG] Starting IMU interval synchronization...")
    final_df = pd.DataFrame()
    if not base_synced_df.empty:
        # Find the next LiDAR frame's timestamp to define the interval
        lidar_df_sorted = lidar_df.sort_values("lidar_ts").reset_index(drop=True)
        lidar_df_sorted["next_lidar_ts"] = lidar_df_sorted["lidar_ts"].shift(-1)

        temp_df = pd.merge(
            base_synced_df,
            lidar_df_sorted[["lidar_index", "next_lidar_ts"]],
            on="lidar_index",
        )

        imu_sync_results = []
        for _, row in temp_df.iterrows():
            if pd.notna(row["next_lidar_ts"]):
                start_ts, end_ts = row["lidar_ts"], row["next_lidar_ts"]
                imu_chunk = imu_df[
                    (imu_df["imu_ts"] >= start_ts) & (imu_df["imu_ts"] < end_ts)
                ]

                if not imu_chunk.empty:
                    # Dynamically build the results dict from the pre-synced columns
                    result = row[output_cols[:-2]].to_dict()
                    result["imu_start_index"] = int(imu_chunk["imu_index"].iloc[0])
                    result["imu_end_index"] = int(imu_chunk["imu_index"].iloc[-1])
                    imu_sync_results.append(result)

        final_df = pd.DataFrame(imu_sync_results)

    print(f"[DEBUG] Found {len(final_df)} final frames with IMU interval.")

    # 4. Save
    if not final_df.empty:
        final_df = final_df[output_cols].astype(int)

    final_df.to_csv(output_path, index=False)
    print(f"\n--- FINAL RESULT ---")
    print(
        f"File '{output_path.name}' saved with {len(final_df)} synchronized data sets.\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize Lidar, Camera, Radar, and IMU data with branching logic."
    )
    parser.add_argument("root_dir", type=str, help="The root directory of the dataset.")
    args = parser.parse_args()

    root_path = Path(args.root_dir)
    if not root_path.is_dir():
        print(f"ERROR: Directory does not exist: {root_path}")
        return

    sequence_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    if not sequence_dirs:
        print(f"No sequences found in: {root_path}")
        return

    print(f"Found {len(sequence_dirs)} sequences. Starting synchronization...\n")
    for seq_dir in sequence_dirs:
        sync_sequence(seq_dir)
        print("-" * 30)
    print("--- Synchronization Complete! ---")


if __name__ == "__main__":
    main()
