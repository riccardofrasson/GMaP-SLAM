import pandas as pd
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import struct

# Import necessary functions from the rosbags library
from rosbags.highlevel import AnyReader


def parse_point_cloud2_full(msg, target_fields=["x", "y", "z", "doppler", "power"]):
    """
    A manual parser that takes a deserialized PointCloud2 message object
    and extracts numerical data from its .data attribute.
    """
    DTYPE_MAP = {7: ("f", 4)}  # 7 = FLOAT32
    point_step = msg.point_step

    field_offsets = {}
    for field in msg.fields:
        if field.name in target_fields:
            dtype_char, dtype_size = DTYPE_MAP.get(field.datatype, (None, None))
            if not dtype_char:
                continue
            field_offsets[field.name] = field.offset

    if not all(f in field_offsets for f in target_fields):
        return None

    points = []
    data = msg.data
    num_points = msg.width * msg.height

    for i in range(num_points):
        try:
            # Unpack a single point's data
            point_data = [
                struct.unpack_from("<f", data, i * point_step + field_offsets[f])[0]
                for f in target_fields
            ]
            if np.isfinite(point_data).all():
                points.append(point_data)
        except (struct.error, IndexError):
            # Stop if data is corrupt or incomplete
            break

    return np.array(points, dtype=np.float32)


def process_bag_file(bag_path, output_dir):
    sequence_path = output_dir / bag_path.stem
    print(
        f"\n--- Starting extraction from: {bag_path.name} (including Ground Truth) ---"
    )
    print(f"Data will be saved to: {sequence_path}")

    topic_map = {
        "/camera_array/left/image_raw": "image_left",
        "/camera_array/right/image_raw": "image_right",
        "/ouster/points": "lidar",
        "/oculii_radar/point_cloud": "radar",
        "/fix": "rtk_fix",
        "/vel": "rtk_vel",
    }

    # Prepare for extraction
    ts_files, counters = {}, {}
    for sensor_type in ["image_left", "image_right", "lidar", "radar"]:
        sensor_path = sequence_path / sensor_type
        sensor_path.mkdir(parents=True, exist_ok=True)
        ts_files[sensor_type] = open(sensor_path / "timestamps.txt", "w")
        counters[sensor_type] = 0

    # Lists to collect navigation data
    rtk_fix_data = []
    rtk_vel_data = []

    try:
        with AnyReader([bag_path]) as reader:
            for connection, timestamp, rawdata in tqdm(
                reader.messages(),
                total=reader.message_count,
                desc=f"Processing {bag_path.name}",
            ):
                msg = reader.deserialize(rawdata, connection.msgtype)
                topic = connection.topic
                if topic not in topic_map:
                    continue

                sensor_type = topic_map.get(topic)
                ts_sec = timestamp / 1e9  # Convert nanoseconds to seconds

                # --- Standard Sensor Data Extraction ---
                if sensor_type in ts_files:
                    count = counters[sensor_type]
                    counters[sensor_type] += 1
                    path = sequence_path / sensor_type
                    ts_files[sensor_type].write(f"{ts_sec}\n")

                    if "image" in sensor_type:
                        cv_image = msg.data.reshape((msg.height, msg.width, -1))
                        if msg.encoding == "rgb8":
                            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(path / f"{count:06d}.png"), cv_image)

                    elif sensor_type == "lidar":
                        points = parse_point_cloud2_full(
                            msg, target_fields=["x", "y", "z"]
                        )
                        if points is not None and len(points) > 0:
                            # Save directly as a numpy array. It's simpler and more robust.
                            np.save(str(path / f"{count:06d}.npy"), points)

                    elif sensor_type == "radar":
                        points_full = parse_point_cloud2_full(msg)
                        if points_full is not None and len(points_full) > 0:
                            np.save(str(path / f"{count:06d}.npy"), points_full)

                # ---  Collect RTK Data ---
                elif sensor_type == "rtk_fix":
                    rtk_fix_data.append(
                        [ts_sec, msg.latitude, msg.longitude, msg.altitude]
                    )

                elif sensor_type == "rtk_vel":
                    rtk_vel_data.append(
                        [
                            ts_sec,
                            msg.twist.linear.x,
                            msg.twist.linear.y,
                            msg.twist.linear.z,
                        ]
                    )

    finally:
        for f in ts_files.values():
            f.close()

    # ---  Post-processing and saving Ground Truth ---
    if rtk_fix_data and rtk_vel_data:
        print("Synchronizing and saving RTK Ground Truth...")
        fix_df = pd.DataFrame(rtk_fix_data, columns=["timestamp", "lat", "lon", "alt"])
        vel_df = pd.DataFrame(rtk_vel_data, columns=["timestamp", "vx", "vy", "vz"])

        # Sort by timestamp just in case
        fix_df.sort_values("timestamp", inplace=True)
        vel_df.sort_values("timestamp", inplace=True)

        # Merge the two dataframes by finding the nearest 'vel' for each 'fix'
        # (0.05s tolerance, half of the 10Hz period)
        ground_truth_df = pd.merge_asof(
            fix_df, vel_df, on="timestamp", direction="nearest", tolerance=0.05
        )

        # Save the final text file
        output_gt_path = sequence_path / "rtk_ground_truth.txt"
        ground_truth_df.to_csv(
            output_gt_path, sep=" ", header=True, index=False, float_format="%.6f"
        )
        print(f"Ground truth saved to: {output_gt_path}")

    print(f"--- Extraction complete for {bag_path.name} ---")


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    bag_files = list(input_dir.glob("*.bag"))
    if not bag_files:
        print(f"No .bag files found in: {input_dir}")
        return
    print(f"Found {len(bag_files)} .bag files to process.")
    for bag_path in bag_files:
        process_bag_file(bag_path, output_dir)
    print("\n*** All files have been processed! ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts data (including RTK Ground Truth) from .bag files."
    )
    parser.add_argument("input_dir", type=str, help="Folder containing the .bag files.")
    parser.add_argument("output_dir", type=str, help="Destination folder.")
    args = parser.parse_args()
    main(args)
