import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import random

# ==============================================================================
# --- UPDATED DATASET CLASS ---
# ==============================================================================


class IMUOdometryDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode="train",
        clip_length=30,
        stride=15,
        train_val_split_ratio=0.85,
        seed=42,
        sequence_names=None,
    ):

        self.root_dir = Path(root_dir)
        self.mode = mode
        self.clip_length = clip_length
        self.imu_data_cache = {}
        self.sequence_names = sequence_names

        # Normalization values
        self.imu_mean = torch.tensor(
            [
                -0.2755765914916992,
                0.21491482853889465,
                9.793403625488281,
                0.0010617460357025266,
                0.0013723340816795826,
                0.0045267497189342976,
            ]
        )
        self.imu_std = torch.tensor(
            [
                0.7938064336776733,
                0.8223322629928589,
                0.7926174998283386,
                0.02224370464682579,
                0.04197201132774353,
                0.06889662146568298,
            ]
        )
        self.epsilon = 1e-8

        print(f"Initializing LIGHTWEIGHT (IMU-only) dataset in '{mode}' mode...")

        all_sequence_dirs = {d.name: d for d in self.root_dir.iterdir() if d.is_dir()}
        self.clips = []

        if self.sequence_names is not None:
            print(f" - Loading specific sequences: {self.sequence_names}")
            target_dirs = [
                all_sequence_dirs[name]
                for name in self.sequence_names
                if name in all_sequence_dirs
            ]
        else:
            # Default train/val/test splits if no specific sequences are provided
            train_val_sequence_names = [
                "URBAN_A0",
                "LOOP_A0",
                "LOOP_B0",
                "LOOP_C0",
                "LOOP_E0",
                "URBAN_A1",
                "RURAL_A0",
                "RURAL_A1",
                "RURAL_A2",
                "RURAL_B2",
                "RURAL_B1",
                "URBAN_C0",
                "URBAN_C1",
                "URBAN_D0",
                "URBAN_D1",
                "URBAN_E0",
                "URBAN_E1",
                "URBAN_F0",
                "URBAN_F1",
                "URBAN_G0",
                "URBAN_G1",
                "URBAN_H0",
                "URBAN_H1",
                "RURAL_C0",
                "RURAL_C1",
                "RURAL_C2",
                "RURAL_D0",
                "RURAL_D1",
                "RURAL_D2",
                "RURAL_E0",
                "RURAL_E1",
                "RURAL_E2",
                "RURAL_F0",
                "RURAL_F1",
                "RURAL_F2",
            ]
            test_sequence_names = ["LOOP_D0", "RURAL_B0"]

            if mode in ["train", "val"]:
                target_dirs = [
                    all_sequence_dirs[name]
                    for name in train_val_sequence_names
                    if name in all_sequence_dirs
                ]
            else:
                target_dirs = [
                    all_sequence_dirs[name]
                    for name in test_sequence_names
                    if name in all_sequence_dirs
                ]

        for seq_dir in target_dirs:
            indices_path = seq_dir / "synchronized_indices.csv"
            gt_path = seq_dir / "relative_poses.csv"
            if not indices_path.exists() or not gt_path.exists():
                continue

            indices_df = pd.read_csv(indices_path)
            gt_df = pd.read_csv(gt_path)

            sequence_clips = []
            for i in range(0, len(indices_df) - self.clip_length + 1, stride):
                clip_indices = indices_df.iloc[i : i + self.clip_length]
                # Ensure lidar indices are consecutive (proxy for consecutive frames)
                if not np.all(np.diff(clip_indices["lidar_index"].values) == 1):
                    continue

                current_clip_info = [
                    {
                        "imu_start": r["imu_start_index"],
                        "imu_end": r["imu_end_index"],
                        "pose": gt_df.iloc[r["lidar_index"]].to_dict(),
                    }
                    for _, r in clip_indices.iterrows()
                ]

                sequence_clips.append((current_clip_info, seq_dir.name))

            if not sequence_clips:
                continue

            if mode in ["train", "val"] and self.sequence_names is None:
                # Split sequence into train/val if not loading specific sequences
                n_train = int(len(sequence_clips) * train_val_split_ratio)
                if self.mode == "train":
                    self.clips.extend(sequence_clips[:n_train])
                elif self.mode == "val":
                    self.clips.extend(sequence_clips[n_train:])
            else:
                # Add all clips (for 'test' mode or if specific sequences were requested)
                self.clips.extend(sequence_clips)

        if not self.clips:
            raise RuntimeError(f"No valid clips found for mode '{self.mode}'.")

        # Oversampling logic
        if self.mode == "train":
            print("🚀 Starting AGGRESSIVE AND TARGETED oversampling...")

            BINS_THRESHOLDS = {
                "straight": 2.0,  # Straights (< 2°/s)
                "light_turn": 10.0,  # Light turns (< 10°/s)
                "sharp_turn": float("inf"),  # Sharp turns (> 10°/s)
            }

            # === Increase aggression on sharp turns ===
            OVERSAMPLING_FACTORS = {
                "straight": 1,  # Straights x1
                "light_turn": 4,  # Light turns x4
                "sharp_turn": 8,  # Sharp turns x8
            }
            dt = 0.1  # 10 Hz

            binned_clips = {key: [] for key in BINS_THRESHOLDS}

            # 1. Analyze and Assign to Bins
            for clip_data in self.clips:
                clip_info_list, _ = clip_data
                max_rate_in_clip = 0
                for frame_info in clip_info_list:
                    q_w = frame_info["pose"]["qw"]
                    angle_rad = 2 * np.arccos(np.clip(np.abs(q_w), -1.0, 1.0))
                    rate_deg_s = (angle_rad / dt) * (180.0 / np.pi)
                    if rate_deg_s > max_rate_in_clip:
                        max_rate_in_clip = rate_deg_s
                for bin_name, threshold in BINS_THRESHOLDS.items():
                    if max_rate_in_clip < threshold:
                        binned_clips[bin_name].append(clip_data)
                        break

            # 2. Build the New Clip Set for the Epoch
            new_clips = []
            print("\n--- Analysis and Epoch Dataset Construction ---")
            for bin_name, clips_in_bin in binned_clips.items():
                factor = OVERSAMPLING_FACTORS[bin_name]
                print(
                    f" - Category '{bin_name}': {len(clips_in_bin)} unique clips. Will be included {factor} times."
                )
                new_clips.extend(clips_in_bin * factor)

            self.clips = new_clips
            random.shuffle(self.clips)  # Shuffle after construction

            print(f"✅ Process complete. Total clips for this epoch: {len(self.clips)}")

        if self.mode == "train":
            random.seed(seed)
            random.shuffle(self.clips)

        print(
            f"Dataset '{self.mode}' initialized. Total clips for the epoch: {len(self.clips)}."
        )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info, sequence_name = self.clips[idx]

        imu_list = []
        poses_gt_list = []

        if sequence_name not in self.imu_data_cache:
            imu_file = self.root_dir / sequence_name / "imu" / "IMU.txt"
            # Load as float32 for PyTorch consistency
            self.imu_data_cache[sequence_name] = (
                np.loadtxt(imu_file, usecols=range(3, 9), dtype=np.float32)
                if imu_file.exists()
                else None
            )
        imu_full_data = self.imu_data_cache[sequence_name]

        # Iterate up to clip_length - 1 to get N-1 relative poses
        for i in range(self.clip_length - 1):
            frame_info = clip_info[i]
            if imu_full_data is not None:
                imu_start, imu_end = int(frame_info["imu_start"]), int(
                    frame_info["imu_end"]
                )

                # 1. Extract the raw data chunk
                raw_imu_chunk = torch.from_numpy(imu_full_data[imu_start : imu_end + 1])

                # 2. Normalize the chunk
                normalized_imu = (raw_imu_chunk - self.imu_mean) / (
                    self.imu_std + self.epsilon
                )

                imu_list.append(normalized_imu)
            else:
                imu_list.append(torch.empty(0, 6, dtype=torch.float32))

            # Load ground truth pose
            pose_dict = frame_info["pose"]
            pose_tensor = torch.tensor(
                [
                    pose_dict["dx"],
                    pose_dict["dy"],
                    pose_dict["dz"],
                    pose_dict["qx"],
                    pose_dict["qy"],
                    pose_dict["qz"],
                    pose_dict["qw"],
                ],
                dtype=torch.float32,
            )
            poses_gt_list.append(pose_tensor)

        return {
            "imu": imu_list,
            "gt_poses": torch.stack(poses_gt_list),
            "sequence_name": sequence_name,
        }


# ==============================================================================
# --- COLLATE FUNCTION ---
# ==============================================================================
def imu_collate_fn(batch):
    collated_batch = {}
    if not batch:
        return collated_batch

    keys = batch[0].keys()
    batch_list = {key: [d[key] for d in batch] for key in keys}

    for key in keys:
        if key not in ["imu", "sequence_name"]:
            collated_batch[key] = torch.stack(batch_list[key])

    if "sequence_name" in keys:
        collated_batch["sequence_name"] = batch_list["sequence_name"]

    if "imu" in keys:
        imu_clips = batch_list["imu"]  # List (B) of lists (N-1) of tensors (S, 6)
        # Flatten the list of all intervals
        imu_intervals_flat = [interval for clip in imu_clips for interval in clip]
        imu_lengths = torch.tensor(
            [len(interval) for interval in imu_intervals_flat], dtype=torch.long
        )

        if any(len(f) > 0 for f in imu_intervals_flat):
            # Pad the flat list of intervals
            padded_imu = torch.nn.utils.rnn.pad_sequence(
                [f for f in imu_intervals_flat if len(f) > 0],
                batch_first=True,
                padding_value=0.0,
            )
            B, N_minus_1 = len(batch), len(imu_clips[0])
            T_max, D_imu = padded_imu.shape[1], padded_imu.shape[2]

            # Reshape back into [B, N-1, T_max, D_imu]
            collated_batch["imu"] = padded_imu.view(B, N_minus_1, T_max, D_imu)
            collated_batch["imu_lengths"] = imu_lengths.view(B, N_minus_1)
        else:
            # Handle empty batch
            B, N_minus_1 = len(batch), len(batch_list.get("gt_poses", [[]])[0])
            collated_batch["imu"] = torch.zeros(B, N_minus_1, 0, 6)
            collated_batch["imu_lengths"] = torch.zeros(B, N_minus_1, dtype=torch.long)

    return collated_batch
