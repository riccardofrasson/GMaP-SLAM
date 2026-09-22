import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import open3d as o3d
import random


## --- UNIVERSAL PARSING FUNCTION --- ##
def load_transform_matrix(filepath):
    """
    Robust universal parsing function.
    Automatically determines the file format (Kalibr, R/T, or Unlabeled) and parses it.
    """
    try:
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        # Check for Kalibr format (anchor string 'T_ic:')
        anchor_string = "T_ic:  (cam0 to imu0):"
        anchor_index = -1
        for i, line in enumerate(lines):
            if anchor_string in line:
                anchor_index = i
                break

        if anchor_index != -1:
            matrix_rows = []
            for i in range(anchor_index + 1, anchor_index + 5):
                cleaned_line = lines[i].replace("[", "").replace("]", "").strip()
                row = [float(num) for num in cleaned_line.split()]
                matrix_rows.append(row)
            return np.array(matrix_rows)

        # Check for labeled R/T format
        r_index, t_index = -1, -1
        for i, line in enumerate(lines):
            if line.strip().startswith("R:"):
                r_index = i
            if line.strip().startswith("T:"):
                t_index = i

        if r_index != -1 and t_index != -1:
            R = np.array(
                [
                    [float(num) for num in lines[i].split()]
                    for i in range(r_index + 1, r_index + 4)
                ]
            )
            t = np.array([float(num) for num in lines[t_index + 1].split()]).reshape(
                3, 1
            )
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            return T

        # Check for Unlabeled format (fixed line numbers)
        try:
            R = np.array(
                [[float(num) for num in lines[i].strip().split()] for i in range(4, 7)]
            )
            t = np.array([float(num) for num in lines[10].strip().split()]).reshape(
                3, 1
            )
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            return T
        except (ValueError, IndexError):
            pass

        raise ValueError(f"Unrecognized calibration file format: {filepath.name}")

    except Exception as e:
        print(f"Critical error while reading file {filepath}: {e}")
        return None


## --- MAIN UNIFIED DATASET CLASS --- ##
class UnifiedOdometryDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode="train",
        image_transform=None,
        clip_length=5,
        lidar_num_points=16384,
        stride=4,
        voxel_size=0.2,
        train_val_split_ratio=0.85,
        seed=42,
        sequence_names=None,
    ):

        self.root_dir = Path(root_dir)
        self.mode = mode
        self.image_transform = image_transform
        self.clip_length = clip_length
        self.lidar_num_points = lidar_num_points
        self.voxel_size = voxel_size
        self.imu_data_cache = {}

        # Normalization statistics
        self.radar_mean = torch.tensor(
            [
                0.846843957901001,
                -0.18832461535930634,
                1.3715522289276123,
                -2.200822353363037,
                10.865551948547363,
            ]
        )
        self.radar_std = torch.tensor(
            [
                6.374578475952148,
                1.2997065782546997,
                1.7640600204467773,
                2.7319979667663574,
                4.506900787353516,
            ]
        )
        self.lidar_mean = torch.tensor(
            [-0.042016200721263885, 0.6049638986587524, -0.10459686070680618]
        )
        self.lidar_std = torch.tensor(
            [14.01001262664795, 13.448746681213379, 2.7760772705078125]
        )
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

        print(f"Initializing unified dataset '{mode}'...")

        all_sequence_dirs = {d.name: d for d in self.root_dir.iterdir() if d.is_dir()}
        self.clips = []

        # Select target sequences
        if mode == "full_sequence":
            if sequence_names is None:
                raise ValueError(
                    "For 'full_sequence' mode, `sequence_names` must be specified."
                )
            target_dirs = [
                all_sequence_dirs[name]
                for name in sequence_names
                if name in all_sequence_dirs
            ]
            print(f"  - Loading {len(target_dirs)} full sequences: {sequence_names}")
        else:
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
                "RURAL_B1",
                "RURAL_B2",
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

        # Build the list of all valid clips
        for seq_dir in target_dirs:
            indices_path, gt_path = (
                seq_dir / "synchronized_indices.csv",
                seq_dir / "relative_poses.csv",
            )
            if not indices_path.exists() or not gt_path.exists():
                continue

            indices_df = pd.read_csv(indices_path)
            gt_df = pd.read_csv(gt_path)
            sequence_clips = []

            max_allowable_lidar_index = len(gt_df)

            for i in range(0, len(indices_df) - self.clip_length + 1, stride):
                clip_indices = indices_df.iloc[i : i + self.clip_length]

                if clip_indices["lidar_index"].max() >= max_allowable_lidar_index:
                    continue

                if not np.all(np.diff(clip_indices["lidar_index"].values) == 1):
                    continue

                current_clip_info = []
                # Handle two possible synchronization CSV formats
                if "radar_index" in clip_indices.columns:
                    current_clip_info = [
                        {
                            "lidar_path": str(
                                seq_dir / "lidar" / f"{r['lidar_index']:06d}.npy"
                            ),
                            "radar_path": str(
                                seq_dir / "radar" / f"{r['radar_index']:06d}.npy"
                            ),
                            "image_path": str(
                                seq_dir / "image_left" / f"{r['camera_index']:06d}.png"
                            ),
                            "imu_start": r["imu_start_index"],
                            "imu_end": r["imu_end_index"],
                            "pose": gt_df.iloc[r["lidar_index"]].to_dict(),
                        }
                        for _, r in clip_indices.iterrows()
                    ]
                else:
                    current_clip_info = [
                        {
                            "lidar_path": str(
                                seq_dir / "lidar" / f"{r['lidar_index']:06d}.npy"
                            ),
                            "radar_path": str(
                                seq_dir / "radar" / f"{r['camera_radar_index']:06d}.npy"
                            ),
                            "image_path": str(
                                seq_dir
                                / "image_left"
                                / f"{r['camera_radar_index']:06d}.png"
                            ),
                            "imu_start": r["imu_start_index"],
                            "imu_end": r["imu_end_index"],
                            "pose": gt_df.iloc[r["lidar_index"]].to_dict(),
                        }
                        for _, r in clip_indices.iterrows()
                    ]

                # Add the clip only if all files exist
                if all(
                    Path(f["lidar_path"]).exists()
                    and Path(f["image_path"]).exists()
                    and Path(f["radar_path"]).exists()
                    for f in current_clip_info
                ):
                    sequence_clips.append((current_clip_info, seq_dir.name))

            if not sequence_clips:
                continue

            # Split into train/val
            if mode in ["train", "val"] and mode != "full_sequence":
                n_train = int(len(sequence_clips) * train_val_split_ratio)
                if self.mode == "train":
                    self.clips.extend(sequence_clips[:n_train])
                elif self.mode == "val":
                    self.clips.extend(sequence_clips[n_train:])
            else:
                self.clips.extend(sequence_clips)

        if not self.clips:
            raise RuntimeError(f"No valid clips found for mode '{self.mode}'.")

        # ==============================================================================
        # --- CUSTOM OVERSAMPLING SECTION ---
        # ==============================================================================
        if self.mode == "train":
            print("🚀 Starting custom oversampling: straights 1x, curves Nx...")

            # --- 1. CONFIGURATION ---

            # Define bin thresholds (in degrees per second)
            BINS_THRESHOLDS = {
                "straight": 2.0,
                "light_turn": 10.0,
                "sharp_turn": float("inf"),
            }

            OVERSAMPLING_FACTOR_CURVES = 1

            dt = 0.1  # Time interval (10 Hz)
            # ----------------------------------------------------------------

            # Structure to hold binned clips by category
            binned_clips = {key: [] for key in BINS_THRESHOLDS}

            # --- 2. Analysis and Bin Assignment ---
            for clip_data in self.clips:
                clip_info_list = clip_data[0]
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

            print("\n--- Analysis of Unique Clips Found ---")
            straight_clips = binned_clips["straight"]
            light_turn_clips = binned_clips["light_turn"]
            sharp_turn_clips = binned_clips["sharp_turn"]

            # Combine both turn categories into a single list
            all_curve_clips = light_turn_clips + sharp_turn_clips

            print(f"Unique straights: {len(straight_clips)}")
            print(f"Unique turns (light + sharp): {len(all_curve_clips)}")
            print("----------------------------------------\n")

            # --- 3. Build New Clip Set for the Epoch ---
            print(f"⚖️ Building dataset for the epoch:")
            print(f" - Including each straight clip 1 time.")
            print(f" - Including each turn clip {OVERSAMPLING_FACTOR_CURVES} times.")

            # The new clip list is the concatenation of:
            # 1. All straight clips, taken once.
            # 2. All turn clips, multiplied by the oversampling factor.
            new_clips = straight_clips + (all_curve_clips * OVERSAMPLING_FACTOR_CURVES)

            # Replace the old clip list with the new list for this epoch
            self.clips = new_clips

            print(f"✅ Process complete.")

        if self.mode == "train":
            # Shuffling is crucial to mix the straights and duplicated curves
            random.seed(seed)
            random.shuffle(self.clips)

        print(
            f"Dataset '{self.mode}' initialized. Total clips for the epoch: {len(self.clips)}."
        )

    def __len__(self):
        return len(self.clips)

    def _voxel_downsample_lidar(self, points):
        if len(points) == 0:
            return points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        return np.asarray(pcd.voxel_down_sample(voxel_size=self.voxel_size).points)

    def _standardize_lidar_points(self, points):
        if len(points) == 0:
            return np.zeros((self.lidar_num_points, 3), dtype=np.float32)
        if len(points) > self.lidar_num_points:
            indices = np.random.choice(
                len(points), self.lidar_num_points, replace=False
            )
        else:
            indices = np.pad(
                np.arange(len(points)),
                (0, self.lidar_num_points - len(points)),
                mode="wrap",
            )
        return points[indices]

    def __getitem__(self, idx):
        clip_info, sequence_name = self.clips[idx]
        lidars = torch.zeros(self.clip_length, self.lidar_num_points, 3)
        radars_list, imu_list, images = (
            [],
            [],
            torch.zeros(self.clip_length, 3, 384, 384),
        )
        poses_gt = torch.zeros(self.clip_length - 1, 7)

        sequence_dir = self.root_dir / sequence_name

        K_path = sequence_dir / "CALIBRATION_CAMERA.txt"
        T_imu_from_cam_path = sequence_dir / "CALIBRATION_CAMERA_IMU.txt"
        T_cam_from_radar_path = sequence_dir / "CALIBRATION_CAMERA_RADAR.txt"

        with open(K_path, "r") as f:
            lines = f.readlines()
        K = np.array(
            [[float(num) for num in lines[i].strip().split()] for i in range(5, 8)]
        )

        T_imu_from_camera = load_transform_matrix(T_imu_from_cam_path)
        T_camera_from_radar = load_transform_matrix(T_cam_from_radar_path)

        if T_imu_from_camera is None or T_camera_from_radar is None:
            raise RuntimeError(
                f"Could not load calibration files for sequence {sequence_name}"
            )

        T_imu_from_radar = T_imu_from_camera @ T_camera_from_radar

        K = torch.from_numpy(K).float()
        T_cam_from_radar_tensor = torch.from_numpy(T_camera_from_radar).float()
        T_imu_from_radar_tensor = torch.from_numpy(T_imu_from_radar).float()

        if sequence_name not in self.imu_data_cache:
            imu_file = sequence_dir / "imu" / "IMU.txt"
            self.imu_data_cache[sequence_name] = (
                np.loadtxt(imu_file, usecols=range(3, 9)) if imu_file.exists() else None
            )
        imu_full_data = self.imu_data_cache[sequence_name]

        for i, frame_info in enumerate(clip_info):
            # Load Lidar
            points_l = np.load(frame_info["lidar_path"], allow_pickle=True)
            points_l = points_l[
                np.isfinite(points_l).all(axis=1) & np.any(points_l != 0, axis=1)
            ]
            downsampled_points_l = self._voxel_downsample_lidar(points_l)
            final_points_l = self._standardize_lidar_points(downsampled_points_l)
            lidars[i] = (torch.from_numpy(final_points_l).float() - self.lidar_mean) / (
                self.lidar_std + self.epsilon
            )

            # Load Radar
            points_r = np.load(frame_info["radar_path"], allow_pickle=True)

            points_r = points_r[
                np.isfinite(points_r).all(axis=1) & np.any(points_r[:, :3] != 0, axis=1)
            ]

            if points_r.shape[0] > 0:
                distance_threshold = 400.0
                distances = np.linalg.norm(points_r[:, :3], axis=1)
                points_r = points_r[distances < distance_threshold]

            if points_r.shape[0] > 0:
                z_coords = points_r[:, 1]
                points_r = points_r[(z_coords > -50.0) & (z_coords < 2.0)]
                y_coords = points_r[:, 2]
                points_r = points_r[(y_coords > 0)]

            if points_r.shape[0] > 0:
                points_r_tensor = torch.from_numpy(points_r).float()
                normalized_points_r = (points_r_tensor - self.radar_mean) / (
                    self.radar_std + self.epsilon
                )
                radars_list.append(normalized_points_r)
            else:
                radars_list.append(torch.empty(0, 5))

            # Load IMU
            if imu_full_data is not None:
                imu_start, imu_end = int(frame_info["imu_start"]), int(
                    frame_info["imu_end"]
                )
                imu_chunk = imu_full_data[imu_start : imu_end + 1]
                imu_list.append(
                    (torch.from_numpy(imu_chunk).float() - self.imu_mean)
                    / (self.imu_std + self.epsilon)
                )
            else:
                imu_list.append(torch.empty(0, 6))

            # Load Images (uses pre-calculated path)
            if self.image_transform:
                images[i] = self.image_transform(
                    Image.open(frame_info["image_path"]).convert("RGB")
                )

            # Load GT Poses (uses pre-loaded dictionary)
            if i < self.clip_length - 1:
                pose_dict = frame_info["pose"]
                poses_gt[i] = torch.tensor(
                    [
                        pose_dict["dx"],
                        pose_dict["dy"],
                        pose_dict["dz"],
                        pose_dict["qx"],
                        pose_dict["qy"],
                        pose_dict["qz"],
                        pose_dict["qw"],
                    ]
                )

        return {
            "lidars": lidars,
            "radars": radars_list,
            "images": images,
            "imu": imu_list,
            "poses_gt": poses_gt,
            "K": K,
            "T_radar_to_cam": T_cam_from_radar_tensor,
            "T_imu_from_radar": T_imu_from_radar_tensor,
            "sequence_name": sequence_name,
        }


def unified_collate_fn(batch):
    collated_batch = {}
    if not batch:
        return collated_batch
    keys = batch[0].keys()
    batch_list = {key: [d[key] for d in batch] for key in keys}

    for key in keys:
        if key not in ["radars", "imu", "sequence_name"]:
            collated_batch[key] = torch.stack(batch_list[key])

    if "sequence_name" in keys:
        collated_batch["sequence_name"] = batch_list["sequence_name"]

    if "radars" in keys:
        radar_flat = [frame for clip in batch_list["radars"] for frame in clip]
        collated_batch["radar_lengths"] = torch.tensor(
            [[f.shape[0] for f in clip] for clip in batch_list["radars"]],
            dtype=torch.long,
        )
        if any(len(f) > 0 for f in radar_flat):
            padded_radars = torch.nn.utils.rnn.pad_sequence(
                [f for f in radar_flat if len(f) > 0], batch_first=True
            )
            B, C_len, max_len, feat_dim = (
                len(batch),
                len(batch_list["radars"][0]),
                padded_radars.shape[1],
                padded_radars.shape[2],
            )
            collated_batch["radars"] = padded_radars.view(B, C_len, max_len, feat_dim)
        else:
            collated_batch["radars"] = torch.zeros(
                len(batch), len(batch_list["radars"][0]), 0, 5
            )

    if "imu" in keys:
        imu_flat = [frame for clip in batch_list["imu"] for frame in clip]
        collated_batch["imu_lengths"] = torch.tensor(
            [[f.shape[0] for f in clip] for clip in batch_list["imu"]], dtype=torch.long
        )
        if any(len(f) > 0 for f in imu_flat):
            padded_imu = torch.nn.utils.rnn.pad_sequence(
                [f for f in imu_flat if len(f) > 0], batch_first=True
            )
            B, C_len, max_len, feat_dim = (
                len(batch),
                len(batch_list["imu"][0]),
                padded_imu.shape[1],
                padded_imu.shape[2],
            )
            collated_batch["imu"] = padded_imu.view(B, C_len, max_len, feat_dim)
        else:
            collated_batch["imu"] = torch.zeros(
                len(batch), len(batch_list["imu"][0]), 0, 6
            )

    return collated_batch
