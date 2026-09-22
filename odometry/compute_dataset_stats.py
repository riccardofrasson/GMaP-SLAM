import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
from dataset_multimodal import UnifiedOdometryDataset, unified_collate_fn


ROOT_DATA_DIR = "/media/arrubuntu20/SSD_2/Riccardo/Extracted Dataset"
# --------------------


# --- SPECIALIZED CLASS WITH DETAILED FILTERS ---
class UnnormalizedDataset(UnifiedOdometryDataset):
    """
    This dataset class inherits from the original but overrides __getitem__
    to apply specific filters for statistics calculation and skip normalization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(
            "\n---> Created Dataset instance with CUSTOM FILTERS for stats calculation.\n"
        )

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
        K, T_radar_to_cam = self.read_calibration_matrices(sequence_dir)
        K, T_radar_to_cam = (
            torch.from_numpy(K).float(),
            torch.from_numpy(T_radar_to_cam).float(),
        )

        if sequence_name not in self.imu_data_cache:
            imu_file = sequence_dir / "imu" / "IMU.txt"
            self.imu_data_cache[sequence_name] = (
                np.loadtxt(imu_file, usecols=range(3, 9)) if imu_file.exists() else None
            )
        imu_full_data = self.imu_data_cache[sequence_name]

        for i, frame_info in enumerate(clip_info):
            # --- LIDAR SECTION WITH FILTER AND NO DOWNSAMPLING ---
            points_l = np.load(frame_info["lidar_path"], allow_pickle=True)
            # Initial cleanup filter for LiDAR
            points_l = points_l[
                np.isfinite(points_l).all(axis=1) & np.any(points_l != 0, axis=1)
            ]

            final_points_l = self._standardize_lidar_points(points_l)
            lidars[i] = torch.from_numpy(final_points_l).float()

            # --- RADAR SECTION WITH MULTIPLE FILTERS ---
            points_r = np.load(frame_info["radar_path"], allow_pickle=True)

            # 1. Initial cleanup filter
            points_r = points_r[
                np.isfinite(points_r).all(axis=1) & np.any(points_r[:, :3] != 0, axis=1)
            ]

            # 2. Spherical distance filter
            if points_r.shape[0] > 0:
                distance_threshold = 400.0
                distances = np.linalg.norm(points_r[:, :3], axis=1)
                points_r = points_r[distances < distance_threshold]

            # 3. Height filter
            if points_r.shape[0] > 0:
                z_coords = points_r[:, 2]
                points_r = points_r[(z_coords > -2.0) & (z_coords < 5.0)]

            # Add filtered radar data to the list
            radars_list.append(torch.from_numpy(points_r).float())

            # --- IMU SECTION ---
            if imu_full_data is not None:
                imu_chunk = imu_full_data[
                    frame_info["imu_start"] : frame_info["imu_end"] + 1
                ]
                imu_list.append(torch.from_numpy(imu_chunk).float())
            else:
                imu_list.append(torch.empty(0, 6))

            if self.image_transform:
                images[i] = self.image_transform(
                    Image.open(frame_info["image_path"]).convert("RGB")
                )

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
            "T_radar_to_cam": T_radar_to_cam,
            "sequence_name": sequence_name,
        }

    @staticmethod
    def read_calibration_matrices(sequence_dir):
        """Helper to read calibration files, matching the original dataset's logic."""
        from pathlib import Path
        import numpy as np

        K, T_radar_to_cam = np.eye(3), np.eye(4)
        calib_cam_path = Path(sequence_dir) / "CALIBRATION_CAMERA.txt"
        calib_radar_path = Path(sequence_dir) / "CALIBRATION_CAMERA_RADAR.txt"
        try:
            with open(calib_cam_path, "r") as f:
                lines = f.readlines()
            K = np.array(
                [[float(num) for num in lines[i].strip().split()] for i in range(5, 8)]
            )
            with open(calib_radar_path, "r") as f:
                lines = f.readlines()
            R = np.array(
                [[float(num) for num in lines[i].strip().split()] for i in range(4, 7)]
            )
            t = np.array([float(num) for num in lines[10].strip().split()]).reshape(
                3, 1
            )
            T_radar_to_cam[:3, :3], T_radar_to_cam[:3, 3] = R, t.flatten()
        except Exception:
            # Silently pass if files are missing, will use identity matrices
            pass
        return K, T_radar_to_cam


# --- CALCULATION FUNCTION ---
def calculate_dataset_stats():
    print("Initializing training dataset for statistics calculation...")

    train_dataset = UnnormalizedDataset(
        root_dir=ROOT_DATA_DIR,
        mode="train",
        clip_length=5,
        stride=4,
        train_val_split_ratio=0.85,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty. Check path and configuration.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=unified_collate_fn,
        num_workers=4,
    )

    print(f"Calculating stats over {len(train_dataset)} clips...")

    radar_sum, radar_sum_sq, radar_count = torch.zeros(5), torch.zeros(5), 0
    lidar_sum, lidar_sum_sq, lidar_count = torch.zeros(3), torch.zeros(3), 0
    imu_sum, imu_sum_sq, imu_count = torch.zeros(6), torch.zeros(6), 0

    for batch in tqdm(train_loader, desc="Calculating Stats", unit="batch"):
        radars_padded, radar_lengths = batch["radars"], batch["radar_lengths"]
        lidars = batch["lidars"]
        imus_padded, imu_lengths = batch["imu"], batch["imu_lengths"]
        B, C, _, _ = lidars.shape

        for i in range(B):
            for j in range(C):
                # Radar stats
                radar_len = radar_lengths[i, j].item()
                if radar_len > 0:
                    real_radar_points = radars_padded[i, j, :radar_len, :]
                    radar_sum += torch.sum(real_radar_points, dim=0)
                    radar_sum_sq += torch.sum(real_radar_points.pow(2), dim=0)
                    radar_count += radar_len

                # Lidar stats
                real_lidar_points = lidars[i, j][(lidars[i, j] != 0).any(dim=-1)]
                if real_lidar_points.shape[0] > 0:
                    lidar_sum += torch.sum(real_lidar_points, dim=0)
                    lidar_sum_sq += torch.sum(real_lidar_points.pow(2), dim=0)
                    lidar_count += real_lidar_points.shape[0]

                # IMU stats
                imu_len = imu_lengths[i, j].item()
                if imu_len > 0:
                    real_imu_data = imus_padded[i, j, :imu_len, :]
                    imu_sum += torch.sum(real_imu_data, dim=0)
                    imu_sum_sq += torch.sum(real_imu_data.pow(2), dim=0)
                    imu_count += imu_len

    epsilon = 1e-8
    radar_mean = radar_sum / (radar_count + epsilon)
    radar_std = torch.sqrt(radar_sum_sq / (radar_count + epsilon) - radar_mean.pow(2))
    lidar_mean = lidar_sum / (lidar_count + epsilon)
    lidar_std = torch.sqrt(lidar_sum_sq / (lidar_count + epsilon) - lidar_mean.pow(2))
    imu_mean = imu_sum / (imu_count + epsilon)
    imu_std = torch.sqrt(imu_sum_sq / (imu_count + epsilon) - imu_mean.pow(2))

    print("\n--- ✅ CALCULATION COMPLETE ---")
    print(
        f"Total Radar Points: {radar_count} | Total LiDAR Points: {lidar_count} | Total IMU Readings: {imu_count}"
    )

    print(
        "\nCopy and paste these values into the __init__ method of UnifiedOdometryDataset:"
    )
    print("-" * 70)
    print("# Normalization statistics")
    print(f"self.radar_mean = torch.tensor({radar_mean.tolist()})")
    print(f"self.radar_std = torch.tensor({radar_std.tolist()})")
    print(f"self.lidar_mean = torch.tensor({lidar_mean.tolist()})")
    print(f"self.lidar_std = torch.tensor({lidar_std.tolist()})")
    print(f"self.imu_mean = torch.tensor({imu_mean.tolist()})")
    print(f"self.imu_std = torch.tensor({imu_std.tolist()})")
    print(f"self.epsilon = 1e-8")
    print("-" * 70)


if __name__ == "__main__":
    calculate_dataset_stats()
