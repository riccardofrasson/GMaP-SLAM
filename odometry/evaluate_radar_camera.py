# Save this code as Evaluate_Model.py
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation

# --- Import custom modules ---
from dataset_multimodal import UnifiedOdometryDataset, unified_collate_fn
from radar_camera_encoder import OdometryPipeline
from train_radar_camera import (
    pose_7d_to_matrix,
    calculate_ate,
    quaternion_angular_error_numpy,
)


# ==============================================================================
# EVALUATION FUNCTION FOR A SINGLE SEQUENCE
# ==============================================================================
def evaluate_sequence(model, dataloader, device):
    """Runs model inference on an entire dataloader (one sequence)."""
    model.eval()

    # NOTE: Instead of flat lists, we collect batches
    list_of_gt_batches = []
    list_of_pred_batches = []

    pbar = tqdm(dataloader, desc=f"🔬 Evaluating sequence", unit="batch")
    with torch.no_grad():
        for batch in pbar:
            radars = batch["radars"].to(device)
            images = batch["images"].to(device)
            K = batch["K"].to(device)
            T_radar_to_cam = batch["T_radar_to_cam"].to(device)
            radar_lengths = batch["radar_lengths"].to(device)
            gt_poses_batch = batch["poses_gt"].to(device)
            imu_data = batch["imu"].to(device)
            imu_lengths = batch["imu_lengths"].to(device)
            T_imu_from_radar = batch["T_imu_from_radar"].to(device)

            B, clip_len_full, _, _, _ = images.shape
            clip_len_poses = clip_len_full - 1

            history = None
            predicted_poses_list = []

            for t in range(clip_len_poses):
                sub_batch = {
                    "radar_t0": radars[:, t, ...],
                    "radar_t1": radars[:, t + 1, ...],
                    "image_t0": images[:, t, ...],
                    "image_t1": images[:, t + 1, ...],
                    "K": K,
                    "T_radar_to_cam": T_radar_to_cam,
                    "T_imu_from_radar": T_imu_from_radar,
                    "lengths_t0": {"radar": radar_lengths[:, t]},
                    "lengths_t1": {"radar": radar_lengths[:, t + 1]},
                    "imu_sequence": imu_data[:, t, ...],
                    "imu_lengths": imu_lengths[:, t],
                }
                predicted_pose, new_history = model(sub_batch, history)
                history = new_history.detach() if new_history is not None else None
                predicted_poses_list.append(predicted_pose)

            predicted_poses = torch.stack(predicted_poses_list, dim=1)

            # Add the batch tensors to the lists
            list_of_gt_batches.append(gt_poses_batch.cpu().numpy())
            list_of_pred_batches.append(predicted_poses.cpu().numpy())

    # Concatenate all batches and reshape them
    # to get a single 2D array [total_num_poses, 7]
    if not list_of_gt_batches:
        return np.empty((0, 7)), np.empty((0, 7))

    # Concatenate along the batch axis
    all_gt_poses_3d = np.concatenate(list_of_gt_batches, axis=0)
    all_pred_poses_3d = np.concatenate(list_of_pred_batches, axis=0)

    # Flatten the batch and clip dimensions
    num_total_clips, clip_len, pose_dim = all_gt_poses_3d.shape
    flat_gt_poses = all_gt_poses_3d.reshape(num_total_clips * clip_len, pose_dim)
    flat_pred_poses = all_pred_poses_3d.reshape(num_total_clips * clip_len, pose_dim)

    return flat_gt_poses, flat_pred_poses


# ==============================================================================
# FUNCTION FOR METRIC CALCULATION AND PLOTTING
# ==============================================================================
def compute_metrics_and_plot(
    gt_relative_poses, pred_relative_poses, sequence_name, results_dir
):
    """
    Calculates ATE, drift, saves the plot, 3D trajectory, and full 7D poses.
    """

    # 1. Reconstruct absolute trajectories (both 7D and 3D position-only)
    pose_gt_mat, gt_trajectory, gt_poses_7d_abs = (
        np.eye(4),
        [np.zeros(3)],
        [np.array([0, 0, 0, 0, 0, 0, 1])],
    )
    for rel_pose in gt_relative_poses:
        T_rel = pose_7d_to_matrix(rel_pose)
        pose_gt_mat = pose_gt_mat @ T_rel
        gt_trajectory.append(pose_gt_mat[:3, 3])
        # Create and store the absolute 7D pose
        pos = pose_gt_mat[:3, 3]
        quat = Rotation.from_matrix(pose_gt_mat[:3, :3]).as_quat()  # xyzw format
        gt_poses_7d_abs.append(np.concatenate([pos, quat]))

    pose_pred_mat, pred_trajectory, pred_poses_7d_abs = (
        np.eye(4),
        [np.zeros(3)],
        [np.array([0, 0, 0, 0, 0, 0, 1])],
    )
    for rel_pose in pred_relative_poses:
        T_rel = pose_7d_to_matrix(rel_pose)
        pose_pred_mat = pose_pred_mat @ T_rel
        pred_trajectory.append(pose_pred_mat[:3, 3])
        # Create and store the absolute 7D pose
        pos = pose_pred_mat[:3, 3]
        quat = Rotation.from_matrix(pose_pred_mat[:3, :3]).as_quat()  # xyzw format
        pred_poses_7d_abs.append(np.concatenate([pos, quat]))

    gt_trajectory, pred_trajectory = np.array(gt_trajectory), np.array(pred_trajectory)
    gt_poses_7d_abs, pred_poses_7d_abs = np.array(gt_poses_7d_abs), np.array(
        pred_poses_7d_abs
    )

    # Save the 3D (x,y,z) trajectory, useful for plots
    pred_traj_path = results_dir / f"predicted_trajectory_{sequence_name}.npy"
    gt_traj_path = results_dir / f"gt_trajectory_{sequence_name}.npy"
    np.save(pred_traj_path, pred_trajectory)
    np.save(gt_traj_path, gt_trajectory)
    print(f"  -> 3D trajectory (positions) saved to: '{pred_traj_path}'")

    # NEW: Save the full 7D poses (position + quaternion) for fusion
    pred_poses_7d_path = results_dir / f"predicted_poses_7d_{sequence_name}.npy"
    gt_poses_7d_path = results_dir / f"gt_poses_7d_{sequence_name}.npy"
    np.save(pred_poses_7d_path, pred_poses_7d_abs)
    np.save(gt_poses_7d_path, gt_poses_7d_abs)
    print(f"  -> 7D poses (pos+rot) saved to: '{pred_poses_7d_path}'")
    # ---------------------------------------------

    # 2. Metric calculation
    ate = calculate_ate(gt_trajectory, pred_trajectory)
    angular_errors = [
        quaternion_angular_error_numpy(p[3:], g[3:])
        for p, g in zip(pred_relative_poses, gt_relative_poses)
    ]
    total_dist = np.sum(np.linalg.norm(gt_relative_poses[:, :3], axis=1))
    rot_drift_deg_per_meter = (
        np.sqrt(np.mean(np.array(angular_errors) ** 2)) / total_dist
        if total_dist > 0
        else 0.0
    )

    # 3. Plot trajectory
    plt.figure(figsize=(10, 10))
    plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], "b-", label="Ground Truth")
    plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], "r--", label="Predicted")
    plt.title(f"Trajectory - {sequence_name} (ATE: {ate:.4f} m)")
    plt.xlabel("X [m]"), plt.ylabel("Y [m]"), plt.legend(), plt.grid(True), plt.axis(
        "equal"
    )
    plot_path = results_dir / f"trajectory_{sequence_name}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"  -> Plot saved to: '{plot_path}'")

    return ate, rot_drift_deg_per_meter


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- List of all sequences to evaluate ---
    ALL_SEQUENCES = [
        #'URBAN_A0','LOOP_A0', 'LOOP_B0', 'LOOP_C0', 'LOOP_E0', 'URBAN_A1', 'RURAL_A0', 'RURAL_A1', 'RURAL_A2', 'RURAL_B2',
        #'URBAN_C0','URBAN_C1','URBAN_D0','URBAN_D1','URBAN_E0','URBAN_E1', 'URBAN_F0','URBAN_F1', 'URBAN_G0','URBAN_G1','URBAN_H0','URBAN_H1',
        #'RURAL_C0','RURAL_C1','RURAL_C2','RURAL_D0','RURAL_D1','RURAL_D2','RURAL_E0','RURAL_E1','RURAL_E2','RURAL_F0','RURAL_F1','RURAL_F2',
        "LOOP_D0",
        "RURAL_B0",
    ]

    # Create results directory if it doesn't exist
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    # --- Image transforms (must be the same as in training) ---
    image_transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # --- Initialize Model (with same parameters as training) ---
    radar_mean_stats = [0.4825, -0.7621, 15.3256, -4.4988, 11.8215]
    radar_std_stats = [10.2584, 2.2869, 13.9149, 3.7959, 4.4538]
    gat_params = {
        "k_neighbors": 8,
        "motion_radius": 1.0,
        "feature_k": 8,
        "dropout": 0.5,
        "radar_mean": radar_mean_stats,
        "radar_std": radar_std_stats,
    }
    cv_params = {"nsample": 32, "cost_ch": 256}
    optimizer_params = {"layer_num": 5}

    model = OdometryPipeline(
        n_sample_points=1024,
        gat_params=gat_params,
        cv_params=cv_params,
        optimizer_params=optimizer_params,
    ).to(DEVICE)

    # --- Load trained model weights ---
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=DEVICE, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    results = []

    # --- Evaluation loop over each sequence ---
    for seq_name in ALL_SEQUENCES:
        print(f"\n{'='*20} Evaluating sequence: {seq_name} {'='*20}")

        # Create a dataset for the single sequence in 'full_sequence' mode
        dataset = UnifiedOdometryDataset(
            root_dir=args.data_dir,
            mode="full_sequence",
            sequence_names=[seq_name],
            clip_length=args.clip_len,
            stride=4,  # Use stride=4 for evaluation
            image_transform=image_transform,
        )

        if len(dataset) == 0:
            print(
                f"WARNING: No clips found for sequence '{seq_name}'. Skipping sequence."
            )
            continue

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=unified_collate_fn,
            num_workers=4,
        )

        # Run inference and get all poses
        gt_poses, pred_poses = evaluate_sequence(model, dataloader, DEVICE)

        # Calculate metrics and generate plots
        ate, rot_drift = compute_metrics_and_plot(
            gt_poses, pred_poses, seq_name, results_dir
        )

        results.append(
            {"Sequence": seq_name, "ATE (m)": ate, "Rotational Drift (°/m)": rot_drift}
        )

    # --- Print final report ---
    print(f"\n{'='*25} FINAL EVALUATION REPORT {'='*25}")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Calculate and print averages
    if not results_df.empty:
        mean_ate = results_df["ATE (m)"].mean()
        mean_drift = results_df["Rotational Drift (°/m)"].mean()
        print("\n--- Overall Averages ---")
        print(f"Mean ATE: {mean_ate:.4f} m")
        print(f"Mean Rotational Drift: {mean_drift:.4f} °/m")

    # Save the report to a CSV file
    report_path = results_dir / "evaluation_report.csv"
    results_df.to_csv(report_path, index=False)
    print(f"\nReport saved to: '{report_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Radar-Camera Odometry model on all dataset sequences."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/arrubuntu20/SSD_2/Riccardo/Extracted Dataset",
        help="Path to the main dataset directory.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained .pth model file (e.g., 'training_RadarCamera_Simplified/best_model_ate.pth').",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation_results_RadarCamera_ABLATION",
        help="Directory to save trajectory plots and the final report.",
    )
    parser.add_argument(
        "--clip_len", type=int, default=5, help="Clip length used during training."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for inference."
    )

    args = parser.parse_args()
    main(args)
