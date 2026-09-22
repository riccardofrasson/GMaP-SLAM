import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import pandas as pd
from scipy.spatial.transform import Rotation
from dataset_imu import IMUOdometryDataset, imu_collate_fn

try:
    from mamba_ssm import Mamba
    from timm.models.layers import DropPath
except ImportError:
    print("WARNING: Mamba dependencies are not installed.")
    print("Run: pip install mamba-ssm timm")
    exit()

# ==============================================================================
# 2. MAMBA MODEL DEFINITION
# ==============================================================================


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        x_norm = self.norm(x)
        return self.mamba(x_norm)


class FFN(nn.Module):
    """Feed-Forward Network (MLP) block."""

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block_mamba(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MambaLayer(dim)
        self.mlp = FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # Bi-directional implementation for full context
        forward_pass = self.attn(x)
        backward_pass = self.attn(x.flip(1)).flip(1)

        x = x + self.drop_path(forward_pass) + self.drop_path(backward_pass)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class IMUEncoderMamba(nn.Module):
    def __init__(self, input_features, embed_dim, num_layers=2):
        super().__init__()
        self.proj_in = nn.Linear(input_features, embed_dim)
        self.mamba_blocks = nn.ModuleList(
            [Block_mamba(dim=embed_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.mamba_blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, -1, :]  # Extract the output of the last token


class OdometryMamba(nn.Module):
    def __init__(
        self, input_features=6, embed_dim=256, num_layers_temporal=4, dropout_prob=0.3
    ):
        super().__init__()
        self.encoder = IMUEncoderMamba(
            input_features=input_features, embed_dim=embed_dim, num_layers=2
        )
        dpr = [x.item() for x in torch.linspace(0, 0.4, num_layers_temporal)]
        self.temporal_mamba = nn.ModuleList(
            [
                Block_mamba(dim=embed_dim, drop_path=dpr[i])
                for i in range(num_layers_temporal)
            ]
        )
        # USE THE TWO REGRESSION HEADS FROM THE TRAINING SCRIPT
        self.regressor_q = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=embed_dim // 2, out_features=4),
        )
        self.regressor_t = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=embed_dim // 2, out_features=3),
        )

    def forward(self, imu_clips):
        if imu_clips.shape[-1] != 6:
            imu_clips = imu_clips[..., :6]
        b, c, s, f = imu_clips.shape
        imu_reshaped = imu_clips.view(b * c, s, f)
        features = self.encoder(imu_reshaped)
        features_reshaped = features.view(b, c, -1)

        temporal_out = features_reshaped
        for layer in self.temporal_mamba:
            temporal_out = layer(temporal_out)

        # RETURN THE (q, t) TUPLE AS IN TRAINING
        pred_q = self.regressor_q(temporal_out)
        pred_t = self.regressor_t(temporal_out)
        pred_q_norm = F.normalize(pred_q, p=2, dim=-1)

        return pred_q_norm, pred_t


# ==============================================================================
# 3. HELPER FUNCTIONS AND METRICS
# ==============================================================================
def pose_7d_to_matrix(pose_vec):
    """Converts a 7D pose vector (t, q) into a 4x4 transformation matrix."""
    t, q_xyzw = pose_vec[:3], pose_vec[3:]
    T = np.eye(4)
    if np.linalg.norm(q_xyzw) > 1e-8:
        # Note: scipy expects the quaternion in (x, y, z, w) format
        T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t
    return T


def calculate_ate_aligned(gt_trajectory, pred_trajectory):
    """Calculates the Absolute Trajectory Error (ATE) with alignment."""
    gt_centroid = np.mean(gt_trajectory, axis=0)
    pred_centroid = np.mean(pred_trajectory, axis=0)
    gt_centered = gt_trajectory - gt_centroid
    pred_centered = pred_trajectory - pred_centroid
    H = pred_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # Align predicted trajectory to GT centroid (not origin)
    pred_aligned = (R @ pred_centered.T).T + gt_centroid
    errors = gt_trajectory - pred_aligned
    return np.sqrt(np.mean(np.sum(errors**2, axis=1)))


def quaternion_angular_error_numpy(q1, q2):
    """Calculates the angular error between two numpy quaternions."""
    dot = np.clip(np.abs(np.sum(q1 * q2)), -1.0, 1.0)
    return 2 * np.arccos(dot) * (180.0 / np.pi)


# ==============================================================================
# 4. EVALUATION FUNCTION FOR A SINGLE SEQUENCE
# ==============================================================================
def evaluate_sequence(model, dataloader, device):
    """Runs model inference on an entire dataloader (one sequence)."""
    model.eval()

    list_of_gt_batches = []
    list_of_pred_batches = []

    desc = f"🔬 Evaluating sequence '{dataloader.dataset.sequence_names[0]}'"
    pbar = tqdm(dataloader, desc=desc, unit="batch", leave=False)

    with torch.no_grad():
        for batch in pbar:
            imu_sequences = batch["imu"].to(device, non_blocking=True)
            gt_poses_batch = batch["gt_poses"]  # Kept on CPU

            predicted_q_sequence, predicted_t_sequence = model(imu_sequences)

            # Ensure clip lengths are consistent
            effective_clip_len = min(
                predicted_q_sequence.shape[1], gt_poses_batch.shape[1]
            )

            pred_q_sliced_np = (
                predicted_q_sequence[:, :effective_clip_len, :].cpu().numpy()
            )

            pred_t_sliced_np = (
                predicted_t_sequence[:, :effective_clip_len, :].cpu().numpy()
            )

            gt_poses_sliced_np = gt_poses_batch[:, :effective_clip_len, :].numpy()

            # Combine the PREDICTED translation with the PREDICTED rotation
            predicted_poses_7d = np.concatenate(
                [pred_t_sliced_np, pred_q_sliced_np], axis=-1
            )

            # Add batch results to the lists
            list_of_pred_batches.append(predicted_poses_7d)
            list_of_gt_batches.append(gt_poses_sliced_np)

    if not list_of_gt_batches:
        return np.empty((0, 7)), np.empty((0, 7))

    all_gt_poses_3d = np.concatenate(list_of_gt_batches, axis=0)
    all_pred_poses_3d = np.concatenate(list_of_pred_batches, axis=0)

    # Flatten the batch and clip dimensions
    num_total_clips, clip_len, pose_dim = all_gt_poses_3d.shape
    flat_gt_poses = all_gt_poses_3d.reshape(num_total_clips * clip_len, pose_dim)
    flat_pred_poses = all_pred_poses_3d.reshape(num_total_clips * clip_len, pose_dim)

    return flat_gt_poses, flat_pred_poses


# ==============================================================================
# 5. METRIC CALCULATION AND PLOTTING FUNCTION
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
        pos = pose_pred_mat[:3, 3]
        quat = Rotation.from_matrix(pose_pred_mat[:3, :3]).as_quat()  # xyzw format
        pred_poses_7d_abs.append(np.concatenate([pos, quat]))

    # Convert lists to numpy arrays
    gt_trajectory, pred_trajectory = np.array(gt_trajectory), np.array(pred_trajectory)
    gt_poses_7d_abs, pred_poses_7d_abs = np.array(gt_poses_7d_abs), np.array(
        pred_poses_7d_abs
    )

    # Save the 3D (x,y,z) trajectory as before
    pred_traj_path = results_dir / f"predicted_trajectory_{sequence_name}.npy"
    gt_traj_path = results_dir / f"gt_trajectory_{sequence_name}.npy"
    np.save(pred_traj_path, pred_trajectory)
    np.save(gt_traj_path, gt_trajectory)
    print(f"  -> 3D trajectory (positions) saved to: '{pred_traj_path}'")

    # Save the full 7D poses (position + quaternion) for fusion
    pred_poses_7d_path = results_dir / f"predicted_poses_7d_{sequence_name}.npy"
    gt_poses_7d_path = results_dir / f"gt_poses_7d_{sequence_name}.npy"
    np.save(pred_poses_7d_path, pred_poses_7d_abs)
    np.save(gt_poses_7d_path, gt_poses_7d_abs)
    print(f"  -> 7D poses (pos+rot) saved to: '{pred_poses_7d_path}'")
    # ---------------------------------------------

    # 2. Metric calculation
    ate = calculate_ate_aligned(gt_trajectory, pred_trajectory)
    angular_errors = [
        quaternion_angular_error_numpy(p[3:], g[3:])
        for p, g in zip(pred_relative_poses, gt_relative_poses)
    ]
    total_dist = np.sum(np.linalg.norm(gt_relative_poses[:, :3], axis=1))
    rot_drift = (
        np.sqrt(np.mean(np.array(angular_errors) ** 2)) * 100 / total_dist
        if total_dist > 1
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

    return ate, rot_drift


# ==============================================================================
# 6. MAIN SCRIPT
# ==============================================================================
def main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- List of sequences to evaluate ---
    ALL_SEQUENCES = [
        #'URBAN_A0','LOOP_A0', 'LOOP_B0', 'LOOP_C0', 'LOOP_E0', 'URBAN_A1', 'RURAL_A0', 'RURAL_A1', 'RURAL_A2', 'RURAL_B2', 'RURAL_B1',
        #'URBAN_C0','URBAN_C1','URBAN_D0','URBAN_D1','URBAN_E0','URBAN_E1', 'URBAN_F0','URBAN_F1', 'URBAN_G0','URGAN_G1','URBAN_H0','URBAN_H1',
        #'RURAL_C0','RURAL_C1','RURAL_C2','RURAL_D0','RURAL_D1','RURAL_D2','RURAL_E0','RURAL_E1','RURAL_E2','RURAL_F0','RURAL_F1','RURAL_F2',
        "LOOP_D0",
        "RURAL_B0",
    ]

    # Create results directory if it doesn't exist
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    # --- Mamba Model Initialization ---
    print("Initializing OdometryMamba model...")
    model = OdometryMamba(
        input_features=6,
        embed_dim=256,  # Equivalent to hidden_units
        num_layers_temporal=4,  # Number of temporal Mamba blocks
        dropout_prob=0.3,
    ).to(DEVICE)

    # --- Load trained model weights ---
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=DEVICE)

    # Check if checkpoint is a dictionary (saved per epoch) or just the state_dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    results = []

    # --- Evaluation loop over each sequence ---
    for seq_name in ALL_SEQUENCES:
        print(f"\n{'='*20} Evaluating sequence: {seq_name} {'='*20}")

        # Create a dataset for the single sequence.
        dataset = IMUOdometryDataset(
            root_dir=args.data_dir,
            mode="val",
            sequence_names=[seq_name],
            clip_length=args.clip_len,
            stride=4,  # Use stride=4 for evaluation
        )

        if len(dataset) == 0:
            print(
                f"WARNING: No data found for sequence '{seq_name}'. Skipping sequence."
            )
            continue

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=imu_collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # Run inference and get all relative poses
        gt_poses, pred_poses = evaluate_sequence(model, dataloader, DEVICE)

        if gt_poses.shape[0] == 0:
            print(
                f"WARNING: No valid poses generated for '{seq_name}'. Skipping sequence."
            )
            continue

        # Calculate metrics and generate plots
        ate, rot_drift = compute_metrics_and_plot(
            gt_poses, pred_poses, seq_name, results_dir
        )

        results.append(
            {
                "Sequence": seq_name,
                "ATE (m)": ate,
                "Rotational Drift (°/100m)": rot_drift,
            }
        )

    # --- Print final report ---
    if not results:
        print("\nNo sequences were evaluated. Exiting.")
        return

    print(f"\n{'='*25} FINAL EVALUATION REPORT (Mamba) {'='*25}")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Calculate and print averages
    mean_ate = results_df["ATE (m)"].mean()
    mean_drift = results_df["Rotational Drift (°/100m)"].mean()
    print("\n--- Overall Averages ---")
    print(f"Mean ATE: {mean_ate:.4f} m")
    print(f"Mean Rotational Drift: {mean_drift:.4f} °/100m")

    # Save the report to a CSV file
    report_path = results_dir / "evaluation_report_imu_mamba.csv"
    results_df.to_csv(report_path, index=False)
    print(f"\nReport saved to: '{report_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Mamba-based IMU Odometry model."
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
        help="Path to the trained Mamba .pth model file (e.g., 'training_IMU_Mamba/best_model_ate.pth').",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation_results_IMU_Mamba_ABLATION",
        help="Directory to save trajectory plots and the final report.",
    )
    parser.add_argument(
        "--clip_len",
        type=int,
        default=5,
        help="Clip length used during model training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference (can be larger than training).",
    )

    args = parser.parse_args()
    main(args)
