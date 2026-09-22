import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.layers import DropPath
from dataset_imu import IMUOdometryDataset, imu_collate_fn
from losses import AdaptiveLoss, quaternion_angular_error

# ==============================================================================
# 2. MAMBA ARCHITECTURE
# ==============================================================================

# --- MAMBA BUILDING BLOCKS ---


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
    """A single Mamba block with bi-directional processing and residual connections."""

    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MambaLayer(dim)
        self.mlp = FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # Bi-directional Mamba processing
        forward_pass = self.attn(x)
        backward_pass = self.attn(x.flip(1)).flip(1)

        x = x + self.drop_path(forward_pass) + self.drop_path(backward_pass)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --- ODOMETRY ARCHITECTURE ---


class IMUEncoderMamba(nn.Module):
    """Encodes a sequence of IMU data into a single feature vector."""

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
        return x[:, -1, :]  # Take the output of the last token


class OdometryMamba(nn.Module):
    """
    IMU Odometry model using Mamba for both spatial (per-clip)
    and temporal (across-clips) feature extraction.
    """

    def __init__(
        self, input_features=6, embed_dim=256, num_layers_temporal=4, dropout_prob=0.3
    ):
        super().__init__()
        self.encoder = IMUEncoderMamba(
            input_features=input_features, embed_dim=embed_dim, num_layers=2
        )

        # Temporal Mamba blocks to process the sequence of clip features
        dpr = [x.item() for x in torch.linspace(0, 0.4, num_layers_temporal)]
        self.temporal_mamba = nn.ModuleList(
            [
                Block_mamba(dim=embed_dim, drop_path=dpr[i])
                for i in range(num_layers_temporal)
            ]
        )

        # Regressors for quaternion (rotation)
        self.regressor_q = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=embed_dim // 2, out_features=4),
        )
        # Regressors for translation
        self.regressor_t = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(in_features=embed_dim // 2, out_features=3),
        )

    def forward(self, imu_clips):
        if imu_clips.shape[-1] != 6:
            imu_clips = imu_clips[..., :6]
        b, c, s, f = imu_clips.shape  # Batch, Clip_len, Seq_len, Features

        # Encode each clip's IMU sequence independently
        imu_reshaped = imu_clips.view(b * c, s, f)
        features = self.encoder(imu_reshaped)
        features_reshaped = features.view(b, c, -1)  # [B, C, Embed_Dim]

        # Process the sequence of clip features
        temporal_out = features_reshaped
        for layer in self.temporal_mamba:
            temporal_out = layer(temporal_out)

        pred_q = self.regressor_q(temporal_out)
        pred_t = self.regressor_t(temporal_out)

        # Normalize the quaternion
        pred_q_norm = F.normalize(pred_q, p=2, dim=-1)

        return pred_q_norm, pred_t


# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def pose_7d_to_matrix(pose_vec):
    """Converts 7D pose (t, q_xyzw) to a 4x4 transformation matrix."""
    t, q_xyzw = pose_vec[:3], pose_vec[3:]
    T = np.eye(4)
    if np.linalg.norm(q_xyzw) > 1e-8:  # Safety check for zero quaternion
        T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t
    return T


def calculate_ate_aligned(gt_trajectory, pred_trajectory):
    """Calculates the Absolute Trajectory Error (ATE) after alignment."""
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
    """Calculates angular error in degrees for numpy quaternions."""
    dot = np.clip(np.abs(np.sum(q1 * q2)), -1.0, 1.0)
    return 2 * np.arccos(dot) * (180.0 / np.pi)


# ==============================================================================
# 4. TRAIN/VALIDATION LOOP
# ==============================================================================
def run_epoch(
    model, dataloader, loss_fn, optimizer, epoch_num, device, save_dir, is_training
):
    if is_training:
        model.train()
        desc = f"🚀 Training Epoch {epoch_num}"
    else:
        model.eval()
        desc = f"🧐 Validating Epoch {epoch_num}"

    # Initialize accumulators for L1 (pbar) and SSE (RMSE)
    total_loss, total_q_err, total_t_l1_err = 0.0, 0.0, 0.0
    total_t_sse = 0.0  # Sum of Squared Errors for RMSE
    total_samples = 0  # Total samples (B * C) for RMSE

    all_gt_by_seq, all_pred_by_seq = defaultdict(list), defaultdict(list)

    pbar = tqdm(dataloader, desc=desc, unit="batch")
    for batch in pbar:
        imu_sequences = batch["imu"].to(device, non_blocking=True)
        gt_poses_batch = batch["gt_poses"].to(device, non_blocking=True)

        if imu_sequences.shape[-1] == 7:
            imu_sequences = imu_sequences[..., :6]  # Ensure only 6 features are used

        with torch.set_grad_enabled(is_training):
            predicted_q_sequence, predicted_t_sequence = model(imu_sequences)

        gt_q_sequence = gt_poses_batch[..., 3:]
        gt_t_sequence = gt_poses_batch[..., :3]

        # Ensure sequence lengths match if there's a mismatch
        effective_clip_len = min(predicted_q_sequence.shape[1], gt_q_sequence.shape[1])

        pred_q_sliced = predicted_q_sequence[:, :effective_clip_len, :]
        gt_q_sliced = gt_q_sequence[:, :effective_clip_len, :]
        pred_t_sliced = predicted_t_sequence[:, :effective_clip_len, :]
        gt_t_sliced = gt_t_sequence[:, :effective_clip_len, :]

        with torch.set_grad_enabled(is_training):
            loss = loss_fn(pred_q_sliced, pred_t_sliced, gt_q_sliced, gt_t_sliced)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            # Aggregate poses for ATE calculation
            seq_names = batch["sequence_name"]
            pred_t_np = pred_t_sliced.detach().cpu().numpy()
            pred_q_np = pred_q_sliced.detach().cpu().numpy()

            for i in range(len(seq_names)):
                for t in range(effective_clip_len):
                    pred_pose_7d = np.concatenate((pred_t_np[i, t], pred_q_np[i, t]))
                    all_pred_by_seq[seq_names[i]].append(pred_pose_7d)
                    all_gt_by_seq[seq_names[i]].append(
                        gt_poses_batch[i, t].cpu().numpy()
                    )

        total_loss += loss.item()
        with torch.no_grad():
            batch_q_err = quaternion_angular_error(pred_q_sliced, gt_q_sliced).mean()
            total_q_err += batch_q_err.item()

            # 1. L1 (MAE) for PBar (fast)
            batch_t_err_l1 = F.l1_loss(pred_t_sliced, gt_t_sliced)
            total_t_l1_err += batch_t_err_l1.item()

            # 2. SSE (Sum of Squared Errors) for RMSE
            batch_t_err_mse = F.mse_loss(pred_t_sliced, gt_t_sliced)
            num_samples_in_batch = pred_t_sliced.numel() / 3.0  # (B*C*3) / 3 = B*C
            total_t_sse += batch_t_err_mse.item() * num_samples_in_batch
            total_samples += num_samples_in_batch

        # pbar uses t_mae (L1)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            q_err=f"{batch_q_err.item():.2f}°",
            t_mae=f"{batch_t_err_l1.item():.4f}m",
        )

    avg_loss = total_loss / len(dataloader)
    avg_q_err = total_q_err / len(dataloader)

    # --- KEY CHANGE: Calculate total epoch RMSE ---
    avg_t_rmse = (
        torch.sqrt(torch.tensor(total_t_sse / total_samples)).item()
        if total_samples > 0
        else 0.0
    )

    if not is_training:
        epoch_metrics = defaultdict(list)
        val_plot_subdir = save_dir / "val_trajectories"
        val_plot_subdir.mkdir(exist_ok=True)
        print("\nCalculating validation ATE and Drift...")
        for seq_name, gt_rel_poses in all_gt_by_seq.items():
            pred_rel_poses = np.array(all_pred_by_seq[seq_name])
            gt_rel_poses = np.array(gt_rel_poses)

            pose_gt, gt_trajectory = np.eye(4), [np.zeros(3)]
            for rel_pose in gt_rel_poses:
                pose_gt = pose_gt @ pose_7d_to_matrix(rel_pose)
                gt_trajectory.append(pose_gt[:3, 3])

            pose_pred, pred_trajectory = np.eye(4), [np.zeros(3)]
            for rel_pose in pred_rel_poses:
                pose_pred = pose_pred @ pose_7d_to_matrix(rel_pose)
                pred_trajectory.append(pose_pred[:3, 3])

            gt_trajectory, pred_trajectory = np.array(gt_trajectory), np.array(
                pred_trajectory
            )

            ate = calculate_ate_aligned(gt_trajectory, pred_trajectory)
            angular_errors_np = [
                quaternion_angular_error_numpy(p[3:], g[3:])
                for p, g in zip(pred_rel_poses, gt_rel_poses)
            ]
            total_dist = np.sum(np.linalg.norm(gt_rel_poses[:, :3], axis=1))
            # Calculate drift as ° per 100m
            rot_drift = (
                np.sqrt(np.mean(np.array(angular_errors_np) ** 2)) * 100 / total_dist
                if total_dist > 1
                else 0.0
            )

            epoch_metrics["ate"].append(ate)
            epoch_metrics["rot_drift"].append(rot_drift)

            plt.figure(figsize=(10, 10))
            plt.plot(
                gt_trajectory[:, 0], gt_trajectory[:, 1], "b-", label="Ground Truth"
            )
            plt.plot(
                pred_trajectory[:, 0], pred_trajectory[:, 1], "r--", label="Predicted"
            )
            plt.title(
                f"Validation Trajectory - {seq_name} - Epoch {epoch_num} (ATE: {ate:.4f} m)"
            )
            plt.xlabel("X [m]"), plt.ylabel("Y [m]"), plt.legend(), plt.grid(
                True
            ), plt.axis("equal")
            plt.savefig(val_plot_subdir / f"traj_{seq_name}_epoch_{epoch_num:03d}.png")
            plt.close()

        avg_ate = np.mean(epoch_metrics["ate"]) if epoch_metrics["ate"] else 0.0
        avg_rot_drift = (
            np.mean(epoch_metrics["rot_drift"]) if epoch_metrics["rot_drift"] else 0.0
        )
        # Return avg_t_rmse
        return avg_loss, avg_q_err, avg_t_rmse, avg_ate, avg_rot_drift

    # Return avg_t_rmse
    return avg_loss, avg_q_err, avg_t_rmse, None, None


# ==============================================================================
# 5. PLOTTING FUNCTION (Modified for RMSE labels)
# ==============================================================================
def plot_final_history(history, save_dir):
    """Generates and saves final summary plots."""
    save_dir = Path(save_dir)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axs = plt.subplots(3, 2, figsize=(20, 22))
    fig.suptitle("Training & Validation History", fontsize=20)

    # Plot 1: Total Loss
    axs[0, 0].plot(epochs, history["train_loss"], "bo-", label="Training Loss")
    axs[0, 0].plot(epochs, history["val_loss"], "ro-", label="Validation Loss")
    axs[0, 0].set_title("Total Adaptive Loss")
    axs[0, 0].set_ylabel("Loss (adaptive)")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Plot 2: Angular Error (Rotation)
    axs[0, 1].plot(epochs, history["train_rot_err"], "b-o", label="Training")
    axs[0, 1].plot(epochs, history["val_rot_err"], "r-o", label="Validation")
    axs[0, 1].set_title("Mean Angular Error (Rotation)")
    axs[0, 1].set_ylabel("Error (°)")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # --- KEY CHANGE: Plot 3: Translation RMSE Error ---
    axs[1, 0].plot(epochs, history["train_t_err"], "b-o", label="Training")
    axs[1, 0].plot(epochs, history["val_t_err"], "r-o", label="Validation")
    axs[1, 0].set_title("Mean RMSE Error (Translation)")  # Updated title
    axs[1, 0].set_ylabel("Error (RMSE, m)")  # Updated label
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Plot 4: ATE
    axs[1, 1].plot(epochs, history["val_ate"], "go-", label="Validation ATE")
    axs[1, 1].set_title("Global Metric: ATE RMSE")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("ATE RMSE (m)")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # Plot 5: Rotational Drift
    axs[2, 0].plot(epochs, history["val_rot_drift"], "mo-", label="Validation Drift")
    axs[2, 0].set_title("Global Metric: Rotational Drift")
    axs[2, 0].set_xlabel("Epoch")
    axs[2, 0].set_ylabel("Drift (°/100m)")
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    axs[2, 1].axis("off")  # Turn off empty subplot

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path = save_dir / "final_summary_plots.png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"\n📈 Final summary plots saved to '{output_path}'")


# ==============================================================================
# 6. MAIN SCRIPT (Modified to log RMSE)
# ==============================================================================
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT_DIR = Path("/media/arrubuntu20/SSD_2/Riccardo/Extracted Dataset")
    CLIP_LEN = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 65
    STRIDE = 4
    NUM_WORKERS = 8

    plot_save_dir = Path("training_IMU_Mamba_ABLATION_STUDY")
    plot_save_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = plot_save_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    print(
        f"Device: {DEVICE}, Clip: {CLIP_LEN}, Batch: {BATCH_SIZE}, Workers: {NUM_WORKERS}"
    )

    train_dataset = IMUOdometryDataset(
        root_dir=ROOT_DIR, mode="train", clip_length=CLIP_LEN, stride=STRIDE
    )
    val_dataset = IMUOdometryDataset(
        root_dir=ROOT_DIR, mode="val", clip_length=CLIP_LEN, stride=STRIDE
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=imu_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=imu_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = OdometryMamba(
        input_features=6, embed_dim=256, num_layers_temporal=4, dropout_prob=0.3
    ).to(DEVICE)

    loss_fn = AdaptiveLoss().to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-5,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = defaultdict(list)
    best_val_ate = float("inf")

    print("--- Starting Training (Mamba Strategy + AdaptiveLoss (t, q)) ---")
    for epoch in range(1, EPOCHS + 1):

        # Capture train_t_rmse
        train_loss, train_q_err, train_t_rmse, _, _ = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            epoch,
            DEVICE,
            plot_save_dir,
            is_training=True,
        )
        val_loss, val_q_err, val_t_rmse, val_ate, val_rot_drift = run_epoch(
            model,
            val_loader,
            loss_fn,
            None,
            epoch,
            DEVICE,
            plot_save_dir,
            is_training=False,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rot_err"].append(train_q_err)
        history["val_rot_err"].append(val_q_err)
        # Save RMSE to history
        history["train_t_err"].append(train_t_rmse)
        history["val_t_err"].append(val_t_rmse)
        history["val_ate"].append(val_ate)
        history["val_rot_drift"].append(val_rot_drift)

        # --- KEY CHANGE: Print summary with RMSE ---
        print(f"\n--- Epoch {epoch}/{EPOCHS} Summary ---")
        print(
            f"  Training:   Loss: {train_loss:.4f}, Err_q: {train_q_err:.2f}°, Err_t (RMSE): {train_t_rmse:.4f} m"
        )
        print(
            f"  Validation: Loss: {val_loss:.4f}, Err_q: {val_q_err:.2f}°, Err_t (RMSE): {val_t_rmse:.4f} m, ATE: {val_ate:.4f} m, Drift: {val_rot_drift:.4f} °/100m"
        )
        print(
            f"  Loss Weights: w_q = {loss_fn.w_q.item():.3f} (exp: {torch.exp(loss_fn.w_q).item():.4f}), w_t = {loss_fn.w_t.item():.3f} (exp: {torch.exp(loss_fn.w_t).item():.4f})"
        )

        scheduler.step(val_loss)

        if val_ate < best_val_ate:
            best_val_ate = val_ate
            torch.save(model.state_dict(), plot_save_dir / "best_model_ate.pth")
            print(f"  🚀 New best model for ATE saved (Val ATE: {best_val_ate:.4f} m)")

        torch.save(
            {"epoch": epoch, "model_state_file": model.state_dict()},
            checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pth",
        )

        if epoch > 1:
            plot_final_history(history, plot_save_dir)

    print("\n--- Training Finished ---")
    torch.save(model.state_dict(), plot_save_dir / "final_model.pth")


if __name__ == "__main__":
    main()
