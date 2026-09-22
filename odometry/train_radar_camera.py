import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms

# AMP: Import autocast and GradScaler
from torch.cuda.amp import autocast, GradScaler

# --- Import custom modules ---
from dataset_multimodal import UnifiedOdometryDataset, unified_collate_fn
from radar_camera_encoder import OdometryPipeline
from losses import AdaptiveLoss, quaternion_angular_error


# ==============================================================================
# 1. SUPPORT FUNCTIONS AND METRICS
# ==============================================================================
def pose_7d_to_matrix(pose_vec):
    """
    Converts a 7D pose vector (3 translation + 4 quaternion) into a 4x4 matrix.
    This version is robust against zero-norm or non-unit quaternions.
    """
    t, q_xyzw = pose_vec[:3], pose_vec[3:]

    norm = np.linalg.norm(q_xyzw)
    if norm < 1e-6:
        # Safety check for zero vector: assume no rotation
        T = np.eye(4)
        T[:3, 3] = t
        return T

    q_xyzw_normalized = q_xyzw / norm

    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw_normalized).as_matrix()
    T[:3, 3] = t
    return T


def calculate_ate(gt_trajectory, pred_trajectory):
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
    pred_aligned = (R @ pred_centered.T).T
    errors = gt_centered - pred_aligned
    return np.sqrt(np.mean(np.sum(errors**2, axis=1)))


def quaternion_angular_error_numpy(q1, q2):
    """Calculates angular error in degrees for numpy quaternions."""
    dot = np.clip(np.abs(np.sum(q1 * q2)), -1.0, 1.0)
    return 2 * np.arccos(dot) * (180.0 / np.pi)


# ==============================================================================
# 2. TRAIN/VALIDATION LOOP (MODIFIED FOR BPTT and AMP)
# ==============================================================================
def run_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    epoch_num,
    device,
    save_dir,
    is_training,
):
    if is_training:
        model.train()
        criterion.train()
        desc = f"🚀 Training Epoch {epoch_num}"
    else:
        model.eval()
        desc = f"🧐 Validating Epoch {epoch_num}"

    total_loss, total_t_err, total_q_err = 0.0, 0.0, 0.0
    all_gt_by_seq, all_pred_by_seq = defaultdict(list), defaultdict(list)

    pbar = tqdm(dataloader, desc=desc, unit="batch")
    for batch in pbar:
        # Load all necessary data from the batch
        radars = batch["radars"].to(device)
        images = batch["images"].to(device)
        K = batch["K"].to(device)
        T_radar_to_cam = batch["T_radar_to_cam"].to(device)
        radar_lengths = batch["radar_lengths"].to(device)
        gt_poses_batch = batch["poses_gt"].to(device)
        T_imu_from_radar = batch["T_imu_from_radar"].to(device)

        B, clip_len_full, _, _, _ = images.shape
        clip_len_poses = clip_len_full - 1

        history = None
        predicted_poses_for_validation = []  # Only used in validation

        # --- BPTT: Loop to process the sequence step-by-step ---
        for t in range(clip_len_poses):
            if is_training:
                optimizer.zero_grad()  # Zero gradient at each step

            # --- AMP: Enable autocast ---
            with autocast(enabled=False):  # (device.type == 'cuda')):
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
                }

                # --- FORWARD PASS (for 1 step) ---
                with torch.set_grad_enabled(is_training):
                    predicted_pose, new_history = model(sub_batch, history)

                # Detach history to prevent gradients from flowing back in time
                history = new_history.detach() if new_history is not None else None

                # --- LOSS CALCULATION (for 1 step) ---
                current_gt_pose = gt_poses_batch[:, t, :]
                pred_q_step, pred_t_step = (
                    predicted_pose[..., 3:],
                    predicted_pose[..., :3],
                )
                gt_q_step, gt_t_step = (
                    current_gt_pose[..., 3:],
                    current_gt_pose[..., :3],
                )

                with torch.set_grad_enabled(is_training):
                    step_loss = criterion(
                        pred_q_step, pred_t_step, gt_q_step, gt_t_step
                    )

            # --- BPTT ---
            if is_training:
                scaler.scale(step_loss).backward()  # Scale loss
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)  # Optimizer step
                scaler.update()  # Update scaler

            total_loss += step_loss.item()
            with torch.no_grad():
                batch_t_err = F.mse_loss(pred_t_step, gt_t_step)
                batch_q_err = quaternion_angular_error(pred_q_step, gt_q_step).mean()
                total_t_err += batch_t_err.item()
                total_q_err += batch_q_err.item()

            if is_training:
                pbar.set_postfix(
                    step_loss=f"{step_loss.item():.4f}",
                    t_err=f"{batch_t_err.item():.4f}",
                    q_err=f"{batch_q_err.item():.2f}°",
                    w_q=f"{criterion.w_q.item():.2f}",
                    w_t=f"{criterion.w_t.item():.2f}",
                )

            if not is_training:
                predicted_poses_for_validation.append(predicted_pose)

        # --- END OF BPTT LOOP ---

        if not is_training:
            # Reconstruct the full predicted pose tensor for the batch
            predicted_poses = torch.stack(predicted_poses_for_validation, dim=1)

            # Aggregate poses by sequence name for ATE calculation
            gt_np = gt_poses_batch.cpu().numpy()
            pred_np = predicted_poses.detach().cpu().numpy()
            seq_names = batch["sequence_name"]
            for i in range(B):
                all_pred_by_seq[seq_names[i]].extend(pred_np[i])
                all_gt_by_seq[seq_names[i]].extend(gt_np[i])

            # Calculate average batch loss for validation postfix
            avg_batch_loss = (
                (total_loss / (pbar.n + 1) / clip_len_poses) if (pbar.n + 1) > 0 else 0
            )
            pbar.set_postfix(clip_loss=f"{avg_batch_loss:.4f}")

    # --- END OF BATCH LOOP ---

    # --- BPTT: Normalize by total number of steps ---
    num_total_steps = len(dataloader) * clip_len_poses
    avg_loss = total_loss / num_total_steps
    avg_t_err = total_t_err / num_total_steps
    avg_q_err = total_q_err / num_total_steps

    if is_training:
        return avg_loss, avg_t_err, avg_q_err

    # --- End-of-epoch logic (Validation only) ---
    epoch_metrics = defaultdict(list)
    val_plot_subdir = save_dir / "val_trajectories"
    val_plot_subdir.mkdir(exist_ok=True, parents=True)

    print("\nCalculating validation ATE and Drift...")
    for seq_name, gt_poses_list in all_gt_by_seq.items():
        pred_poses_list = np.array(all_pred_by_seq[seq_name])
        gt_poses_list = np.array(gt_poses_list)

        # Reconstruct ground truth trajectory
        pose_gt, gt_trajectory = np.eye(4), [np.zeros(3)]
        for rel_pose in gt_poses_list:
            pose_gt = pose_gt @ pose_7d_to_matrix(rel_pose)
            gt_trajectory.append(pose_gt[:3, 3])

        # Reconstruct predicted trajectory
        pose_pred, pred_trajectory = np.eye(4), [np.zeros(3)]
        for rel_pose in pred_poses_list:
            pose_pred = pose_pred @ pose_7d_to_matrix(rel_pose)
            pred_trajectory.append(pose_pred[:3, 3])

        gt_trajectory, pred_trajectory = np.array(gt_trajectory), np.array(
            pred_trajectory
        )

        # Calculate metrics
        ate = calculate_ate(gt_trajectory, pred_trajectory)
        angular_errors = [
            quaternion_angular_error_numpy(p[3:], g[3:])
            for p, g in zip(pred_poses_list, gt_poses_list)
        ]
        total_dist = np.sum(np.linalg.norm(gt_poses_list[:, :3], axis=1))
        rot_drift = (
            np.sqrt(np.mean(np.array(angular_errors) ** 2)) / total_dist
            if total_dist > 0
            else 0.0
        )

        epoch_metrics["ate"].append(ate)
        epoch_metrics["rot_drift"].append(rot_drift)
        print(f"    -> Seq: {seq_name}, ATE: {ate:.4f} m, Drift: {rot_drift:.4f} °/m")

        # Plot trajectory
        plt.figure(figsize=(10, 10))
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], "b-", label="Ground Truth")
        plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], "r--", label="Predicted")
        plt.title(
            f"Validation Trajectory - {seq_name} - Epoch {epoch_num} (ATE: {ate:.4f} m)"
        )
        plt.xlabel("X [m]"), plt.ylabel("Y [m]"), plt.legend(), plt.grid(
            True
        ), plt.axis("equal")
        plt.savefig(val_plot_subdir / f"traj_{seq_name}_epoch_{epoch_num:03d}.png")
        plt.close()

    avg_ate = np.mean(epoch_metrics["ate"])
    avg_rot_drift = np.mean(epoch_metrics["rot_drift"])

    return avg_loss, avg_t_err, avg_q_err, avg_ate, avg_rot_drift


# ==============================================================================
# 3. HISTORY PLOTTING FUNCTION
# ==============================================================================
def plot_final_history(history, save_dir):
    """
    Generates and saves final summary plots for training and validation.
    """
    save_dir = Path(save_dir)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axs = plt.subplots(3, 2, figsize=(20, 22))
    fig.suptitle("Training & Validation History", fontsize=20, y=0.98)

    axs[0, 0].plot(epochs, history["train_loss"], "bo-", label="Training Loss")
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    train_trans_rmse = np.sqrt(history["train_trans_err"])
    val_trans_rmse = np.sqrt(history["val_trans_err"])
    axs[0, 1].plot(epochs, train_trans_rmse, "b-o", label="Training")
    axs[0, 1].plot(epochs, val_trans_rmse, "r-o", label="Validation")
    axs[0, 1].set_title("Translation Error (RMSE)")
    axs[0, 1].set_ylabel("Error (m)")
    axs[0, 1].grid(True, which="both")
    axs[0, 1].legend()

    axs[1, 0].plot(epochs, history["train_rot_err"], "b-o", label="Training")
    axs[1, 0].plot(epochs, history["val_rot_err"], "r-o", label="Validation")
    axs[1, 0].set_title("Mean Angular Error")
    axs[1, 0].set_ylabel("Error (°)")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(epochs, history["val_ate"], "go-", label="Validation ATE")
    axs[1, 1].set_title("Global Metric: ATE RMSE")
    axs[1, 1].set_ylabel("ATE RMSE (m)")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    axs[2, 0].plot(epochs, history["val_rot_drift"], "mo-", label="Validation Drift")
    axs[2, 0].set_title("Global Metric: Rotational Drift")
    axs[2, 0].set_xlabel("Epoch")
    axs[2, 0].set_ylabel("Drift (°/m)")
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    axs[1, 1].set_xlabel("Epoch")
    fig.delaxes(axs[2, 1])  # Delete unused subplot

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    output_path = save_dir / "final_summary_plots.png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"\n📈 Final summary plots saved to '{output_path}'")


# ==============================================================================
# 4. MAIN SCRIPT (MODIFIED FOR AMP)
# ==============================================================================
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Hyperparameters ---
    ROOT_DIR = "/media/arrubuntu20/SSD_2/Riccardo/Extracted Dataset"
    CLIP_LEN = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 65
    STRIDE = 4
    WEIGHT_DECAY = 1e-5

    plot_save_dir = Path("training_RadarCamera_ABLATION_STUDY")
    plot_save_dir.mkdir(exist_ok=True, parents=True)

    checkpoints_dir = plot_save_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    image_transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # --- Dataloaders ---
    train_dataset = UnifiedOdometryDataset(
        root_dir=ROOT_DIR,
        mode="train",
        clip_length=CLIP_LEN,
        stride=STRIDE,
        image_transform=image_transform,
    )
    val_dataset = UnifiedOdometryDataset(
        root_dir=ROOT_DIR,
        mode="val",
        clip_length=CLIP_LEN,
        stride=STRIDE,
        image_transform=image_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=unified_collate_fn,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=unified_collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    # --- Model, Loss, Optimizer ---
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
    criterion = AdaptiveLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # AMP: Initialize GradScaler.
    scaler = GradScaler(enabled=False)  # (enabled=(DEVICE.type == 'cuda'))

    history = defaultdict(list)
    best_val_ate = float("inf")

    # --- Training Loop ---
    for epoch in range(1, EPOCHS + 1):
        # AMP/BPTT: Pass scaler to run_epoch
        train_loss, train_t_err, train_q_err = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            epoch,
            DEVICE,
            plot_save_dir,
            is_training=True,
        )
        # AMP/BPTT: Pass None for optimizer and scaler during validation
        val_loss, val_t_err, val_q_err, val_ate, val_rot_drift = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            None,
            epoch,
            DEVICE,
            plot_save_dir,
            is_training=False,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_trans_err"].append(train_t_err)
        history["val_trans_err"].append(val_t_err)
        history["train_rot_err"].append(train_q_err)
        history["val_rot_err"].append(val_q_err)
        history["val_ate"].append(val_ate)
        history["val_rot_drift"].append(val_rot_drift)

        print(f"\n--- Epoch {epoch}/{EPOCHS} Summary ---")
        # Updated to reflect per-step metrics (MSE_t is now smaller)
        print(
            f"  Training:   Loss: {train_loss:.4f}, MSE_t: {train_t_err:.6f}, Err_q: {train_q_err:.2f}°"
        )
        print(
            f"  Validation: Loss: {val_loss:.4f}, MSE_t: {val_t_err:.6f}, Err_q: {val_q_err:.2f}°, ATE: {val_ate:.4f} m, Drift: {val_rot_drift:.4f} °/m"
        )

        scheduler.step(val_loss)

        if val_ate < best_val_ate:
            best_val_ate = val_ate
            torch.save(model.state_dict(), plot_save_dir / "best_model_ate.pth")
            print(
                f"    🚀 New best model for ATE saved (Val ATE: {best_val_ate:.4f} m)"
            )

        checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_ate": best_val_ate,
            },
            checkpoint_path,
        )
        print(f"    💾 Checkpoint saved to '{checkpoint_path}'")

        if epoch > 1:
            plot_final_history(history, plot_save_dir)

    print("\n--- Training Finished ---")
    torch.save(model.state_dict(), plot_save_dir / "final_model.pth")


if __name__ == "__main__":
    main()
