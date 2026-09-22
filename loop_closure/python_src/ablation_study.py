import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import scipy.linalg  # For Umeyama/alignment
import warnings

# --- Helper Functions (Pose, GT, ATE) ---


def pose_7d_to_matrix(pose_7d):
    """
    Converts pose [tx,ty,tz, qx,qy,qz,qw] to a 4x4 matrix.
    (Uses SciPy)
    """
    if pose_7d is None or len(pose_7d) != 7:
        return None
    t = pose_7d[:3]
    q_xyzw = pose_7d[3:]
    if (
        np.isnan(t).any()
        or np.isinf(t).any()
        or np.isnan(q_xyzw).any()
        or np.isinf(q_xyzw).any()
    ):
        return None
    q_norm = np.linalg.norm(q_xyzw)
    if q_norm < 1e-6:
        return None
    q_xyzw = q_xyzw / q_norm
    try:
        rot_mat = R.from_quat(q_xyzw).as_matrix()
    except ValueError as e:
        return None
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = t
    return T.astype(np.float32)


def reconstruct_gt_trajectory(gt_relative_df, indices_df, gt_lidar_col_name):
    """Reconstructs the absolute GT trajectory from relative GT poses."""
    print("Reconstructing GT trajectory...")
    gt_absolute_poses_mat = [np.eye(4, dtype=np.float32)]
    gt_relative_poses_mat_valid = []
    num_sync_frames = len(indices_df)
    max_allowable_lidar_index_gt = len(gt_relative_df)
    current_gt_pose = np.eye(4, dtype=np.float32)
    processed_valid_steps = 0

    if gt_lidar_col_name not in indices_df.columns:
        print(
            f"CRITICAL ERROR: GT column '{gt_lidar_col_name}' does not exist in indices_df for GT reconstruction."
        )
        return [], []

    for i in tqdm(range(num_sync_frames), desc="  Reconstructing GT", leave=False):
        try:
            row_i = indices_df.iloc[i]
            lidar_idx_gt = int(row_i[gt_lidar_col_name])

            if lidar_idx_gt < max_allowable_lidar_index_gt:
                gt_rel_pose_row = gt_relative_df.iloc[lidar_idx_gt]
                gt_rel_pose_7d = np.array(
                    [
                        gt_rel_pose_row[k]
                        for k in ["dx", "dy", "dz", "qx", "qy", "qz", "qw"]
                    ]
                )
                T_rel_gt = pose_7d_to_matrix(gt_rel_pose_7d)
                if T_rel_gt is None:
                    raise ValueError(f"Invalid relative GT pose {lidar_idx_gt}")
                T_rel_gt = T_rel_gt.astype(np.float32)
                gt_relative_poses_mat_valid.append(T_rel_gt.copy())
                current_gt_pose = current_gt_pose @ T_rel_gt
                gt_absolute_poses_mat.append(current_gt_pose.copy())
                processed_valid_steps += 1
            else:
                break
        except (KeyError, ValueError, IndexError, TypeError) as e:
            break
    num_abs_poses = processed_valid_steps + 1
    gt_absolute_poses_mat = gt_absolute_poses_mat[:num_abs_poses]
    return gt_absolute_poses_mat, gt_relative_poses_mat_valid


# --- Metric Functions (ATE) ---


def align_umeyama(model, data):
    """Calculates the alignment transformation (Sim(3)) from data to model."""
    model_mean = model.mean(axis=0)
    data_mean = data.mean(axis=0)
    model_centered = model - model_mean
    data_centered = data - data_mean
    cov_matrix = data_centered.T @ model_centered / len(model)
    U, S, Vt = scipy.linalg.svd(cov_matrix)
    V = Vt.T
    det_UVt = np.linalg.det(U @ Vt)
    diag_fix = np.diag([1] * (model.shape[1] - 1) + [det_UVt])
    R_align = V @ diag_fix @ U.T
    var_data = np.var(data_centered, axis=0).sum()
    c = np.trace(np.diag(S) @ diag_fix) / var_data if var_data > 1e-8 else 1.0
    t = model_mean - c * R_align @ data_mean
    T = np.eye(4)
    T[:3, :3] = c * R_align
    T[:3, 3] = t
    return T, t, R_align, c


def calculate_ate(gt_poses_mat, estimated_poses_mat):
    """
    Calculates the Absolute Trajectory Error (ATE) RMSE after Umeyama alignment.
    RETURNS: ate_rmse only
    """
    if isinstance(gt_poses_mat, np.ndarray):
        gt_poses_mat = list(gt_poses_mat)
    if isinstance(estimated_poses_mat, np.ndarray):
        estimated_poses_mat = list(estimated_poses_mat)

    if len(gt_poses_mat) != len(estimated_poses_mat) or len(gt_poses_mat) < 2:
        print(
            f"ATE Error: Trajectories do not have the same length ({len(gt_poses_mat)} vs {len(estimated_poses_mat)}) or are too short."
        )
        return None

    gt_positions = np.array([pose[:3, 3] for pose in gt_poses_mat])
    est_positions = np.array([pose[:3, 3] for pose in estimated_poses_mat])

    try:
        T_align, _, _, _ = align_umeyama(gt_positions, est_positions)
    except Exception as e:
        print(f"Error during align_umeyama in calculate_ate: {e}")
        return None

    est_positions_hom = np.hstack((est_positions, np.ones((len(est_positions), 1))))
    est_positions_aligned_hom = (T_align @ est_positions_hom.T).T
    est_positions_aligned = est_positions_aligned_hom[:, :3]
    translation_errors = np.linalg.norm(gt_positions - est_positions_aligned, axis=1)
    ate_rmse = np.sqrt(np.mean(translation_errors**2))

    return ate_rmse


# --- Main Script ---
if __name__ == "__main__":
    # 1. ARGUMENT PARSING
    parser = argparse.ArgumentParser(
        description="Compare 3 predicted trajectories with GT (plot only)."
    )
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/arrubuntu20/SSD_2/Riccardo/Extracted Dataset",
    )
    parser.add_argument("--sequence_name", type=str, required=True)

    # Model arguments
    parser.add_argument(
        "--results_dir_rci",
        type=str,
        required=True,
        help="Results folder for Model 1 (e.g., RCI)",
    )
    parser.add_argument(
        "--model_1_name",
        type=str,
        default="Model-1",
        help="Name for the legend (e.g., Ours)",
    )

    parser.add_argument(
        "--results_dir_m2", type=str, required=True, help="Results folder for Model 2"
    )
    parser.add_argument(
        "--model_2_name",
        type=str,
        default="Model-2",
        help="Name for the legend (e.g., Model-B)",
    )

    parser.add_argument(
        "--results_dir_m3", type=str, required=True, help="Results folder for Model 3"
    )
    parser.add_argument(
        "--model_3_name",
        type=str,
        default="Model-3",
        help="Name for the legend (e.g., Model-C)",
    )

    # Plot arguments
    parser.add_argument("--save_plot", action="store_true")
    parser.add_argument(
        "--use_gps_filter",
        action="store_true",
        help="[PLOT ONLY] If specified, use 'East/North' as axis labels (otherwise X/Y).",
    )

    args = parser.parse_args()

    # 2. LOAD AND PREPARE DATA
    print("\nLoading and preparing data (GT + 3 Models)...")
    sequence_dir = Path(args.data_dir) / args.sequence_name
    indices_path = sequence_dir / "synchronized_indices.csv"
    gt_relative_path = sequence_dir / "relative_poses.csv"

    # .npy file paths for the 3 models
    pred_path_rci = (
        Path(args.results_dir_rci) / f"predicted_poses_7d_{args.sequence_name}.npy"
    )
    pred_path_m2 = (
        Path(args.results_dir_m2) / f"predicted_poses_7d_{args.sequence_name}.npy"
    )
    pred_path_m3 = (
        Path(args.results_dir_m3) / f"predicted_poses_7d_{args.sequence_name}.npy"
    )

    if not (
        sequence_dir.exists()
        and indices_path.exists()
        and gt_relative_path.exists()
        and pred_path_rci.exists()
        and pred_path_m2.exists()
        and pred_path_m3.exists()
    ):
        print("ERROR: One or more files/directories missing (GT, Sync, or .npy files).")
        if not pred_path_rci.exists():
            print(f"  Missing: {pred_path_rci}")
        if not pred_path_m2.exists():
            print(f"  Missing: {pred_path_m2}")
        if not pred_path_m3.exists():
            print(f"  Missing: {pred_path_m3}")
        exit()

    try:
        indices_df = pd.read_csv(indices_path)
        gt_relative_df = pd.read_csv(gt_relative_path)
        pred_7d_rci = np.load(pred_path_rci)
        pred_7d_m2 = np.load(pred_path_m2)
        pred_7d_m3 = np.load(pred_path_m3)
    except Exception as e:
        print(f"Error loading base data: {e}")
        exit()

    # Helper function to load and validate poses
    def load_and_validate_poses(poses_7d_np, model_name):
        poses_mat = []
        valid_indices = []
        for i in range(poses_7d_np.shape[0]):
            T_abs = pose_7d_to_matrix(poses_7d_np[i])
            if T_abs is not None:
                poses_mat.append(T_abs)
                valid_indices.append(i)
        print(f"  Loaded {len(poses_mat)} valid poses for {model_name}.")
        return poses_mat, valid_indices

    # Validate and Prepare Predicted Poses
    pred_poses_mat_rci, valid_indices_rci = load_and_validate_poses(
        pred_7d_rci, args.model_1_name
    )
    pred_poses_mat_m2, valid_indices_m2 = load_and_validate_poses(
        pred_7d_m2, args.model_2_name
    )
    pred_poses_mat_m3, valid_indices_m3 = load_and_validate_poses(
        pred_7d_m3, args.model_3_name
    )

    # Find the valid indices COMMON to all 3 models
    common_valid_indices = sorted(
        list(set(valid_indices_rci) & set(valid_indices_m2) & set(valid_indices_m3))
    )

    if len(common_valid_indices) < 2:
        print("Error: Insufficient valid poses (<2) common to all 3 models.")
        exit()

    def filter_poses_by_indices(
        rci_poses,
        rci_indices,
        m2_poses,
        m2_indices,
        m3_poses,
        m3_indices,
        common_indices_list,
    ):
        """
        Filters the three pose lists, keeping only those whose
        original indices are present in the common_indices_list.
        """
        try:
            pose_map_rci = {idx: pose for idx, pose in zip(rci_indices, rci_poses)}
            pose_map_m2 = {idx: pose for idx, pose in zip(m2_indices, m2_poses)}
            pose_map_m3 = {idx: pose for idx, pose in zip(m3_indices, m3_poses)}

            filtered_rci = [pose_map_rci[i] for i in common_indices_list]
            filtered_m2 = [pose_map_m2[i] for i in common_indices_list]
            filtered_m3 = [pose_map_m3[i] for i in common_indices_list]

            return filtered_rci, filtered_m2, filtered_m3

        except KeyError as e:
            print(
                f"CRITICAL ERROR in filter_poses_by_indices: Could not find index {e} in pose maps."
            )
            exit(1)
        except Exception as e:
            print(f"Unexpected error in filter_poses_by_indices: {e}")
            exit(1)

    # Call the filter function
    pred_poses_mat_rci, pred_poses_mat_m2, pred_poses_mat_m3 = filter_poses_by_indices(
        pred_poses_mat_rci,
        valid_indices_rci,
        pred_poses_mat_m2,
        valid_indices_m2,
        pred_poses_mat_m3,
        valid_indices_m3,
        common_valid_indices,
    )

    num_valid_common_poses = len(pred_poses_mat_rci)

    try:
        max_valid_index = max(common_valid_indices) if common_valid_indices else -1
        if len(indices_df) <= max_valid_index:
            raise IndexError(
                f"Error: Index file ({len(indices_df)} rows) shorter than max pose idx ({max_valid_index})."
            )

        # Filter indices_df based on *common* valid indices
        indices_df_filtered = indices_df.iloc[common_valid_indices].reset_index(
            drop=True
        )

        if len(indices_df_filtered) != num_valid_common_poses:
            raise ValueError("Index vs. pose alignment error.")

    except (IndexError, ValueError) as e:
        print(f"Fatal error filtering indices_df: {e}.")
        exit()

    # Reconstruct GT Trajectory (based on filtered indices_df)
    gt_lidar_col_name = "lidar_index"
    gt_absolute_poses_mat, _ = reconstruct_gt_trajectory(
        gt_relative_df, indices_df_filtered, gt_lidar_col_name
    )

    # Align Lists (GT vs Models)
    lengths = [
        len(pred_poses_mat_rci),
        len(pred_poses_mat_m2),
        len(pred_poses_mat_m3),
        len(gt_absolute_poses_mat),
    ]
    final_num_frames = min(lengths)

    print(f"Final alignment to {final_num_frames} common frames.")

    # Apply final cut for consistency
    pred_poses_mat_rci = pred_poses_mat_rci[:final_num_frames]
    pred_poses_mat_m2 = pred_poses_mat_m2[:final_num_frames]
    pred_poses_mat_m3 = pred_poses_mat_m3[:final_num_frames]
    gt_absolute_poses_mat = gt_absolute_poses_mat[:final_num_frames]

    if final_num_frames <= 0:
        print("ERROR: No valid frames after final alignment.")
        exit()

    print(f"Data loading and preparation complete ({final_num_frames} frames).")

    # 3. METRIC CALCULATION (ATE for all inputs)
    print("\n" + "=" * 50)
    print("Calculating Evaluation Metrics (ATE)...")

    ate_rci_input = calculate_ate(gt_absolute_poses_mat, pred_poses_mat_rci)
    ate_m2_input = calculate_ate(gt_absolute_poses_mat, pred_poses_mat_m2)
    ate_m3_input = calculate_ate(gt_absolute_poses_mat, pred_poses_mat_m3)

    # Print ATE
    print("\n--- Absolute Trajectory Error (ATE RMSE) ---")
    print(f"  (Calculated on {final_num_frames} common frames)")
    print(f"  - {args.model_1_name:<15} (Input) vs GT: {ate_rci_input:.4f} meters")
    print(f"  - {args.model_2_name:<15} (Input) vs GT: {ate_m2_input:.4f} meters")
    print(f"  - {args.model_3_name:<15} (Input) vs GT: {ate_m3_input:.4f} meters")
    print("=" * 50 + "\n")

    # 4. VISUALIZATION PHASE
    print("Visualizing results (4-way comparison)...")

    # Extract raw (unaligned) XY trajectories
    gt_trajectory_xy = np.array([pose[0:2, 3] for pose in gt_absolute_poses_mat])
    pred_trajectory_xy_rci = np.array([pose[0:2, 3] for pose in pred_poses_mat_rci])
    pred_trajectory_xy_m2 = np.array([pose[0:2, 3] for pose in pred_poses_mat_m2])
    pred_trajectory_xy_m3 = np.array([pose[0:2, 3] for pose in pred_poses_mat_m3])

    if gt_trajectory_xy.shape[0] < 1:
        print("Error: Empty pose lists, cannot plot.")
        exit()

    # --- Common plot setup ---
    save_dir = Path("plots_ablation_test")
    save_dir.mkdir(exist_ok=True, parents=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.titlesize": 18,
        }
    )

    # Color palette
    gt_color = "#1f77b4"  # Tab10 Blue
    pred_rci_color = "#d62728"  # Tab10 Red (Model 1)
    pred_m2_color = "#2ca02c"  # Tab10 Green (Model 2)
    pred_m3_color = "#9467bd"  # Tab10 Purple (Model 3)

    # Labels with ATE
    gt_label = "Ground Truth"
    pred_rci_label = f"{args.model_1_name} (ATE: {ate_rci_input:.3f}m)"
    pred_m2_label = f"{args.model_2_name} (ATE: {ate_m2_input:.3f}m)"
    pred_m3_label = f"{args.model_3_name} (ATE: {ate_m3_input:.3f}m)"

    # Axis labels (based on dummy argument)
    axis_xlabel = "East [m]" if args.use_gps_filter else "X [m]"
    axis_ylabel = "North [m]" if args.use_gps_filter else "Y [m]"

    # Legend helper
    def create_legend(ax, handles_map, ordered_labels, title="Legend", loc="best"):
        handles_for_legend, labels_for_legend = [], []
        for label in ordered_labels:
            if label in handles_map:
                handles_for_legend.append(handles_map[label])
                labels_for_legend.append(label)

        if handles_for_legend:
            ax.legend(
                handles_for_legend,
                labels_for_legend,
                loc=loc,
                title=title,
                fancybox=True,
                framealpha=0.8,
            )

    # --- SINGLE COMPARISON PLOT ---
    print("Generating Comparison Plot...")
    fig_comp, ax_comp = plt.subplots(figsize=(12, 10))

    handles_map_comp = {}

    # Plot GT (Blue, solid)
    (gt_line,) = ax_comp.plot(
        gt_trajectory_xy[:, 0],
        gt_trajectory_xy[:, 1],
        color=gt_color,
        linestyle="-",
        label=gt_label,
        linewidth=2.0,
        alpha=0.8,
        zorder=3,
    )
    handles_map_comp[gt_label] = gt_line

    # Plot Model 2 (Green, dashed)
    (pred_m2_line,) = ax_comp.plot(
        pred_trajectory_xy_m2[:, 0],
        pred_trajectory_xy_m2[:, 1],
        color=pred_m2_color,
        linestyle="--",
        label=pred_m2_label,
        linewidth=2.0,
        alpha=0.9,
        zorder=1,
    )
    handles_map_comp[pred_m2_label] = pred_m2_line

    # Plot Model 3 (Purple, dotted)
    (pred_m3_line,) = ax_comp.plot(
        pred_trajectory_xy_m3[:, 0],
        pred_trajectory_xy_m3[:, 1],
        color=pred_m3_color,
        linestyle=":",
        label=pred_m3_label,
        linewidth=2.0,
        alpha=0.9,
        zorder=2,
    )
    handles_map_comp[pred_m3_label] = pred_m3_line

    # Plot Model 1 (Red, solid)
    (opt_rci_line,) = ax_comp.plot(
        pred_trajectory_xy_rci[:, 0],
        pred_trajectory_xy_rci[:, 1],
        color=pred_rci_color,
        linestyle="-",
        label=pred_rci_label,
        linewidth=2.5,
        alpha=1.0,
        zorder=4,
    )
    handles_map_comp[pred_rci_label] = opt_rci_line

    ax_comp.set_title(f"Trajectories Comparison - {args.sequence_name}")
    ax_comp.set_xlabel(axis_xlabel)
    ax_comp.set_ylabel(axis_ylabel)

    # Legend without loops
    ordered_labels = [gt_label, pred_rci_label, pred_m2_label, pred_m3_label]
    create_legend(
        ax_comp, handles_map_comp, ordered_labels=ordered_labels, title="Legend"
    )

    ax_comp.grid(True, linestyle=":", alpha=0.6)
    ax_comp.axis("equal")
    fig_comp.tight_layout()

    if args.save_plot:
        try:
            # Save with a different filename to avoid overwrites
            save_path_pdf = save_dir / f"{args.sequence_name}_comparison_NO_SLAM.pdf"
            save_path_png = save_dir / f"{args.sequence_name}_comparison_NO_SLAM.png"
            fig_comp.savefig(save_path_pdf, bbox_inches="tight")
            fig_comp.savefig(save_path_png, bbox_inches="tight", dpi=300)
            print(f"  -> Comparison plot (NO SLAM) saved to '{save_dir.name}/'")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        print("\n--save_plot not specified. Showing comparison graph...")
        plt.show()

    plt.close(fig_comp)
    print("\nScript finished.")
