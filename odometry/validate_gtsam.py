import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import argparse
from pathlib import Path


def plot_relative_poses(csv_path):
    """
    Reads a CSV file with relative poses (dx, dy, dz, qx, qy, qz, qw)
    and plots both the technical components and an intuitive speed visualization.
    """
    if not csv_path.exists():
        print(f"ERROR: The specified file was not found:\n{csv_path}")
        return

    print(f"📊 Loading and analyzing file: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # --- 1. Calculate Additional Metrics ---
    # Calculate Yaw from quaternions
    quaternions = df[["qx", "qy", "qz", "qw"]].values
    rotations = Rotation.from_quat(quaternions)
    euler_angles_rad = rotations.as_euler("zyx")
    df["yaw_deg"] = np.rad2deg(euler_angles_rad[:, 0])
    df["yaw_rad"] = euler_angles_rad[:, 0]

    # Calculate 2D speed magnitude (always positive)
    df["speed"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    print("🚗 Reconstructing 2D trajectory from relative poses...")
    path_x, path_y = [0], [0]
    current_x, current_y, current_theta = 0.0, 0.0, 0.0

    for i, row in df.iterrows():
        # Relative poses (dx, dy) are in the vehicle's local frame.
        # They must be rotated into the global frame before being summed.
        dx_global = row["dx"] * np.cos(current_theta) - row["dy"] * np.sin(
            current_theta
        )
        dy_global = row["dx"] * np.sin(current_theta) + row["dy"] * np.cos(
            current_theta
        )

        current_x += dx_global
        current_y += dy_global
        current_theta += row["yaw_rad"]  # Update the angle

        path_x.append(current_x)
        path_y.append(current_y)

    # Define a "turn" as a frame where the absolute rotation exceeds a threshold.
    TURN_THRESHOLD_DEG = 0.5  # Degrees per frame. You can adjust this value.
    turn_indices = df[abs(df["yaw_deg"]) > TURN_THRESHOLD_DEG].index

    # Extract the trajectory coordinates where turns occur
    turn_x = [path_x[i] for i in turn_indices]
    turn_y = [path_y[i] for i in turn_indices]
    print(
        f"   -> Identified {len(turn_indices)} significant turn sections (yaw > {TURN_THRESHOLD_DEG}°)."
    )

    # --- 2. Create Plots ---
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Complete Relative Pose Analysis", fontsize=16)

    # Plot 1: Intuitive Speed Validation
    axs[0, 0].plot(
        df.index, df["speed"], label="Speed (always positive)", color="green"
    )
    axs[0, 0].set_title("✅ Intuitive Validation: Relative Speed")
    axs[0, 0].set_xlabel("Frame Index")
    axs[0, 0].set_ylabel("Displacement per frame (meters)")
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)
    axs[0, 0].set_ylim(bottom=0)

    # Plot 2: Relative Yaw
    axs[0, 1].plot(df.index, df["yaw_deg"], label="Yaw Variation", color="purple")
    axs[0, 1].set_title("Relative Yaw Trend (Rotation)")
    axs[0, 1].set_xlabel("Frame Index")
    axs[0, 1].set_ylabel("Rotation (degrees)")
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)

    # Plot 3: Technical Analysis of Translation Components
    axs[1, 0].plot(df.index, df["dx"], label="dx (Local X-axis component)", alpha=0.9)
    axs[1, 0].plot(df.index, df["dy"], label="dy (Local Y-axis component)", alpha=0.9)
    axs[1, 0].set_title("Technical Analysis: [dx, dy] Components")
    axs[1, 0].set_xlabel("Frame Index")
    axs[1, 0].set_ylabel("Displacement (meters)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)

    # Plot 4: Detail on dy (Lateral Displacement)
    axs[1, 1].plot(df.index, df["dy"], color="orange", label="dy")
    axs[1, 1].set_title("Lateral Displacement Detail (dy)")
    axs[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[1, 1].set_xlabel("Frame Index")
    axs[1, 1].set_ylabel("Displacement (meters)")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Trajectory Plot ---
    fig_traj, ax_traj = plt.subplots(figsize=(12, 10))
    fig_traj.suptitle("Reconstructed 2D Trajectory with Turns Highlighted", fontsize=16)

    ax_traj.plot(path_x, path_y, label="Trajectory", color="blue", linewidth=2)
    # Highlight turning points with red circles
    ax_traj.scatter(
        turn_x,
        turn_y,
        color="red",
        s=20,
        zorder=5,
        label=f"Turns (Yaw > {TURN_THRESHOLD_DEG}°)",
    )
    # Highlight start and end
    ax_traj.plot(path_x[0], path_y[0], "go", markersize=10, label="Start")
    ax_traj.plot(
        path_x[-1], path_y[-1], "o", color="magenta", markersize=10, label="End"
    )

    ax_traj.set_xlabel("X (meters)")
    ax_traj.set_ylabel("Y (meters)")
    ax_traj.set_title("Reconstructed Path Map")
    ax_traj.grid(True, linestyle="--", alpha=0.7)
    ax_traj.legend()
    # Use 'equal' to ensure correct X/Y proportions
    ax_traj.set_aspect("equal", adjustable="box")
    fig_traj.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot relative poses from a GTSAM-generated CSV file."
    )
    parser.add_argument(
        "csv_file", type=str, help="Path to the 'relative_poses.csv' file."
    )
    args = parser.parse_args()

    plot_relative_poses(Path(args.csv_file))
