# Save this code as radar_camera_encoder.py
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

# --- Pipeline Module Imports ---
try:
    from radar_painting import (
        get_feature_extractor_and_preprocessor,
        point_painting_batch,
    )
    from graph_attention import GATOdometryNet
    from cost_volume import CostVolumeLayerComplete
    from optimization import ClipWindowOptimizer
    from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    raise

import torch.nn.functional as F


class LearnedAggregation(nn.Module):
    """
    Replaces max pooling with a learned, weighted aggregation.
    Input:  cost_volume [B, C, N]
    Output: global_feat [B, C]
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        # Two-stage MLP to calculate importance scores
        self.scoring_mlp = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, cost_volume: torch.Tensor) -> torch.Tensor:
        scores = self.scoring_mlp(cost_volume)
        weights = F.softmax(scores, dim=2)
        global_feat = torch.sum(cost_volume * weights, dim=2)
        return global_feat


def transform_point_cloud_batch(pc_batch, T_batch):
    """
    Transforms a batch of 3D point clouds (and their features)
    using a batch of 4x4 transformation matrices.
    """
    # Take only xyz coordinates for transformation
    coords_only = pc_batch[..., :3]

    # Convert points to homogeneous coordinates [x, y, z, 1]
    pc_homogeneous = F.pad(coords_only, (0, 1), mode="constant", value=1.0)

    # Apply transformation: T @ p.T, then transpose back
    transformed_coords = torch.bmm(T_batch, pc_homogeneous.transpose(1, 2)).transpose(
        1, 2
    )

    # Reconstruct the full tensor with new coordinates and old features
    transformed_pc = torch.cat([transformed_coords[..., :3], pc_batch[..., 3:]], dim=-1)

    return transformed_pc


class OdometryPipeline(nn.Module):
    """
    OPTIMIZED Radar-Camera Pipeline.
    Performs FPS sampling BEFORE the GAT for maximum efficiency.
    """

    def __init__(
        self,
        n_sample_points=1024,
        n_canceled_layers=2,
        patch_size=3,
        gat_params=None,
        cv_params=None,
        optimizer_params=None,
    ):
        super().__init__()
        self.n_sample_points = n_sample_points
        self.patch_size = patch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"OPTIMIZED Odometry Pipeline created. Device: {self.device}")

        self.feature_extractor, _, _ = get_feature_extractor_and_preprocessor(
            n_canceled_layers
        )

        # Get the feature dimension from the CNN extractor
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 384, 384).to(self.device)
            dummy_output = self.feature_extractor(dummy_input)
            feature_dim_cam = dummy_output.shape[1]

        if gat_params is None:
            gat_params = {}
        self.gat_model = GATOdometryNet(
            feature_dim_cam=feature_dim_cam, **gat_params
        ).to(self.device)

        gat_output_dim = 128 * 3  # Output dimension from GATOdometryNet

        if cv_params is None:
            cv_params = {"nsample": 16, "cost_ch": 128}
        self.cost_volume_layer = CostVolumeLayerComplete(
            nsample=cv_params["nsample"],
            feat_ch=gat_output_dim,
            cost_ch=cv_params["cost_ch"],
        ).to(self.device)

        self.aggregation_layer = LearnedAggregation(in_channels=cv_params["cost_ch"])

    def _unpad_and_separate(self, painted_pc_padded, lengths):
        """Unpads a padded tensor and separates it into PC data, Cam features, and batch vector."""
        B, max_len, _ = painted_pc_padded.shape
        mask = torch.arange(max_len, device=self.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        pc_painted_unpadded = painted_pc_padded[mask]
        batch_indices = torch.arange(B, device=self.device).unsqueeze(1).expand_as(mask)
        batch_vector = batch_indices[mask]
        pc_unpadded = pc_painted_unpadded[:, :5]
        cam_feat_unpadded = pc_painted_unpadded[:, 5:]
        return pc_unpadded, cam_feat_unpadded, batch_vector

    def _safe_farthest_point_sample(self, xyz_padded, mask):
        """
        Performs FPS on padded batches by replacing padded points
        with the first valid point to avoid sampling NaNs or zeros.
        """
        B, N_max, _ = xyz_padded.shape
        xyz_for_fps = xyz_padded.clone()
        first_points = xyz_for_fps[:, 0, :].unsqueeze(1)
        first_points_expanded = first_points.expand(-1, N_max, -1)
        xyz_for_fps[~mask] = first_points_expanded[~mask]
        return furthest_point_sample(xyz_for_fps, self.n_sample_points)

    def forward(self, batch, history=None):
        radar_t0_padded = batch["radar_t0"].to(self.device)
        radar_t1_padded = batch["radar_t1"].to(self.device)
        images_t0 = batch["image_t0"].to(self.device)
        images_t1 = batch["image_t1"].to(self.device)
        K = batch["K"].to(self.device)
        T_radar_to_cam = batch["T_radar_to_cam"].to(self.device)
        lengths_t0 = batch["lengths_t0"]["radar"].to(self.device)
        lengths_t1 = batch["lengths_t1"]["radar"].to(self.device)

        # --- MOD 1: Load the final transformation matrix from the batch ---
        T_imu_from_radar = batch["T_imu_from_radar"].to(self.device)

        with torch.no_grad():
            feature_maps_t0 = self.feature_extractor(images_t0)
            feature_maps_t1 = self.feature_extractor(images_t1)

        feature_map_size = (feature_maps_t0.shape[2], feature_maps_t0.shape[3])
        original_img_size = (384, 384)

        # Perform point painting. Output coordinates are still in the RADAR frame
        painted_pc_t0_radar = point_painting_batch(
            radar_t0_padded,
            feature_maps_t0,
            K,
            T_radar_to_cam,
            original_img_size,
            feature_map_size,
            patch_size=self.patch_size,
        )
        painted_pc_t1_radar = point_painting_batch(
            radar_t1_padded,
            feature_maps_t1,
            K,
            T_radar_to_cam,
            original_img_size,
            feature_map_size,
            patch_size=self.patch_size,
        )

        # ---  Transform point clouds into the IMU frame ---
        painted_pc_t0_imu = transform_point_cloud_batch(
            painted_pc_t0_radar, T_imu_from_radar
        )
        painted_pc_t1_imu = transform_point_cloud_batch(
            painted_pc_t1_radar, T_imu_from_radar
        )

        # ---  Use the new '_imu' tensors for all subsequent steps ---
        mask_t0 = (
            torch.arange(painted_pc_t0_imu.shape[1], device=self.device)[None, :]
            < lengths_t0[:, None]
        )
        mask_t1 = (
            torch.arange(painted_pc_t1_imu.shape[1], device=self.device)[None, :]
            < lengths_t1[:, None]
        )

        # Perform FPS on the IMU-frame coordinates
        sampled_indices_t0 = self._safe_farthest_point_sample(
            painted_pc_t0_imu[..., :3], mask_t0
        )
        sampled_indices_t1 = self._safe_farthest_point_sample(
            painted_pc_t1_imu[..., :3], mask_t1
        )

        painted_pc_t0_imu_transposed = painted_pc_t0_imu.transpose(1, 2).contiguous()
        painted_pc_t1_imu_transposed = painted_pc_t1_imu.transpose(1, 2).contiguous()

        # Gather the sampled points (coordinates + all features)
        sampled_painted_pc_t0_padded = gather_operation(
            painted_pc_t0_imu_transposed, sampled_indices_t0
        ).transpose(1, 2)
        sampled_painted_pc_t1_padded = gather_operation(
            painted_pc_t1_imu_transposed, sampled_indices_t1
        ).transpose(1, 2)

        # All batches now have exactly n_sample_points
        new_lengths = torch.full(
            (radar_t0_padded.shape[0],), self.n_sample_points, device=self.device
        )

        # Unpad and separate into PC (xyz, vr, rcs), Cam Features, and Batch Vector
        pc_t, cam_feat_t, batch_t = self._unpad_and_separate(
            sampled_painted_pc_t0_padded, new_lengths
        )
        pc_t1, cam_feat_t1, batch_t1 = self._unpad_and_separate(
            sampled_painted_pc_t1_padded, new_lengths
        )

        # Process through GAT
        enriched_features, _ = self.gat_model(
            pc_t=pc_t,
            pc_t_plus_1=pc_t1,
            cam_feat_t=cam_feat_t,
            cam_feat_t_plus_1=cam_feat_t1,
            batch_t=batch_t,
            batch_t_plus_1=batch_t1,
        )

        # Separate GAT features back into t0 and t1
        n_points_t_sampled = pc_t.shape[0]
        enriched_feat_t = enriched_features[:n_points_t_sampled]
        enriched_feat_t1 = enriched_features[n_points_t_sampled:]

        # Convert back to padded batch format for Cost Volume
        coords_t_sampled_padded, _ = to_dense_batch(pc_t[:, :3], batch_t)
        feat_t_enriched_padded, _ = to_dense_batch(enriched_feat_t, batch_t)

        coords_t1_sampled_padded, _ = to_dense_batch(pc_t1[:, :3], batch_t1)
        feat_t1_enriched_padded, _ = to_dense_batch(enriched_feat_t1, batch_t1)

        # Calculate Cost Volume
        cost_volume_features = self.cost_volume_layer(
            xyz1=coords_t_sampled_padded,
            feat1=feat_t_enriched_padded.transpose(1, 2),
            xyz2=coords_t1_sampled_padded,
            feat2=feat_t1_enriched_padded.transpose(1, 2),
        )

        # Aggregate Cost Volume into a global feature vector
        global_feat = self.aggregation_layer(cost_volume_features)

        return global_feat
