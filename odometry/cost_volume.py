import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG
from pointnet2_ops import pointnet2_utils


class FeatureWeightNet(nn.Module):
    """
    A parallel MLP that learns weights based on feature similarity.
    The input is the features themselves, not coordinates.
    """

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(FeatureWeightNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        # We use Conv2d for efficiency on grouped points (as a point-wise MLP)
        self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
        self.mlp_bns.append(nn.InstanceNorm2d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.InstanceNorm2d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
        self.mlp_bns.append(nn.InstanceNorm2d(out_channel))

    def forward(self, features_concatenated):
        weights = features_concatenated
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = F.relu(bn(conv(weights)))
        return weights


class FeatureExtractor(nn.Module):
    """
    Feature Extractor that is going to use FPS to downsample the point cloud and then
    Multi-Scale Grouping is employed just as stated in pointnet++ paper.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        npoint: int = 2048,
        radii: list = None,
        nsamples: list = None,
        mlps: list = None,
    ):
        super().__init__()

        # Use passed parameters or default values
        radii = radii or [0.05, 0.1, 0.2]  # For more fine-grained data, reduce this
        nsamples = nsamples or [16, 32, 64]
        mlps = mlps or [[3, 32, 32, 64], [3, 64, 64, 128], [3, 64, 96, 128]]

        # Set Abstraction Module with Multi-Scale Grouping (MSG)
        self.sa_module = PointnetSAModuleMSG(
            npoint=npoint,  # Subsampled points from FPS
            radii=radii,  # ball query radius (radii for each scale)
            nsamples=nsamples,  # Max points within the radius
            mlps=mlps,  # MLPs config for each scale
            use_xyz=True,
        )

        # Conv1D is employed to be more efficient during calculations
        self.final_mlp = nn.Sequential(
            nn.Conv1d(320, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, feature_dim, 1),
        )

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward Pass
        """

        # The input tensor `xyz` from slicing might not be contiguous. Fix it here.
        xyz_contiguous = xyz.contiguous()

        # Input features for SA module are just coordinates (x,y,z)
        input_features = xyz_contiguous.transpose(1, 2).contiguous()

        # Apply Pointnet++
        xyz_sampled, features_sampled = self.sa_module(xyz_contiguous, input_features)

        # Final MLP
        features_extracted = self.final_mlp(features_sampled)

        # xyz_sampled data format: (B, npoint, 3)
        # features_extracted data format: (B, feature_dim, npoint)
        return xyz_sampled, features_extracted


class FeatureAggregationLayer(nn.Module):
    """
    Feature aggregator that is going to downsample the GAT enriched points to extract
    the most valuable points in the scene and use them to obtain a more detailed
    cost volume enriched features.
    """

    def __init__(
        self,
        npoint: int,
        in_channel: int,
        radii: list = None,
        nsamples: list = None,
        mlp_channels: list = None,
    ):
        super().__init__()

        # --- Default configs ---
        if radii is None:
            radii = [0.1, 0.2, 0.4]
        if nsamples is None:
            nsamples = [16, 32, 64]
        if mlp_channels is None:
            # Each MLP inputs the features (from GAT) and the coordinates
            mlp_channels = [
                [in_channel, 128, 128],
                [in_channel, 128, 256],
                [in_channel, 128, 256],
            ]

        # ---  PointNet++ MSG (Multi-Scale-Grouping) ---
        self.sa_module = PointnetSAModuleMSG(
            npoint=npoint,
            radii=radii,
            nsamples=nsamples,
            mlps=mlp_channels,
            use_xyz=True,  # adds relative coordinates to the features
        )

        total_out_channels = sum([mlp[-1] for mlp in mlp_channels])

        # --- Final MLP to Refine the Aggregated Features ---
        self.final_mlp = nn.Sequential(
            nn.Conv1d(total_out_channels, 512, 1),
            nn.InstanceNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
        )

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # xyz: (B, N, 3)
        # features: (B, C, N)
        xyz_sampled, features_sampled = self.sa_module(xyz, features)

        final_features = self.final_mlp(features_sampled)

        return xyz_sampled, final_features


def square_distance(src, dst):
    """
    Calculates the squared Euclidean distance matrix between two point clouds.
    src: (B, N, C)
    dst: (B, M, C)
    Returns: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Finds the indices of the 'nsample' nearest points in 'xyz'
    for each query point in 'new_xyz'.
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def index_points_group(points, knn_idx):
    """
    Gathers (groups) points from 'points' tensor using the 'knn_idx'.
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    # input: (B, C, N), (B, N_query, nsample)
    # output: (B, C, N_query, nsample)
    new_points = pointnet2_utils.grouping_operation(
        points_flipped.float(), knn_idx.int()
    ).permute(0, 2, 3, 1)
    return new_points


class WeightNet(nn.Module):
    """MLP that is going to learn the local geometry of the scene"""

    # hidden_unit defines the number of neurons for each hidden layer
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        # Use Conv2d for efficiency (as a point-wise MLP) on grouped points
        self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
        self.mlp_bns.append(nn.InstanceNorm2d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.InstanceNorm2d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
        self.mlp_bns.append(nn.InstanceNorm2d(out_channel))

    def forward(self, localized_xyz):
        # localized_xyz: (B, C, nsample, npoint)
        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = F.relu(bn(conv(weights)))
        return weights  # (B, out_channel, nsample, npoint)


# --- COST VOLUME LAYER ---
class CostVolumeLayerComplete(nn.Module):
    def __init__(self, nsample: int, feat_ch: int, cost_ch: int):
        super().__init__()
        self.nsample = nsample

        # MLP for the Point-to-Patch cost
        self.mlp_cost = nn.Sequential(
            nn.Conv2d(2 * feat_ch + 3, cost_ch, 1),
            nn.InstanceNorm2d(cost_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(cost_ch, cost_ch, 1),
        )

        # WeightNet based on geometry (relative coordinates)
        self.geometry_weightnet = WeightNet(3, cost_ch)

        # Input channel is 2 * feat_ch because we concatenate features from p1 and p2
        self.feature_weightnet = FeatureWeightNet(
            2 * feat_ch, cost_ch
        )  # <-- Weight based on features

        # Trainable parameter 'beta' to balance the two weights
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.weightnet2 = WeightNet(3, cost_ch)
        self.final_bn = nn.InstanceNorm1d(cost_ch)

    def forward(self, xyz1, feat1, xyz2, feat2):
        # xyz1: (B, N1, 3), feat1: (B, D, N1)
        # xyz2: (B, N2, 3), feat2: (B, D, N2)
        B, D, N1 = feat1.shape

        # --- Stage 1: Point-to-Patch Cost with DUAL WEIGHTING ---

        # 1. Find neighbors and calculate relative positions
        knn_idx = knn_point(self.nsample, xyz2, xyz1)  # (B, N1, nsample)
        neighbor_xyz = index_points_group(xyz2, knn_idx)  # (B, N1, nsample, 3)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, 3)  # (B, N1, nsample, 3)

        # 2. Group features
        grouped_feat2 = index_points_group(
            feat2.permute(0, 2, 1), knn_idx
        )  # (B, N1, nsample, D)
        feat1_expanded = (
            feat1.permute(0, 2, 1).view(B, N1, 1, D).expand(-1, -1, self.nsample, -1)
        )  # (B, N1, nsample, D)

        # 3. Calculate raw matching cost
        cost_input = torch.cat(
            [feat1_expanded, grouped_feat2, direction_xyz], dim=-1
        )  # (B, N1, nsample, 2D+3)
        cost_input_permuted = cost_input.permute(0, 3, 2, 1)  # (B, 2D+3, nsample, N1)
        point_to_patch_cost_raw = self.mlp_cost(
            cost_input_permuted
        )  # (B, cost_ch, nsample, N1)

        # 4. Calculate both types of weights in parallel
        # Geometric weights
        geometry_weights = self.geometry_weightnet(
            direction_xyz.permute(0, 3, 2, 1)
        )  # (B, cost_ch, nsample, N1)

        # Feature-based weights
        feature_input = torch.cat([feat1_expanded, grouped_feat2], dim=-1).permute(
            0, 3, 2, 1
        )  # (B, 2D, nsample, N1)
        feature_weights = self.feature_weightnet(
            feature_input
        )  # (B, cost_ch, nsample, N1)

        # 5. Apply weights and combine using beta (as in CAO-RONet)
        # Use sigmoid to keep beta between 0 and 1
        beta_norm = torch.sigmoid(self.beta)

        cost_geom = torch.sum(
            point_to_patch_cost_raw * geometry_weights, dim=2
        )  # (B, cost_ch, N1)
        cost_feat = torch.sum(
            point_to_patch_cost_raw * feature_weights, dim=2
        )  # (B, cost_ch, N1)

        point_to_patch_cost = (
            beta_norm * cost_feat + (1 - beta_norm) * cost_geom
        )  # (B, cost_ch, N1)

        # --- Stage 2: Patch-to-Patch Cost ---
        knn_idx_2 = knn_point(self.nsample, xyz1, xyz1)  # (B, N1, nsample)
        direction_xyz_2 = index_points_group(xyz1, knn_idx_2) - xyz1.view(
            B, N1, 1, 3
        )  # (B, N1, nsample, 3)
        weights2 = self.weightnet2(
            direction_xyz_2.permute(0, 3, 2, 1)
        )  # (B, cost_ch, nsample, N1)

        # Use the new 'point_to_patch_cost' calculated above
        grouped_point_to_patch_cost = index_points_group(
            point_to_patch_cost.permute(0, 2, 1), knn_idx_2
        )  # (B, N1, nsample, cost_ch)

        patch_to_patch_cost = torch.sum(
            weights2 * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim=2
        )  # (B, cost_ch, N1)
        patch_to_patch_cost = self.final_bn(patch_to_patch_cost)

        return patch_to_patch_cost
