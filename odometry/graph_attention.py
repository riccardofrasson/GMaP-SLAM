import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, InstanceNorm
from torch_geometric.utils import to_undirected
from torch_cluster import knn_graph, radius, knn


# --- Helper MLP Module ---
# A simple, reusable Multi-Layer Perceptron block.
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim * 2]  # Default hidden layer

        layers = []
        current_dim = input_dim
        # Generate a hidden layer for each element in hidden_dims
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# --- Main GAT Odometry Network ---
class GATOdometryNet(nn.Module):
    """ """

    def __init__(
        self,
        feature_dim_cam,
        k_neighbors=8,
        motion_radius=1.0,
        feature_k=4,
        dropout=0.5,
        radar_mean=None,
        radar_std=None,
    ):
        super().__init__()

        # Store graph construction parameters
        self.k_neighbors = k_neighbors
        self.motion_radius = motion_radius
        self.feature_k = feature_k
        self.dropout = dropout

        if radar_mean is not None and radar_std is not None:
            # Register as buffers to automatically move them to the correct device (CPU/GPU)
            self.register_buffer(
                "radar_mean", torch.tensor(radar_mean, dtype=torch.float32)
            )
            self.register_buffer(
                "radar_std", torch.tensor(radar_std, dtype=torch.float32)
            )
        else:
            # Default values if not provided, to prevent crashes
            self.register_buffer("radar_mean", torch.zeros(5))
            self.register_buffer("radar_std", torch.ones(5))

        # --- Feature Encoders ---
        # MLP for spatial coordinates (x, y, z)
        self.mlp_pos = MLP(input_dim=3, output_dim=32)
        # MLP for dynamic features (Doppler velocity, RCS)
        self.mlp_dyn = MLP(input_dim=2, output_dim=16)

        # Total dimension of node features after encoding and concatenation
        node_feature_dim = 32 + 16 + feature_dim_cam

        # Initial normalization layer before the GAT
        self.initial_norm = nn.LayerNorm(node_feature_dim)

        # --- GAT Layers ---
        # The architecture uses multi-head attention to capture diverse relationships.
        self.gat_layer1 = GATv2Conv(node_feature_dim, 64, heads=4, dropout=dropout)
        self.bn1 = InstanceNorm(64 * 4)  # Normalization after first GAT layer

        self.gat_layer2 = GATv2Conv(64 * 4, 128, heads=3, dropout=dropout)
        self.bn2 = InstanceNorm(128 * 3)  # Normalization after second GAT layer

        # The output of this network will be enriched node features.
        # A "head" network would be needed to regress the final odometry (pose).

    def _unpad_and_format(self, padded_pc, padded_cam_feat, lengths):
        """
        Converts padded tensors from the DataLoader to the "long" format
        required by PyTorch Geometric using vectorized operations.
        """
        device = padded_pc.device
        B, max_len, _ = padded_pc.shape

        # 1. Create a boolean mask for real points
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        # 2. Apply the mask to get unpadded ("long") tensors
        pc_unpadded = padded_pc[mask]
        cam_feat_unpadded = padded_cam_feat[mask]

        # 3. Create the batch vector for PyG
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand_as(mask)
        batch_vector = batch_indices[mask]

        return pc_unpadded, cam_feat_unpadded, batch_vector

    def build_graph(
        self,
        coords_t,
        coords_t_plus_1,
        pc_feat_t,
        cam_feat_t,
        cam_feat_t_plus_1,
        batch_t,
        batch_t_plus_1,
    ):
        """
        Builds a complex graph connecting nodes within and between two frames using
        fully vectorized operations for GPU acceleration.
        """
        edge_index_list = []
        n_points_t = coords_t.shape[0]
        num_nodes = n_points_t + coords_t_plus_1.shape[0]

        # --- 1. INTRA-FRAME EDGES (t) ---
        # Connects spatially close points within the first frame, for the whole batch.
        edge_index_t = knn_graph(coords_t, k=self.k_neighbors, batch=batch_t)
        edge_index_list.append(edge_index_t)

        # --- 2. INTRA-FRAME EDGES (t+1) ---
        # Connects spatially close points within the second frame, for the whole batch.
        edge_index_t_plus_1 = knn_graph(
            coords_t_plus_1, k=self.k_neighbors, batch=batch_t_plus_1
        )
        # Offset indices to place them correctly in the combined graph
        edge_index_t_plus_1 += n_points_t
        edge_index_list.append(edge_index_t_plus_1)

        # --- 3. MOVEMENT EDGES (t -> t+1) ---
        # Predicts position at t+1 using Doppler and connects to nearby points.

        coords_t_raw = coords_t * self.radar_std[:3] + self.radar_mean[:3]
        v_doppler_normalized = pc_feat_t[
            :, 0
        ]  # vr (radial velocity) is the first feature of pc_feat_t
        v_doppler_raw = v_doppler_normalized * self.radar_std[3] + self.radar_mean[3]

        dt = 1.0 / 15.0  # Time delta (example: 15 Hz)
        point_radii = torch.norm(coords_t_raw, p=2, dim=1)
        unit_vectors = coords_t_raw / (point_radii.unsqueeze(1) + 1e-8)
        velocity_cartesian = v_doppler_raw.unsqueeze(1) * unit_vectors
        predicted_pos_raw = coords_t_raw + velocity_cartesian * dt

        # Denormalize
        coords_t_plus_1_raw = coords_t_plus_1 * self.radar_std[:3] + self.radar_mean[:3]

        # Find connections within a radius between the predicted points and the next frame.
        edge_index_motion = radius(
            x=coords_t_plus_1_raw,
            y=predicted_pos_raw,
            r=self.motion_radius,
            batch_x=batch_t_plus_1,
            batch_y=batch_t,
            max_num_neighbors=4,
        )

        # Offset destination indices
        edge_index_motion[1] += n_points_t
        edge_index_list.append(edge_index_motion)

        # --- 4. SIMILARITY EDGES (t -> t+1) ---
        # For greater stability, we search for neighbors in t for each point in t+1.
        # This creates edges t+1 -> t.
        edge_index_sim = knn(
            x=cam_feat_t,
            y=cam_feat_t_plus_1,
            k=self.feature_k,
            batch_x=batch_t,
            batch_y=batch_t_plus_1,
        )

        # The output is [col, row] for y -> x, so [indices_t, indices_t+1].
        # These are edges t+1 -> t.

        # We want edges t -> t+1, so we swap source and destination.
        edge_index_sim = knn(
            x=cam_feat_t,
            y=cam_feat_t_plus_1,
            k=self.feature_k,
            batch_x=batch_t,
            batch_y=batch_t_plus_1,
        )

        if edge_index_sim.numel() > 0:
            # The output of knn(x=t, y=t+1) is [row, col] where the edge is y->x (t+1 -> t).
            # So 'row' are indices from t+1, 'col' are indices from t.
            # To create an edge t -> t+1, the source is 'col' and the destination is 'row'.
            source = edge_index_sim[1]  # Indices from t, our source
            dest = edge_index_sim[0]  # Indices from t+1, our destination

            # Apply offset only to the destination indices
            dest_offset = dest + n_points_t

            final_edge_sim = torch.stack([source, dest_offset], dim=0)
            edge_index_list.append(final_edge_sim)

        if not edge_index_list:
            # If there are no edges, return an empty tensor
            edge_index = torch.empty((2, 0), dtype=torch.long, device=coords_t.device)
        else:
            # Otherwise, concatenate all edges
            edge_index = torch.cat(edge_index_list, dim=1)

        return to_undirected(edge_index, num_nodes=num_nodes)

    def forward(
        self, pc_t, pc_t_plus_1, cam_feat_t, cam_feat_t_plus_1, batch_t, batch_t_plus_1
    ):
        """
        The forward pass of the network.

        Args:
            pc_t (Tensor): Point cloud at time t [N_t, 5] (x,y,z,vr,rcs).
            pc_t_plus_1 (Tensor): Point cloud at time t+1 [N_t+1, 5].
            cam_feat_t (Tensor): Camera features for pc_t [N_t, C_cam].
            cam_feat_t_plus_1 (Tensor): Camera features for pc_t+1 [N_t+1, C_cam].
            batch_t (Tensor): Batch index for each point in pc_t.
            batch_t_plus_1 (Tensor): Batch index for each point in pc_t_plus_1.

        Returns:
            Tensor: Enriched node features after GAT layers [N_total, 128*4].
            Tensor: The batch vector for the combined graph.
        """

        # --- 1. Prepare Node Features ---
        # Combine data from both timestamps into single tensors for feature processing
        all_coords = torch.cat([pc_t[:, :3], pc_t_plus_1[:, :3]], dim=0)
        all_dyn_feat = torch.cat([pc_t[:, 3:5], pc_t_plus_1[:, 3:5]], dim=0)
        all_cam_feat = torch.cat([cam_feat_t, cam_feat_t_plus_1], dim=0)

        # Create a single batch tensor for the combined graph. This is needed for pooling later.
        all_batch = torch.cat([batch_t, batch_t_plus_1], dim=0)

        # Encode raw features using MLPs
        x_pos = self.mlp_pos(all_coords)
        x_dyn = self.mlp_dyn(all_dyn_feat)

        # Concatenate all features to create the initial node representation
        x = torch.cat([x_pos, x_dyn, all_cam_feat], dim=1)
        x = self.initial_norm(x)

        # --- 2. Build Graph ---
        # Call the graph builder with the correct, separate tensors for each frame
        edge_index = self.build_graph(
            coords_t=pc_t[:, :3],
            coords_t_plus_1=pc_t_plus_1[:, :3],
            pc_feat_t=pc_t[:, 3:5],
            cam_feat_t=cam_feat_t,
            cam_feat_t_plus_1=cam_feat_t_plus_1,
            batch_t=batch_t,
            batch_t_plus_1=batch_t_plus_1,
        )

        # --- 3. GAT Propagation ---
        # Apply first GAT layer, followed by normalization, activation, and dropout
        x = self.gat_layer1(x, edge_index)
        x = self.bn1(x, all_batch)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply second GAT layer
        x = self.gat_layer2(x, edge_index)
        x = self.bn2(x, all_batch)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x, all_batch
