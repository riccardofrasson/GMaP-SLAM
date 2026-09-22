import torch
import torch.nn as nn
import torch.nn.functional as F


class RaciFusionModule(nn.Module):
    """
    Implements the 2-stage adaptive fusion mechanism (Self-Attention and Cross-Attention)
    described in the Raci-Net paper, adapted for two input modalities (IMU and Radar-Camera).
    """

    def __init__(self, feature_dim: int, hidden_dim_ratio: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        hidden_dim = feature_dim // hidden_dim_ratio

        # MLPs for the Self-Attention stage
        self.self_attn_imu = self._create_attention_mlp(feature_dim, hidden_dim)
        self.self_attn_rc = self._create_attention_mlp(feature_dim, hidden_dim)

        # MLPs for the Cross-Attention stage
        self.cross_attn_imu = self._create_attention_mlp(feature_dim, hidden_dim)
        self.cross_attn_rc = self._create_attention_mlp(feature_dim, hidden_dim)

    def _create_attention_mlp(self, in_dim: int, hid_dim: int):
        """Helper to create the attention mask generator MLP as per the paper."""
        return nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, imu_feat: torch.Tensor, rc_feat: torch.Tensor):
        # --- STAGE 1: SELF-ATTENTION ---
        # Generate self-attention masks
        a_imu_self = self.self_attn_imu(imu_feat)
        a_rc_self = self.self_attn_rc(rc_feat)

        # Apply self-attention
        imu_feat_tilde = a_imu_self * imu_feat
        rc_feat_tilde = a_rc_self * rc_feat

        # --- STAGE 2: CROSS-ATTENTION ---
        # Generate cross-attention masks
        a_rc_to_imu = self.cross_attn_imu(
            rc_feat_tilde
        )  # Mask for IMU, generated from RC
        a_imu_to_rc = self.cross_attn_rc(
            imu_feat_tilde
        )  # Mask for RC, generated from IMU

        # Apply cross-attention
        imu_feat_bar = a_rc_to_imu * imu_feat_tilde
        rc_feat_bar = a_imu_to_rc * rc_feat_tilde

        # --- FINAL FUSION ---
        # Concatenate the refined features
        fused_vector = torch.cat([imu_feat_bar, rc_feat_bar], dim=-1)
        return fused_vector
