import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
from mamba_ssm import Mamba
from timm.models.layers import DropPath


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
        # Bi-directional implementation for complete context
        forward_pass = self.attn(x)
        backward_pass = self.attn(x.flip(1)).flip(1)

        x = x + self.drop_path(forward_pass) + self.drop_path(backward_pass)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --- NEW MAMBA-BASED ODOMETRY ARCHITECTURE ---


class IMUEncoderMamba(nn.Module):
    """Encodes a single clip of IMU data into a feature vector."""

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
        return x[:, -1, :]


class OdometryMamba(nn.Module):
    """Main model using Mamba for rotation regression."""

    def __init__(self, input_features=6, embed_dim=256, num_layers_temporal=4):
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

    def forward(self, imu_clips):
        # Ensure only 6 features (accel, gyro) are used
        if imu_clips.shape[-1] != 6:
            imu_clips = imu_clips[..., :6]

        b, c, s, f = imu_clips.shape  # Batch, Clip_len, Seq_len, Features

        # Reshape to process all clips in parallel
        imu_reshaped = imu_clips.view(b * c, s, f)
        features = self.encoder(imu_reshaped)

        # Reshape back to [B, C, Embed_Dim]
        features_reshaped = features.view(b, c, -1)

        # Process the sequence of clip features
        temporal_out = features_reshaped
        for layer in self.temporal_mamba:
            temporal_out = layer(temporal_out)

        return temporal_out
