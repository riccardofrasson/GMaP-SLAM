import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import numpy as np
from timm.models.layers import DropPath

# --- BLOCCHI COSTRUTTIVI PER L'OTTIMIZZATORE (BASATI SU MAMBA) ---


class LearnedAggregation(nn.Module):
    """
    Replaces max pooling with a learned, weighted aggregation.

    Input:  cost_volume [B, C, N]
    Output: global_feat [B, C]
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        # A 1D Conv-based MLP (point-wise) to calculate importance scores.
        self.scoring_mlp = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, cost_volume: torch.Tensor) -> torch.Tensor:
        # Calculate raw scores -> [B, 1, N]
        scores = self.scoring_mlp(cost_volume)

        # Normalize scores to weights via softmax -> [B, 1, N]
        weights = F.softmax(scores, dim=2)

        # Apply weighted sum across the N dimension -> [B, C]
        global_feat = torch.sum(cost_volume * weights, dim=2)

        return global_feat


# Define how the mamba block should work (LSTM is worse and more unstable)
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=2, expand=2):
        super().__init__()
        self.dim = dim
        # Apply Layernorm before processing with the mamba block to stabilize the input
        self.norm = nn.LayerNorm(dim)
        # Initialize the mamba block
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        # Mamba takes as inpput x that is a sequence of global vectors (B,seq_lenght,C)
        B, L, C = x.shape
        # Normalize and apply mamba
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)
        return x_mamba


# Simply define 2 MLPs
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# Combine the two classes above generating an elabration block with residual connections
class Block_mamba(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MambaLayer(dim)
        self.mlp = FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # Implementazione bidirezionale come nel paper
        x = (
            x
            + self.drop_path(self.attn(x))
            + self.drop_path(self.attn(x.flip(1))).flip(1)
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --- CLASSE PRINCIPALE DELL'OTTIMIZZATORE ---


class ClipWindowOptimizer(nn.Module):
    def __init__(
        self, cost_volume_channels: int, lstm_hidden: int = 128, layer_num: int = 5
    ):
        super().__init__()

        # Defines a series of values of drop_path from 0 to x
        dpr = [x.item() for x in np.linspace(0, 0.4, layer_num)]

        self.attn = nn.ModuleList(
            [
                Block_mamba(dim=cost_volume_channels, drop_path=dpr[i])
                for i in range(layer_num)
            ]
        )

        self.aggregation_layer = LearnedAggregation(in_channels=cost_volume_channels)

        # MLPs for the pose regression
        final_feat_dim = cost_volume_channels
        self.rotation_mlp = nn.Sequential(
            nn.Linear(final_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # Quaternione
        )
        self.translation_mlp = nn.Sequential(
            nn.Linear(final_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),  # Traslazione
        )

    def forward(self, cost_volume, history=None):

        # 1. Max pooling that is going to provide us a Global Vector
        global_feat = self.aggregation_layer(cost_volume)

        # 2. Memory management of the optimizer if the story is none generate a new history made of only one global vector, otherwise group the new vector with the others
        sequence = (
            global_feat.unsqueeze(1)
            if history is None
            else torch.cat([global_feat.unsqueeze(1), history], dim=1)
        )

        # 3. Processing the global vectors sequence to find a movement pattern
        processed_sequence = sequence
        for layer in self.attn:
            processed_sequence = layer(processed_sequence)

        # 4. Extract the updated state that represents the current frame (our actual frame has been updated with historic context)
        current_updated_feat = processed_sequence[:, 0, :]

        # 5. Pose regression through 2 MLPs
        rotation_q = self.rotation_mlp(current_updated_feat)
        translation_v = self.translation_mlp(current_updated_feat)
        rotation_q = F.normalize(rotation_q, p=2, dim=1)

        return rotation_q, translation_v, processed_sequence
