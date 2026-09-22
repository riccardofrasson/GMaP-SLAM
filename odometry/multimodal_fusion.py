# Save this file as main_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- SECTION 1: IMPORTS FROM CUSTOM SCRIPTS ---

# 1. Import the Radar-Camera pipeline
from radar_camera_fusion import OdometryPipeline as RadarCameraFeatureExtractor

# 2. Import the fusion module
from raci_fusion import RaciFusionModule

# 3. Import the IMU pipeline
from imu_encoder import OdometryMamba as IMUFeatureExtractor

# 4. Import the Mamba block from the optimizer file
from optimization import Block_mamba


# --- SECTION 2: MAIN ORCHESTRATION MODEL ---


class FusedOdometryModel(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_optimizer_layers=5,
        rc_params=None,
        imu_params=None,
        LSTM_dim=256,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Initialize feature extractors
        if rc_params is None:
            rc_params = {}
        if imu_params is None:
            imu_params = {"embed_dim": embed_dim}

        self.rc_feature_extractor = RadarCameraFeatureExtractor(**rc_params)
        self.imu_feature_extractor = IMUFeatureExtractor(**imu_params)

        # 2. Projection layers to unify feature dimensions
        rc_feat_dim = rc_params.get("cv_params", {}).get("cost_ch", 128)
        imu_feat_dim = imu_params.get("embed_dim", embed_dim)

        self.rc_projection = (
            nn.Linear(rc_feat_dim, embed_dim)
            if rc_feat_dim != embed_dim
            else nn.Identity()
        )
        self.imu_projection = (
            nn.Linear(imu_feat_dim, embed_dim)
            if imu_feat_dim != embed_dim
            else nn.Identity()
        )

        # 3. Initialize the fusion module
        self.fusion_module = RaciFusionModule(feature_dim=embed_dim)

        # 4. FINAL OPTIMIZER (based on ClipWindowOptimizer logic)
        fused_dim = embed_dim * 2
        self.fusion_projection = nn.Linear(fused_dim, LSTM_dim)

        # Use Mamba blocks as defined in optimization.py
        dpr = [x.item() for x in np.linspace(0, 0.4, num_optimizer_layers)]
        self.temporal_optimizer = nn.ModuleList(
            [
                Block_mamba(dim=LSTM_dim, drop_path=dpr[i])
                for i in range(num_optimizer_layers)
            ]
        )

        # Use MLP regressors as defined in ClipWindowOptimizer
        self.rotation_mlp = nn.Sequential(
            nn.Linear(LSTM_dim, LSTM_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(LSTM_dim // 2, 4),  # Quaternion
        )
        self.translation_mlp = nn.Sequential(
            nn.Linear(LSTM_dim, LSTM_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(LSTM_dim // 2, 3),  # Translation
        )

    def forward(self, imu_clips, rc_batch, history=None):
        # --- STEP 1: Feature Extraction ---
        rc_feat_raw = self.rc_feature_extractor(rc_batch)
        imu_feat_sequence = self.imu_feature_extractor(imu_clips)

        # --- STEP 2: Projection and Fusion ---
        num_clips = imu_feat_sequence.shape[1]

        rc_feat = self.rc_projection(rc_feat_raw)
        imu_feat_seq_proj = self.imu_projection(imu_feat_sequence)

        # Expand radar-camera features to match the IMU sequence length
        rc_feat_expanded = rc_feat.unsqueeze(1).expand(-1, num_clips, -1)

        fused_sequence = self.fusion_module(imu_feat_seq_proj, rc_feat_expanded)
        fused_sequence_proj = self.fusion_projection(fused_sequence)

        # --- STEP 3: Temporal Optimization ---

        # 1. Build the raw input buffer
        current_raw_buffer = (
            fused_sequence_proj
            if history is None
            else torch.cat([fused_sequence_proj, history], dim=1)
        )

        # 2. Process the entire buffer from scratch
        processed_sequence = current_raw_buffer
        for layer in self.temporal_optimizer:
            processed_sequence = layer(processed_sequence)

        # 3. Extract the output
        current_updated_feat = processed_sequence[:, num_clips - 1, :]
        new_history = current_raw_buffer

        # --- STEP 4: Final Pose Regression ---
        rotation_q = self.rotation_mlp(current_updated_feat)
        translation_v = self.translation_mlp(current_updated_feat)

        # Normalize the quaternion
        rotation_q = F.normalize(rotation_q, p=2, dim=1, eps=1e-8)

        return translation_v, rotation_q, new_history


# --- SECTION 3: USAGE EXAMPLE ---
if __name__ == "__main__":
    print("--- Fused Model Instantiation Example (with ClipWindowOptimizer logic) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters for sub-modules
    rc_params = {"n_sample_points": 1024, "cv_params": {"cost_ch": 128}}
    imu_params = {"input_features": 6, "embed_dim": 256, "num_layers_temporal": 4}

    try:
        model = FusedOdometryModel(
            embed_dim=256,
            num_optimizer_layers=5,
            rc_params=rc_params,
            imu_params=imu_params,
        ).to(device)
        print("FusedOdometryModel created successfully.")

    except Exception as e:
        print(f"\nERROR during model creation: {e}")
