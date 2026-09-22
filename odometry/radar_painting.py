import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms


def get_feature_extractor_and_preprocessor(n_canceled_layers):
    """
    Loads a truncated EfficientNet-V2 model to use as a feature extractor
    and returns the corresponding preprocessing pipeline.
    """
    print(f"Loading EfficientNet-V2 and removing {n_canceled_layers} final layers...")

    # Load pre-trained weights from torchvision
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)

    # Define standard normalization constants (from ImageNet)
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    target_size = (384, 384)  # Expected input size for this model

    # Define the image preprocessing pipeline.
    preprocess = transforms.Compose(
        [
            # Anamorphic resize (distorts aspect ratio) to the model's expected input size.
            transforms.Resize(target_size, antialias=True),
            transforms.ToTensor(),
            # Normalize using the standard ImageNet statistics
            transforms.Normalize(mean=image_mean, std=image_std),
        ]
    )

    print("Preprocessing pipeline defined:")
    print(preprocess)

    # Get all layers from the 'features' part of the model
    model_layers = list(model.features.children())

    # Select all layers except for the last 'n_canceled_layers'
    selected_layers = model_layers[:-n_canceled_layers]

    # Create the truncated model (feature extractor)
    feature_extractor = torch.nn.Sequential(*selected_layers)
    # Set model to evaluation mode (disables dropout, batch norm updates)
    feature_extractor.eval()

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor.to(device)
    print(f"Feature extractor will use: {device}")

    return feature_extractor, preprocess, device


def read_calibration_files(filepath):
    """
    Reads camera intrinsic (K) and radar-to-camera extrinsic (T)
    matrices from hardcoded calibration file paths.
    """
    file_calib_camera = os.path.join(filepath, "CALIBRATION_CAMERA.txt")
    file_calib_radar = os.path.join(filepath, "CALIBRATION_CAMERA_RADAR.txt")

    try:
        # Read camera intrinsic matrix (K)
        with open(file_calib_camera, "r") as f:
            lines = f.readlines()
        k_matrix = []
        # K matrix is on lines 5, 6, 7 (0-indexed)
        for i in range(5, 8):
            row = [float(num) for num in lines[i].strip().split()]
            k_matrix.append(row)
        K = np.array(k_matrix)

        # Read radar-to-camera extrinsic (R, t)
        with open(file_calib_radar, "r") as f:
            lines = f.readlines()

        r_matrix = []
        # R matrix is on lines 4, 5, 6
        for i in range(4, 7):
            row = [float(num) for num in lines[i].strip().split()]
            r_matrix.append(row)
        R = np.array(r_matrix)

        # t vector is on line 10
        t_line = lines[10].strip().split()
        t = np.array([float(num) for num in t_line]).reshape(3, 1)

        # Assemble the 4x4 rigid transformation matrix T_radar_to_cam
        T_radar_to_cam = np.eye(4)
        T_radar_to_cam[:3, :3] = R
        T_radar_to_cam[:3, 3] = t.flatten()

        print("✅ Calibration files read successfully.")
        return K, T_radar_to_cam

    except (FileNotFoundError, IndexError, ValueError) as e:
        print(f"Error: Could not read calibration files.\n{e}")
        return None, None


def point_painting_batch(
    radar_points,
    feature_maps,
    K,
    T_radar_to_cam,
    original_img_size,
    feature_map_size,
    patch_size=3,
):
    """
    Performs PointPainting fusion in a vectorized, batch-wise manner.

    Projects radar points onto the image feature map and "paints" (concatenates)
    the corresponding image features onto the points.
    """
    # Get dimensions
    B, N, _ = radar_points.shape  # (B)atch, (N)umber of points
    C, H_feat, W_feat = feature_maps.shape[1], feature_map_size[0], feature_map_size[1]
    H_img, W_img = original_img_size
    device = radar_points.device

    # --- 1. Point Projection ---

    # Extract (x, y, z) coordinates and convert to homogeneous (B, N, 4)
    rad_coords = radar_points[:, :, :3]
    ones = torch.ones(B, N, 1, device=device)
    rad_points_hom = torch.cat([rad_coords, ones], dim=2)

    # Transform points from radar to camera coordinates (B, N, 4)
    points_cam = torch.bmm(rad_points_hom, T_radar_to_cam.transpose(1, 2))

    # Project 3D camera coordinates to 2D image plane (B, N, 3)
    points_cam_xyz = points_cam[:, :, :3]
    points_img_proj = torch.bmm(points_cam_xyz, K.transpose(1, 2))

    # Perform perspective division (z-normalization) to get (u, v)
    # Add epsilon for numerical stability
    u = points_img_proj[:, :, 0] / (points_img_proj[:, :, 2] + 1e-8)
    v = points_img_proj[:, :, 1] / (points_img_proj[:, :, 2] + 1e-8)

    # Create a mask for points that fall within the original image boundaries
    valid_mask = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)

    # --- 2. Point Painting ---

    # Scale (u, v) coordinates from image space to feature map space
    u_feat = (u * (W_feat / W_img)).round().long()
    v_feat = (v * (H_feat / H_img)).round().long()

    # Clamp coordinates to be within the feature map dimensions
    u_feat = torch.clamp(u_feat, 0, W_feat - 1)
    v_feat = torch.clamp(v_feat, 0, H_feat - 1)

    half_patch = patch_size // 2
    # Use 'unfold' to create a view of all possible (patch_size x patch_size) patches
    # Shape: (B, C * patch_size * patch_size, H_feat * W_feat)
    unfolded_features = F.unfold(
        feature_maps, kernel_size=patch_size, padding=half_patch
    )

    # Create batch indices corresponding to each point (B, N)
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)

    # Get the flat (linear) indices for the feature map
    linear_indices = v_feat * W_feat + u_feat

    # Filter indices and batch indices using the valid_mask
    valid_batch_indices = batch_indices[valid_mask]
    valid_linear_indices = linear_indices[valid_mask]

    # Select the patches corresponding to valid points
    # Permute unfolded_features to (B, H*W, C*k*k) for easier indexing
    patches_to_pool = unfolded_features.permute(0, 2, 1)[
        valid_batch_indices, valid_linear_indices
    ]

    # Reshape patches to (Num_Valid_Points, C, patch_size * patch_size)
    patches_reshaped = patches_to_pool.view(
        patches_to_pool.shape[0], C, patch_size * patch_size
    )

    # Apply max pooling across the spatial dimensions (k*k) of each patch
    # Shape: (Num_Valid_Points, C)
    pooled_features, _ = torch.max(patches_reshaped, dim=2)

    # --- 3. Feature Concatenation ---

    # Initialize an empty tensor for the "painted" features
    painted_features = torch.zeros(B, N, C, device=device, dtype=radar_points.dtype)
    # Fill the tensor with pooled features at the valid indices
    painted_features[valid_mask] = pooled_features.to(painted_features.dtype)

    # Concatenate original radar features (e.g., xyz, velocity) with new image features
    return torch.cat([radar_points, painted_features], dim=2)
