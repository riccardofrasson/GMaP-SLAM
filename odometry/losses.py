import torch
import torch.nn as nn
import torch.nn.functional as F


def quaternion_angular_error_rad(pred_q, gt_q):
    """
    Calculates the angular error in RADIANS between two quaternions.
    This function is suitable for evaluation/metrics, not for training loss.
    """
    # Normalize quaternions to ensure they are unit quaternions
    pred_q_norm = F.normalize(pred_q, p=2, dim=-1)
    gt_q_norm = F.normalize(gt_q, p=2, dim=-1)

    # Calculate the dot product. Use absolute value to handle the double cover (q and -q)
    dot_product = torch.abs(torch.sum(pred_q_norm * gt_q_norm, dim=-1))

    # Clamp for numerical stability of arccosine
    dot_product_clamped = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)

    # Calculate the angle (2 * acos(|q1 · q2|))
    angle_rad = 2 * torch.acos(dot_product_clamped)

    return angle_rad


class AdaptiveLoss(nn.Module):
    """
    Implements an adaptive loss for jointly learning rotation (quaternion)
    and translation.

    This weights the two loss components (rotation and translation) by
    learning their associated uncertainties (log-variances), as proposed in:
    "Multi-Task Learning Using Uncertainty to Weigh Losses..." (Kendall, 2018)
    """

    def __init__(self, w_q_initial=-2.5, w_t_initial=0.0):
        super().__init__()
        # These parameters are the learned log-variances for each task
        self.w_q = nn.Parameter(torch.tensor(w_q_initial))  # Log-variance for rotation
        self.w_t = nn.Parameter(
            torch.tensor(w_t_initial)
        )  # Log-variance for translation

    def forward(self, pred_q, pred_t, gt_q, gt_t):

        # --- 1. Rotation Loss (using a stable proxy) ---
        pred_q_norm = F.normalize(pred_q, p=2, dim=-1)
        gt_q_norm = F.normalize(gt_q, p=2, dim=-1)
        dot_product = torch.sum(pred_q_norm * gt_q_norm, dim=-1)

        # Calculate the mean rotation loss over the batch (scalar)
        loss_q = (1.0 - torch.abs(dot_product)).mean()

        # --- 2. Translation Loss ---
        loss_t = F.smooth_l1_loss(pred_t, gt_t)

        # --- 3. Combine losses with adaptive weights ---
        total_loss = (
            loss_q * torch.exp(-self.w_q)
            + self.w_q
            + loss_t * torch.exp(-self.w_t)
            + self.w_t
        )

        return total_loss


def quaternion_angular_error(pred_q, gt_q):
    """
    Helper function to calculate angular error in DEGREES for logging/metrics.
    """
    angle_rad = quaternion_angular_error_rad(pred_q, gt_q)
    angle_deg = angle_rad * (180.0 / torch.pi)
    return angle_deg
