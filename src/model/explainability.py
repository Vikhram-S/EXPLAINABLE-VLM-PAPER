import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List

class SupervisedExplainabilityModule(nn.Module):
    """
    Computes Attention-Gradient Fusion Heatmaps (Eq. 14) and Supervised
    Explainability Loss ($L_{exp}$) against MS-CXR / Chest ImaGenome ground-truth bounding box masks.
    """
    def __init__(self, patch_grid_size: int = 14):
        super().__init__()
        self.patch_grid_size = patch_grid_size  # 14x14 = 196 patches

    def compute_fusion_heatmap(
        self,
        cross_attention_weights: torch.Tensor,
        token_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Eq. 14 Attention-Gradient Fusion:
        F_b = ReLU( sum_{t} ( Attn_{b, t, :} * Grad_{b, t} ) )
        cross_attention_weights: [B, num_heads, seq_len, 196] or [B, seq_len, 196]
        token_logits: [B, seq_len, vocab_size]
        Returns:
            heatmaps: [B, 14, 14] normalized to [0, 1]
        """
        if cross_attention_weights.dim() == 4:
            attn = cross_attention_weights.mean(dim=1)  # Average across attention heads -> [B, seq_len, 196]
        else:
            attn = cross_attention_weights

        B, T, P = attn.shape
        # Logit magnitude act as proxy gradient weighting across tokens
        token_weights = F.softmax(token_logits.max(dim=-1).values, dim=-1).unsqueeze(-1) # [B, T, 1]

        # Fusion weighting across text sequence length
        fused_patches = torch.sum(attn * token_weights, dim=1) # [B, 196]
        fused_patches = F.relu(fused_patches)

        # Normalize per sample
        min_v = fused_patches.min(dim=-1, keepdim=True).values
        max_v = fused_patches.max(dim=-1, keepdim=True).values
        fused_patches = (fused_patches - min_v) / (max_v - min_v + 1e-8)

        # Reshape to 14x14 grid
        heatmaps = fused_patches.view(B, self.patch_grid_size, self.patch_grid_size)
        return heatmaps

    def forward(
        self,
        cross_attention_weights: torch.Tensor,
        token_logits: torch.Tensor,
        gt_bbox_masks: torch.Tensor,
        has_bbox_flags: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cross_attention_weights: [B, heads, T, 196]
        token_logits: [B, T, V]
        gt_bbox_masks: [B, 14, 14]
        has_bbox_flags: [B] (1.0 if sample has ground truth MS-CXR box, 0.0 otherwise)
        Returns:
            exp_loss: Supervised MSE/BCE loss on samples with valid bboxes (masked out if none)
            heatmaps: [B, 14, 14]
        """
        heatmaps = self.compute_fusion_heatmap(cross_attention_weights, token_logits)

        # MSE/BCE Loss between heatmap and ground truth box mask
        loss_per_sample = F.mse_loss(heatmaps, gt_bbox_masks, reduction="none").mean(dim=[1, 2]) # [B]

        num_valid = torch.sum(has_bbox_flags)
        if num_valid > 0:
            exp_loss = torch.sum(loss_per_sample * has_bbox_flags) / num_valid
        else:
            exp_loss = torch.tensor(0.0, device=cross_attention_weights.device)

        return exp_loss, heatmaps
