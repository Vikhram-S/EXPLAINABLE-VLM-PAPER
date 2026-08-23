import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAlignmentLoss(nn.Module):
    """
    InfoNCE Cross-Modal Contrastive Alignment Loss ($L_{align}$).
    Aligns global image embeddings with report text representations in a shared contrastive space.
    """
    def __init__(self, temperature: float = 0.07, embed_dim: int = 768):
        super().__init__()
        self.temperature = temperature
        self.img_proj = nn.Linear(embed_dim, 256)
        self.txt_proj = nn.Linear(embed_dim, 256)

    def forward(self, global_image_embeds: torch.Tensor, report_token_embeds: torch.Tensor) -> torch.Tensor:
        """
        global_image_embeds: [B, 768]
        report_token_embeds: [B, T, 768] or [B, 768]
        """
        if report_token_embeds.dim() == 3:
            # Pool token embeddings over length dimension
            txt_embeds = report_token_embeds.mean(dim=1)
        else:
            txt_embeds = report_token_embeds

        # Normalize projections
        v = F.normalize(self.img_proj(global_image_embeds), p=2, dim=-1)
        t = F.normalize(self.txt_proj(txt_embeds), p=2, dim=-1)

        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(v, t.T) / self.temperature  # [B, B]
        labels = torch.arange(v.shape[0], device=v.device)

        loss_i2t = F.cross_entropy(sim_matrix, labels)
        loss_t2i = F.cross_entropy(sim_matrix.T, labels)

        return 0.5 * (loss_i2t + loss_t2i)
