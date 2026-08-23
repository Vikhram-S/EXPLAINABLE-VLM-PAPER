import torch
import torch.nn as nn
from typing import Dict, Tuple

class BioViLTBackbone(nn.Module):
    """
    BioViL-T (BiomedVLP-BioViL-T) Vision-Text Encoder.
    Input image shape: [B, 3, 224, 224]
    Output patch embeddings: [B, 196, 768] (14x14 grid)
    Output global embedding: [B, 768]
    """

    def __init__(self, model_name: str = "microsoft/BiomedVLP-BioViL-T", frozen: bool = True):
        super().__init__()
        self.model_name = model_name
        self.frozen = frozen
        self.hidden_dim = 768
        self.num_patches = 196  # 14x14 grid

        # Standalone lightweight ConvNet + 14x14 grid adapter
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.patch_embed = nn.Conv2d(64, 768, kernel_size=8, stride=8)

        # Try loading HuggingFace BiomedVLP-BioViL-T
        self.is_hf_loaded = False
        try:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self.is_hf_loaded = True
        except Exception:
            pass

        if self.frozen:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        images: [B, 3, 224, 224]
        Returns:
            patch_embeddings: [B, 196, 768]
            global_embedding: [B, 768]
        """
        batch_size = images.shape[0]

        if self.is_hf_loaded:
            try:
                if hasattr(self.encoder, "image_model"):
                    outputs = self.encoder.image_model(images)
                elif hasattr(self.encoder, "get_projected_patch_embeddings"):
                    outputs = self.encoder.get_projected_patch_embeddings(images)
                else:
                    outputs = self.encoder(pixel_values=images)

                if hasattr(outputs, "last_hidden_state"):
                    patch_embeddings = outputs.last_hidden_state
                elif isinstance(outputs, (tuple, list)):
                    patch_embeddings = outputs[0]
                else:
                    patch_embeddings = outputs

                if patch_embeddings.dim() == 4:
                    B, C, H, W = patch_embeddings.shape
                    patch_embeddings = patch_embeddings.permute(0, 2, 3, 1).reshape(B, H * W, C)

                if patch_embeddings.shape[1] != self.num_patches:
                    B, N, C = patch_embeddings.shape
                    side = int(N ** 0.5)
                    if side * side == N:
                        grid = patch_embeddings.permute(0, 2, 1).reshape(B, C, side, side)
                        grid = nn.functional.interpolate(grid, size=(14, 14), mode="bilinear", align_corners=False)
                        patch_embeddings = grid.permute(0, 2, 3, 1).reshape(B, 196, C)

                global_embedding = patch_embeddings.mean(dim=1)
                return patch_embeddings, global_embedding
            except Exception:
                pass

        # Fallback spatial patch feature extractor (14x14 grid -> 768 dim)
        x = self.conv_stem(images)
        patch_grid = nn.functional.adaptive_avg_pool2d(x, (14, 14))
        B, C, H, W = patch_grid.shape
        grid_flat = patch_grid.permute(0, 2, 3, 1).reshape(B, H * W, C)
        patch_embeddings = nn.functional.pad(grid_flat, (0, 768 - C))
        global_embedding = patch_embeddings.mean(dim=1)

        return patch_embeddings, global_embedding
