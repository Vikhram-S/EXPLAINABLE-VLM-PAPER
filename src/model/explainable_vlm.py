import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .backbone import BioViLTBackbone
from .decoder import ReportDecoder
from .alignment import CrossModalAlignmentLoss
from .explainability import SupervisedExplainabilityModule

def get_project_root():
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, "configs")) or os.path.exists(os.path.join(curr, "requirements.txt")):
            return curr
        curr = os.path.dirname(curr)
    return os.getcwd()

class ExplainableVLMRad(nn.Module):
    """
    ExplainableVLM-Rad Architecture.
    Combines frozen BioViL-T vision-text backbone with a lightweight trainable
    report decoder (Visual Mapper + LoRA adapters), InfoNCE cross-modal alignment,
    and supervised attention-gradient explainability.
    """

    def __init__(
        self,
        backbone_name: str = "microsoft/BiomedVLP-BioViL-T",
        decoder_name: str = "distilgpt2",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lambda_ce: float = 1.0,
        lambda_align: float = 0.2,
        lambda_exp: float = 0.3,
    ):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_align = lambda_align
        self.lambda_exp = lambda_exp

        # 1. Vision Backbone (Frozen)
        self.backbone = BioViLTBackbone(model_name=backbone_name, frozen=True)

        # 2. Report Decoder (Trainable mapper + LoRA)
        self.decoder = ReportDecoder(
            model_name=decoder_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        # 3. Cross-Modal Alignment Module
        self.alignment_module = CrossModalAlignmentLoss(temperature=0.07, embed_dim=768)

        # 4. Supervised Explainability Module
        self.explainability_module = SupervisedExplainabilityModule(patch_grid_size=14)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        bbox_masks: Optional[torch.Tensor] = None,
        has_bbox_flags: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        patch_embeds, global_img_embeds = self.backbone(images)
        logits, cross_attn = self.decoder(patch_embeds, input_ids, attention_mask)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=50256)

        loss_align = self.alignment_module(global_img_embeds, patch_embeds)

        if bbox_masks is not None and has_bbox_flags is not None:
            loss_exp, heatmaps = self.explainability_module(cross_attn, logits, bbox_masks, has_bbox_flags)
        else:
            loss_exp = torch.tensor(0.0, device=images.device)
            heatmaps = self.explainability_module.compute_fusion_heatmap(cross_attn, logits)

        total_loss = (
            self.lambda_ce * loss_ce
            + self.lambda_align * loss_align
            + self.lambda_exp * loss_exp
        )

        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_align": loss_align,
            "loss_exp": loss_exp,
            "logits": logits,
            "heatmaps": heatmaps,
        }

    def generate_report(self, images: torch.Tensor, max_new_tokens: int = 128) -> List[str]:
        with torch.no_grad():
            patch_embeds, _ = self.backbone(images)
            reports = self.decoder.generate(patch_embeds, max_new_tokens=max_new_tokens)
        return reports

    def export_model_spec(self, output_path: str = None):
        root = get_project_root()
        target_path = output_path or os.path.join(root, "configs", "model_spec.yaml")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        trainable_pct = round((trainable_params / total_params) * 100, 2) if total_params > 0 else 0.0

        spec = {
            "model_name": "ExplainableVLM-Rad",
            "version": "1.0.0",
            "backbone": {
                "name": "BiomedVLP-BioViL-T",
                "frozen": True,
                "input_resolution": [224, 224],
                "hidden_dim": 768,
            },
            "decoder": {
                "name": "CvT2DistilGPT2-Adapter",
                "hidden_dim": 768,
                "trainable_adapter": "VisualMapper + LoRA",
            },
            "parameter_counts": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "frozen_parameters": frozen_params,
                "trainable_percentage": trainable_pct,
            },
            "loss_weights": {
                "lambda_ce": self.lambda_ce,
                "lambda_align": self.lambda_align,
                "lambda_exp": self.lambda_exp,
            },
        }

        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w") as f:
            yaml.dump(spec, f, default_flow_style=False)
        print(f"[SPEC EXPORT] Model architecture spec written to: {target_path}")
