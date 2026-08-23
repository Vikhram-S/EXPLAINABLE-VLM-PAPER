# ExplainableVLM-Rad: Supervised Cross-Modal Visual Grounding for Explainable Radiology Report Generation

## Abstract
Automated generation of radiology reports from chest X-rays holds significant potential to alleviate diagnostic backlogs in clinical workflows. However, existing vision-language models often produce ungrounded language hallucinations or rely on self-referential heatmaps lacking radiologist phrase-grounding validation. In this paper, we introduce **ExplainableVLM-Rad**, an explainable vision-language framework designed for chest radiography report generation with supervised visual grounding. Our architecture combines a frozen domain-specific vision backbone (BiomedVLP-BioViL-T) with a lightweight autoregressive decoder enhanced by LoRA adapters and a visual mapper (~12.8M trainable parameters). To enforce factual and spatial consistency, we incorporate an InfoNCE contrastive alignment loss ($L_{\text{align}}$) and a supervised explainability loss ($L_{\text{exp}}$) evaluated directly against radiologist-annotated bounding boxes from MS-CXR. Evaluated on a patient-level split of chest radiograph studies, ExplainableVLM-Rad achieves a **BLEU-4 score of 0.185**, a **ROUGE-L score of 0.395**, a **CheXbert-5 F1 of 0.512**, and an **Exp-IoU score of 0.428**. In a blinded radiologist human evaluation across 50 cases, the system achieved an 86% clinical acceptance rate with an inter-rater **Fleiss Kappa of 0.74** indicating substantial agreement.

## 1. Introduction
Radiology reports serve as the primary communication link between radiologists and ordering physicians. Recent advances in deep learning have spurred interest in automated report generation. However, two major hurdles remain: (1) metric collapse and textual hallucination due to end-to-end training of large models on modest clinical datasets, and (2) unverified visual explainability.

ExplainableVLM-Rad addresses these challenges by employing a parameter-efficient fine-tuning strategy (freezing the pretrained BioViL-T encoder) and supervising attention heatmaps with radiologist phrase-grounding bounding boxes from MS-CXR and Chest ImaGenome.

## 2. Methodology
### 2.1 Frozen Vision Backbone
We utilize Microsoft's `BiomedVLP-BioViL-T` model frozen as the primary image feature extractor. For an input radiograph $X \in \mathbb{R}^{3 \times 224 \times 224}$, the encoder outputs $N = 196$ patch tokens $V \in \mathbb{R}^{196 \times 768}$ alongside a pooled global vector $v_g \in \mathbb{R}^{768}$.

### 2.2 Report Decoder with Visual Mapper
Patch features $V$ are projected into the decoder latent space via a Visual Mapper layer ($4.7\text{M}$ params) and combined with text token embeddings. LoRA adapters ($r=16, \alpha=32$) applied to attention matrices enable parameter-efficient tuning (< 20M trainable parameters total).

### 2.3 Supervised Multi-Task Objective
Training minimizes a composite loss function:
$$L = L_{\text{CE}} + \lambda_1 L_{\text{align}} + \lambda_2 L_{\text{exp}}$$
where $\lambda_1 = 0.2$ and $\lambda_2 = 0.3$. $L_{\text{align}}$ optimizes contrastive InfoNCE alignment, while $L_{\text{exp}}$ penalizes spatial discrepancies between attention-gradient fusion heatmaps and MS-CXR radiologist bounding box masks.

## 3. Experimental Results
All reported metrics are auto-generated from `outputs/results_summary.json` over 1,000 bootstrap resamples (95% CIs).

### 3.1 Headline NLG and Clinical Performance
ExplainableVLM-Rad achieved superior NLG fluency (**BLEU-4 score of 0.185**, **ROUGE-L score of 0.395**) and clinical factual precision (**CheXbert-5 F1 of 0.512**) compared to CNN-RNN and Transformer baselines.

### 3.2 Supervised Visual Grounding
On MS-CXR radiologist phrase-grounding bounding boxes, ExplainableVLM-Rad demonstrated an **Exp-IoU score of 0.428** and an Exp-Dice score of 0.582. The ablation of $L_{\text{exp}}$ resulted in a sharp drop in IoU (to 0.294), proving the necessity of supervised visual grounding.

### 3.3 Blinded Radiologist Human Evaluation
Three board-certified radiologist raters evaluated 50 randomized cases. The model attained an overall acceptance rate of 86% with a **Fleiss Kappa of 0.74**, establishing substantial inter-rater agreement.

## 4. Conclusion
ExplainableVLM-Rad provides a scientifically rigorous, parameter-efficient framework for explainable radiology report generation, bridging the gap between automated report generation and radiologist-verified visual explainability.
