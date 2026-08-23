# Model Card: ExplainableVLM-Rad

## Model Details
- **Model Name**: ExplainableVLM-Rad
- **Model Type**: Explainable Vision-Language Model for Chest Radiography Report Generation
- **Backbone Vision-Text Encoder**: BiomedVLP-BioViL-T (Microsoft, MIT License, frozen)
- **Decoder Architecture**: CvT2DistilGPT2 / DistilGPT2 with Visual Adapter Mapper Layer & LoRA (PEFT)
- **Trainable Parameters**: ~12.8M (out of 145M total parameters)
- **Primary Tasks**: Automated report generation (Findings & Impression) and pixel-level heatmap grounding.

## Intended Use
- **Primary Intended Use**: Research and academic evaluation of explainable AI methods in medical imaging.
- **Intended Users**: Computer vision researchers, biomedical informatics scientists, and radiologist evaluators.
- **Out of Scope / Prohibited Use**: **NOT FOR CLINICAL DEPLOYMENT OR DIRECT PATIENT DIAGNOSIS.** This model is an experimental research system and must not be used as a primary diagnostic tool in healthcare settings.

## Datasets and Provenance
- **Pretraining**: BiomedVLP-BioViL-T pretrained on PubMed and CXR-ReFound datasets.
- **Stage 1 Fine-Tuning**: IU X-Ray dataset (Open-i, NLM NIH) - 3,955 report studies.
- **Stage 2 Fine-Tuning**: MIMIC-CXR-JPG (PhysioNet) stratified subsample (30,000 studies, patient-level split).
- **Ground-Truth Visual Verification**: MS-CXR and Chest ImaGenome radiologist phrase-grounding bounding boxes.

## Training & Loss Specifications
- **Objective Function**: $L = L_{\text{CE}} + \lambda_1 L_{\text{align}} + \lambda_2 L_{\text{exp}}$
  - $L_{\text{CE}}$: Cross-entropy loss on target tokens.
  - $L_{\text{align}}$: InfoNCE contrastive cross-modal feature alignment.
  - $L_{\text{exp}}$: Supervised Grad-CAM / attention bounding-box mask alignment loss.
- **Precision**: FP16 mixed precision with Gradient Accumulation.
- **Hardware Target**: Single NVIDIA T4 (16GB VRAM) / T4x2 / P100.

## Performance Metrics & Evaluation Summary
All performance numbers are auto-generated from `outputs/results_summary.json`:
- **NLG Metrics**: BLEU-1..4, ROUGE-L, METEOR, CIDEr.
- **Clinical Accuracy**: CheXbert-F1, RadGraph-F1.
- **Explainability**: Bounding box Grad-CAM IoU & Dice against MS-CXR, Deletion-Insertion faithfulness AUC.
- **Statistical Rigor**: 95% Bootstrap Confidence Intervals over 1,000 resamples; Wilcoxon signed-rank test.

## Ethical Considerations & Limitations
- **Biases**: Subject to demographic and imaging scanner biases present in MIMIC-CXR and Open-i datasets.
- **Hallucinations**: Language decoders may produce fluent but clinically inaccurate statements without visual grounding supervision.
- **Laterality**: Images must NOT be horizontally flipped during inference or training to preserve cardiomegaly and anatomical laterality assessment.
