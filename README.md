# ExplainableVLM-Rad

[![Reproducibility Check](https://img.shields.io/badge/Reproducibility-Verified-success.svg)](#reproducibility-and-consistency-check)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An explainable vision-language model for automated chest radiography report generation with supervised visual phrase-grounding.

---

## Repository Structure

```
explainablevlm-rad/
├── README.md                          # Master reproduction guide
├── MODEL_CARD.md                      # Model specifications, scope, and limitations
├── DATA_USE.md                        # Compliance, ethics, and DUA statement
├── requirements.txt                   # Pinned dependency requirements
├── environment.yml                    # Conda environment specification
├── configs/
│   ├── data_manifest.yaml             # Provenance and split tracking
│   ├── model_spec.yaml                # Architecture spec and parameter breakdown
│   └── experiments/
│       ├── stage1_iu_xray.yaml        # Stage 1 experiment config
│       ├── stage2_mimic_cxr.yaml      # Stage 2 experiment config
│       └── ablation_no_exp_loss.yaml  # Ablation experiment config
├── src/
│   ├── data/                          # Dataset schemas and transforms (patient-level split)
│   ├── model/                         # BioViL-T + Decoder + Alignment + Explainability
│   ├── train.py                       # Training engine with checkpoint auto-resume
│   ├── eval/                          # NLG, clinical F1, explainability, and bootstrapping
│   └── viz/                           # Journal-quality figure and LaTeX table generators
├── scripts/
│   ├── sanity_check_data.py           # Visual data pipeline check
│   ├── check_manuscript_consistency.py# Text-data drift safeguard
│   └── generate_human_eval_sheet.py   # Blinded human evaluation generator & Kappa
├── notebooks/                         # Thin Colab/Kaggle launchers
├── outputs/
│   ├── checkpoints/                   # Model weights (git-ignored)
│   ├── figures/                       # 300 DPI PNG, SVG, PDF figures
│   ├── tables/                        # Booktabs LaTeX tables
│   └── results_summary.json           # Single source of truth for all metrics
└── manuscript/
    └── manuscript.md                  # Journal paper draft
```

---

## Quick Start & Reproduction Steps

### 1. Environment Setup
```bash
git clone https://github.com/explainablevlm-rad/explainablevlm-rad.git
cd explainablevlm-rad

# Option A: Conda
conda env create -f environment.yml
conda activate explainablevlm-rad

# Option B: Pip
pip install -r requirements.txt
```

### 2. Run Data Pipeline Sanity Check
Before spending GPU hours, visually verify the data loader, patient-level splits, and bounding box overlays:
```bash
python scripts/sanity_check_data.py --num_samples 4
```

### 3. Execute Stage 1 Training (IU X-Ray)
```bash
python src/train.py --config configs/experiments/stage1_iu_xray.yaml
```

### 4. Run Master Evaluation & Generate Publication Outputs
Run evaluation across proposed model, baselines, and ablation matrix:
```bash
python src/eval/evaluator.py
```
Generate all 8 journal-quality figures (300 DPI PNG, SVG, PDF) and camera-ready LaTeX tables:
```bash
python src/viz/plot_curves.py
python src/viz/plot_comparison.py
python src/viz/plot_heatmaps.py
python src/viz/plot_ablation.py
python src/viz/plot_pathology.py
python src/viz/plot_human_eval.py
python src/viz/plot_faithfulness.py
python src/viz/plot_qualitative_grid.py
python src/viz/generate_latex_tables.py
```

### 5. Verify Manuscript Consistency (Text-Data Drift Safeguard)
```bash
python scripts/check_manuscript_consistency.py
```

---

## Key Results Summary

All headline metrics are generated directly from `outputs/results_summary.json`:

| Model | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr | CheXbert-5 F1 | Exp-IoU (MS-CXR) |
|---|---|---|---|---|---|---|
| CNN-RNN Baseline | 0.285 | 0.092 | 0.245 | 0.850 | 0.380 | 0.182 |
| Transformer Baseline | 0.342 | 0.128 | 0.310 | 1.120 | 0.425 | 0.245 |
| CvT2DistilGPT2 | 0.385 | 0.155 | 0.352 | 1.340 | 0.468 | 0.312 |
| **ExplainableVLM-Rad (Ours)** | **0.428** | **0.185** | **0.395** | **1.620** | **0.512** | **0.428** |

---

## Ethical Compliance & Data Governance
See [DATA_USE.md](DATA_USE.md) for details on PhysioNet DUAs, patient privacy protection, patient-level split enforcement, and dataset licenses.
See [MODEL_CARD.md](MODEL_CARD.md) for model limitations and explicit out-of-scope clinical deployment warnings.
