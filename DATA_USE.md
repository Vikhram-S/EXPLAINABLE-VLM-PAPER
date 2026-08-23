# Data Use, Compliance, and Governance Statement

## 1. Overview and Ethical Compliance
ExplainableVLM-Rad utilizes chest radiograph datasets for research purposes in automated radiology report generation and visual explainability. All data handling and modeling activities adhere strictly to international data protection principles, institutional review board (IRB) approvals, and research-only data use agreements (DUAs).

## 2. Dataset Specifics and Licensing

### Tier 1 — Public / Open Access
1. **IU X-Ray (Open-i)**
   - **Source**: National Library of Medicine (NLM), National Institutes of Health (NIH).
   - **Size**: 7,470 chest radiographs paired with 3,955 radiology reports.
   - **License / Access**: Open access for research. No credentialing required.
   - **Usage**: Primary dataset for pipeline prototyping, fast iteration, and stage-1 pre-fine-tuning.

2. **NIH ChestX-ray14**
   - **Source**: NIH Clinical Center.
   - **Size**: 112,120 frontal view X-ray images from 30,805 unique patients with 14 disease labels.
   - **License / Access**: Public domain.
   - **Usage**: Auxiliary classification pre-training and pathology distribution verification.

### Tier 2 — PhysioNet Credentialed Access
1. **MIMIC-CXR / MIMIC-CXR-JPG (v2.0.0)**
   - **Source**: PhysioNet / Beth Israel Deaconess Medical Center.
   - **Size**: 377,110 images corresponding to 227,835 radiographic studies.
   - **Credentialing**: CITI Training ("Data or Specimens Only Research") + PhysioNet Signed DUA.
   - **License**: PhysioNet Restricted Health Data License 1.5.
   - **Usage**: Main dataset for Stage 2 fine-tuning on stratified 30,000 study subsample.

2. **MS-CXR (v1.0.0)**
   - **Source**: PhysioNet / Microsoft Research.
   - **Size**: 1,162 radiologist-annotated phrase-grounding bounding boxes paired with DICOM image regions.
   - **Credentialing**: PhysioNet Credentialed Access.
   - **Usage**: Ground-truth validation of attention heatmaps and Grad-CAM visual explainability.

3. **Chest ImaGenome (v1.0.0)**
   - **Source**: PhysioNet.
   - **Size**: 29 anatomical region bounding box annotations across MIMIC-CXR studies.
   - **Credentialing**: PhysioNet Credentialed Access.
   - **Usage**: Secondary anatomical grounding evaluation.

### Tier 3 — Registration Gated
1. **CheXpert Plus**
   - **Source**: Stanford AIMI.
   - **Size**: 223,228 images with RadGraph structured annotations.
   - **Usage**: Auxiliary clinical factual completeness evaluation.

## 3. Data Protection and Non-Redistribution Rules
- **Zero Raw Data Distribution**: No raw DICOM/JPG images or original free-text reports are stored in version control or distributed in public code repositories.
- **Patient Privacy**: All datasets are de-identified under HIPAA Safe Harbor prior to acquisition.
- **Split Integrity**: All data splitters implement strict patient-level splitting (by `patient_id` / `subject_id`) to eliminate patient overlap across training, validation, and test sets.
- **DUA Location**: Signed DUAs for PhysioNet credentialed datasets are archived locally in `credentials/dua_physionet.pdf` (git-ignored).
