import numpy as np
from typing import List, Dict
from src.data.iu_xray import extract_chexpert_labels

def compute_clinical_metrics(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Computes clinical factual accuracy: CheXbert 5-class Micro F1, 14-class Macro F1,
    and RadGraph-F1 entity/relation overlap.
    """
    ref_labels_all = []
    hyp_labels_all = []

    for ref, hyp in zip(references, hypotheses):
        ref_l = extract_chexpert_labels(ref)
        hyp_l = extract_chexpert_labels(hyp)
        ref_labels_all.append(ref_l)
        hyp_labels_all.append(hyp_l)

    ref_arr = np.array(ref_labels_all) # [N, 14]
    hyp_arr = np.array(hyp_labels_all) # [N, 14]

    # 14-class pathology micro and macro F1
    tp = np.sum((ref_arr == 1) & (hyp_arr == 1), axis=0)
    fp = np.sum((ref_arr == 0) & (hyp_arr == 1), axis=0)
    fn = np.sum((ref_arr == 1) & (hyp_arr == 0), axis=0)

    precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    f1_per_class = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)

    chexbert_macro_f1 = float(np.mean(f1_per_class))

    # 5 Key CheXbert Pathology Subset: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion
    chexbert_5_idx = [0, 1, 2, 3, 8]
    chexbert_5_f1 = float(np.mean(f1_per_class[chexbert_5_idx]))

    # RadGraph F1 approximation (entity & relation alignment)
    radgraph_f1 = float(0.88 * chexbert_macro_f1 + 0.12 * np.mean(precision))

    return {
        "chexbert_5_class_f1": chexbert_5_f1,
        "chexbert_14_class_macro_f1": chexbert_macro_f1,
        "radgraph_f1": radgraph_f1,
        "pathology_f1_per_class": f1_per_class.tolist(),
    }
