import os
import sys
import random
import pandas as pd
import numpy as np

# Ensure src module and project files are importable from anywhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

def generate_blinded_human_eval_sheet(
    num_cases: int = 50,
    output_path: str = "outputs/blinded_human_eval_sheet.csv",
    seed: int = 42,
):
    """
    Generates a randomized, blinded CSV spreadsheet for radiologist evaluation.
    Model identities are anonymized to Model A, Model B, Model C.
    """
    random.seed(seed)
    cases = []
    for i in range(1, num_cases + 1):
        case_id = f"CASE_{i:03d}"
        models = ["ExplainableVLM-Rad", "CvT2DistilGPT2", "Transformer"]
        random.shuffle(models)

        for rank_idx, model_name in enumerate(models):
            anon_label = chr(65 + rank_idx)  # Model A, Model B, Model C
            cases.append({
                "case_id": case_id,
                "blinded_model_label": f"Model {anon_label}",
                "image_ref": f"data/iu_xray/images/case_{i}.jpg",
                "ground_truth_report": "FINDINGS: Lungs are clear without focal consolidation. Heart size normal. IMPRESSION: Normal chest radiograph.",
                "generated_report": f"Sample generated text from Model {anon_label} for Case {i}.",
                "heatmap_overlay_ref": f"outputs/figures/heatmaps/case_{i}_{anon_label}.png",
                "rater1_clinical_accuracy_1to5": "",
                "rater1_visual_alignment_1to5": "",
                "rater2_clinical_accuracy_1to5": "",
                "rater2_visual_alignment_1to5": "",
                "rater3_clinical_accuracy_1to5": "",
                "rater3_visual_alignment_1to5": "",
                "accept_for_clinical_use_yes_no": "",
                "comments": "",
            })

    df = pd.DataFrame(cases)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[HUMAN EVAL] Blinded radiologist evaluation sheet generated: {output_path}")

def compute_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Computes Fleiss' Kappa for inter-rater agreement across multiple raters.
    ratings_matrix: [N_cases, N_categories]
    """
    N, k = ratings_matrix.shape
    n = np.sum(ratings_matrix[0, :])

    p_j = np.sum(ratings_matrix, axis=0) / (N * n)
    P_i = (np.sum(ratings_matrix**2, axis=1) - n) / (n * (n - 1))

    P_bar = np.mean(P_i)
    P_e_bar = np.sum(p_j**2)

    if P_e_bar == 1.0:
        return 1.0

    kappa = (P_bar - P_e_bar) / (1.0 - P_e_bar)
    return float(kappa)

if __name__ == "__main__":
    generate_blinded_human_eval_sheet()
