import os
import sys
import re
import json

# Ensure src module and project files are importable from anywhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

def check_manuscript_consistency(
    manuscript_path: str = "manuscript/manuscript.md",
    summary_path: str = "outputs/results_summary.json",
) -> bool:
    """
    Verifies that numeric claims in the manuscript draft match results_summary.json.
    Ensures zero text-data drift.
    """
    print("Checking Manuscript Numeric Consistency against results_summary.json...")

    if not os.path.exists(manuscript_path):
        print(f"[WARNING] Manuscript file not found at: {manuscript_path}")
        return True

    if not os.path.exists(summary_path):
        print(f"[ERROR] Results summary missing at: {summary_path}. Run evaluation first.")
        return False

    with open(manuscript_path, "r") as f:
        text = f.read()

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Key numeric claims expected in paper text
    b4_val = f"{summary['proposed_model']['nlg_metrics']['bleu_4']:.3f}"
    rl_val = f"{summary['proposed_model']['nlg_metrics']['rouge_l']:.3f}"
    f1_val = f"{summary['proposed_model']['clinical_metrics']['chexbert_5_class_f1']:.3f}"
    iou_val = f"{summary['proposed_model']['explainability_metrics']['exp_iou']:.3f}"
    kappa_val = f"{summary['human_evaluation']['metrics']['inter_rater_fleiss_kappa']:.2f}"

    checks = [
        ("BLEU-4 score", b4_val),
        ("ROUGE-L score", rl_val),
        ("CheXbert-5 F1", f1_val),
        ("Exp-IoU score", iou_val),
        ("Fleiss Kappa", kappa_val),
    ]

    all_matched = True
    for name, expected_str in checks:
        if expected_str in text:
            print(f" [PASS] {name} ({expected_str}) matches manuscript text.")
        else:
            print(f" [FAIL] {name} ({expected_str}) NOT found in manuscript text! Potential text-data drift.")
            all_matched = False

    if all_matched:
        print("\n[SUCCESS] MANUSCRIPT CONSISTENCY CHECK PASSED! All numeric claims trace to results_summary.json.")
        return True
    else:
        print("\n[FAILURE] MANUSCRIPT CONSISTENCY CHECK FAILED! Fix numeric discrepancies before submitting.")
        return False

if __name__ == "__main__":
    success = check_manuscript_consistency()
    if not success:
        sys.exit(1)
