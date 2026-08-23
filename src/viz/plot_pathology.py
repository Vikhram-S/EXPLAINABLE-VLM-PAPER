import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.data.dataset import CHEXPERT_PATHOLOGIES

sns.set_theme(style="whitegrid", font="sans-serif")
COLOR_PALETTE = sns.color_palette("colorblind")

def plot_pathology_breakdown(summary_path: str = "outputs/results_summary.json", output_dir: str = "outputs/figures"):
    """Generates Figure 6: Per-pathology F1 breakdown across 14 CheXpert categories."""
    if not os.path.exists(summary_path):
        from src.eval.evaluator import run_full_evaluation
        summary = run_full_evaluation(summary_path)
    else:
        with open(summary_path, "r") as f:
            summary = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    f1_list = summary["proposed_model"]["clinical_metrics"]["pathology_f1_per_class"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    y_pos = np.arange(len(CHEXPERT_PATHOLOGIES))
    bars = ax.barh(y_pos, f1_list, color=COLOR_PALETTE[0], edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(CHEXPERT_PATHOLOGIES, fontsize=10, fontweight="bold")
    ax.invert_yaxis()  # top-down pathology order
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Per-Pathology Clinical F1 Performance Breakdown (14 CheXpert Categories)", fontsize=14, fontweight="bold", pad=12)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.015, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(os.path.join(output_dir, f"fig6_pathology_breakdown.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[VIZ] Figure 6 (Pathology Breakdown) saved successfully.")

if __name__ == "__main__":
    plot_pathology_breakdown()
