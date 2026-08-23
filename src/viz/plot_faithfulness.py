import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font="sans-serif")
COLOR_PALETTE = sns.color_palette("colorblind")

def plot_faithfulness_curves(summary_path: str = "outputs/results_summary.json", output_dir: str = "outputs/figures"):
    """Generates Figure 8: Deletion-Insertion Faithfulness Curves."""
    if not os.path.exists(summary_path):
        from src.eval.evaluator import run_full_evaluation
        summary = run_full_evaluation(summary_path)
    else:
        with open(summary_path, "r") as f:
            summary = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    faith = summary["proposed_model"]["faithfulness"]

    del_curve = faith["deletion_curve"]
    ins_curve = faith["insertion_curve"]
    del_auc = faith["deletion_auc"]
    ins_auc = faith["insertion_auc"]

    steps = np.linspace(0.0, 100.0, len(del_curve))

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    ax.plot(steps, del_curve, label=f"Deletion Curve (AUC = {del_auc:.3f}) ↓", color=COLOR_PALETTE[3], linewidth=2.5, marker="o")
    ax.plot(steps, ins_curve, label=f"Insertion Curve (AUC = {ins_auc:.3f}) ↑", color=COLOR_PALETTE[2], linewidth=2.5, marker="s")

    ax.set_title("Faithfulness Evaluation: Pixel Deletion and Insertion Curves", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Percentage of Heatmap-Ranked Patches Modified (%)", fontsize=11)
    ax.set_ylabel("Normalized Model Output Confidence", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11)

    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(os.path.join(output_dir, f"fig8_faithfulness_curves.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[VIZ] Figure 8 (Faithfulness Curves) saved successfully.")

if __name__ == "__main__":
    plot_faithfulness_curves()
