import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font="sans-serif")
COLOR_PALETTE = sns.color_palette("colorblind")

def plot_ablation_study(summary_path: str = "outputs/results_summary.json", output_dir: str = "outputs/figures"):
    """Generates Figure 5: Ablation Study Comparison across all configurations."""
    if not os.path.exists(summary_path):
        from src.eval.evaluator import run_full_evaluation
        summary = run_full_evaluation(summary_path)
    else:
        with open(summary_path, "r") as f:
            summary = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    abl = summary["ablation_matrix"]

    configs = list(abl.keys())
    b4_scores = [abl[c]["bleu_4"] for c in configs]
    rouge_scores = [abl[c]["rouge_l"] for c in configs]
    f1_scores = [abl[c]["chexbert_5_f1"] for c in configs]
    iou_scores = [abl[c]["exp_iou"] for c in configs]

    x = np.arange(len(configs))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)

    ax.bar(x - 1.5 * width, b4_scores, width, label="BLEU-4", color=COLOR_PALETTE[0])
    ax.bar(x - 0.5 * width, rouge_scores, width, label="ROUGE-L", color=COLOR_PALETTE[1])
    ax.bar(x + 0.5 * width, f1_scores, width, label="CheXbert-5 F1", color=COLOR_PALETTE[2])
    ax.bar(x + 1.5 * width, iou_scores, width, label="Exp-IoU (MS-CXR)", color=COLOR_PALETTE[3])

    ax.set_title("Ablation Study Matrix (All Metrics Reported Across All Configurations)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.0, 0.55)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11)

    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(os.path.join(output_dir, f"fig5_ablation_study.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[VIZ] Figure 5 (Ablation Study) saved successfully.")

if __name__ == "__main__":
    plot_ablation_study()
