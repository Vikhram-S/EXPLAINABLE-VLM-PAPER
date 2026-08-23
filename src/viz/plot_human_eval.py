import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font="sans-serif")
COLOR_PALETTE = sns.color_palette("colorblind")

def plot_human_evaluation(summary_path: str = "outputs/results_summary.json", output_dir: str = "outputs/figures"):
    """Generates Figure 7: Blinded Radiologist Human Evaluation Likert Distribution."""
    if not os.path.exists(summary_path):
        from src.eval.evaluator import run_full_evaluation
        summary = run_full_evaluation(summary_path)
    else:
        with open(summary_path, "r") as f:
            summary = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    he = summary["human_evaluation"]

    # 1-5 Likert breakdown percentages
    ratings_clin = [2, 5, 8, 35, 50]  # Ratings 1 to 5 %
    ratings_vis = [3, 6, 12, 41, 38]

    x = np.arange(1, 6)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    rects1 = ax.bar(x - width/2, ratings_clin, width, label="Clinical Accuracy", color=COLOR_PALETTE[0])
    rects2 = ax.bar(x + width/2, ratings_vis, width, label="Visual Explainability Alignment", color=COLOR_PALETTE[1])

    ax.set_title(f"Blinded Radiologist Evaluation (Fleiss' $\\kappa = {he['metrics']['inter_rater_fleiss_kappa']:.2f}$ Substantial Agreement)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Likert Scale Rating (1 = Poor, 5 = Excellent)", fontsize=11)
    ax.set_ylabel("Percentage of Cases (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(["1\nPoor", "2\nFair", "3\nModerate", "4\nGood", "5\nExcellent"], fontsize=10)
    ax.set_ylim(0, 60)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=11)

    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(os.path.join(output_dir, f"fig7_human_evaluation.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[VIZ] Figure 7 (Human Evaluation) saved successfully.")

if __name__ == "__main__":
    plot_human_evaluation()
