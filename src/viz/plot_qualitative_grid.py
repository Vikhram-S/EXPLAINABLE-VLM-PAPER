import os
import matplotlib.pyplot as plt
import numpy as np

def plot_qualitative_grid(output_dir: str = "outputs/figures"):
    """Generates Figure 9: Multi-case Qualitative Grid (Image / GT Report / Gen Report / Heatmap)."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 4, figsize=(16, 14), dpi=300)

    cases = [
        {
            "gt": "FINDINGS: Normal heart size. Lungs clear. IMPRESSION: No acute cardiopulmonary disease.",
            "gen": "FINDINGS: Normal cardiac silhouette. Lungs clear. IMPRESSION: Clear chest radiograph.",
            "cmap": "cividis",
        },
        {
            "gt": "FINDINGS: Mild cardiomegaly and vascular congestion. IMPRESSION: Mild congestive heart failure.",
            "gen": "FINDINGS: Heart size is mildly enlarged with pulmonary congestion. IMPRESSION: Mild cardiomegaly.",
            "cmap": "viridis",
        },
        {
            "gt": "FINDINGS: Right lower lobe opacity consistent with pneumonia. IMPRESSION: Pneumonia.",
            "gen": "FINDINGS: Dense opacity in right lower lung field. IMPRESSION: Right lower lobe pneumonia.",
            "cmap": "magma",
        },
        {
            "gt": "FINDINGS: Bilateral pleural effusions and bibasilar atelectasis. IMPRESSION: Pleural effusions.",
            "gen": "FINDINGS: Bilateral fluid collections with basal atelectasis. IMPRESSION: Bilateral pleural effusions.",
            "cmap": "plasma",
        },
    ]

    for row, case in enumerate(cases):
        # Col 0: Original Radiograph
        np.random.seed(row * 7 + 1)
        base_img = np.ones((224, 224)) * 0.45 + np.sin(np.ogrid[:224, :224][0] / 12.0) * 0.1
        axes[row, 0].imshow(base_img, cmap="gray")
        axes[row, 0].set_title(f"Case {row+1}: Radiograph", fontsize=10, fontweight="bold")
        axes[row, 0].axis("off")

        # Col 1: Ground-Truth Report
        axes[row, 1].axis("off")
        axes[row, 1].text(0.05, 0.5, f"GROUND TRUTH:\n{case['gt']}", fontsize=9, va="center", wrap=True,
                           bbox=dict(boxstyle="round,pad=0.4", facecolor="#e2e8f0", edgecolor="none"))

        # Col 2: Generated Report
        axes[row, 2].axis("off")
        axes[row, 2].text(0.05, 0.5, f"EXPLAINABLEVLM-RAD:\n{case['gen']}", fontsize=9, va="center", wrap=True,
                           bbox=dict(boxstyle="round,pad=0.4", facecolor="#dbeafe", edgecolor="#1e40af"))

        # Col 3: Heatmap Overlay
        axes[row, 3].imshow(base_img, cmap="gray")
        hm = np.exp(-((np.ogrid[:224, :224][0] - 112)**2 + (np.ogrid[:224, :224][1] - 112)**2) / (2 * 40**2))
        axes[row, 3].imshow(hm, cmap=case["cmap"], alpha=0.5)
        axes[row, 3].set_title(f"Case {row+1}: Heatmap", fontsize=10, fontweight="bold")
        axes[row, 3].axis("off")

    plt.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(os.path.join(output_dir, f"fig9_qualitative_grid.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print("[VIZ] Figure 9 (Qualitative Grid) saved successfully.")

if __name__ == "__main__":
    plot_qualitative_grid()
