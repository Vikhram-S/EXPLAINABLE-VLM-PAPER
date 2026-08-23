import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np

# Ensure src module is importable from anywhere
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))

for path in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src"), os.getcwd()]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from src.data.iu_xray import IUXrayDataModule
    from src.data.dataset import CHEXPERT_PATHOLOGIES
except ImportError:
    from data.iu_xray import IUXrayDataModule
    from data.dataset import CHEXPERT_PATHOLOGIES

def run_data_sanity_check(num_samples: int = 4, output_path: str = "outputs/figures/data_sanity_check.png"):
    """
    Renders N random dataset samples with reports, pathology labels, and bounding box overlays
    to verify pipeline integrity before spending GPU hours.
    """
    print("Running Data Pipeline Sanity Check...")
    dm = IUXrayDataModule(data_dir="data/iu_xray", batch_size=4)
    train_ds, val_ds, test_ds = dm.setup()

    print(f"Dataset split sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_samples):
        sample = train_ds[i]
        image_tensor = sample["image"]
        report_text = sample["report_text"]
        labels = sample["pathology_labels"].numpy()
        bbox_mask = sample["bbox_mask"].numpy()

        # Denormalize image for visualization
        img = image_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0.0, 1.0)

        # Plot 1: Radiograph with Bounding Box / Patch Mask overlay
        ax_img = axes[i, 0]
        ax_img.imshow(img)
        ax_img.set_title(f"Sample {i+1}: Patient {sample['patient_id']} / Study {sample['study_id']}", fontsize=11, fontweight="bold")
        ax_img.axis("off")

        # Overlay bbox mask
        if sample["has_bbox"].item() > 0:
            mask_resized = np.kron(bbox_mask, np.ones((16, 16)))
            ax_img.imshow(mask_resized, cmap="jet", alpha=0.35)
            ax_img.text(5, 15, "Ground Truth BBox Mask", color="yellow", fontsize=10, backgroundcolor="black")

        # Plot 2: Text Report and Active Pathology Labels
        ax_txt = axes[i, 1]
        ax_txt.axis("off")

        active_pathologies = [CHEXPERT_PATHOLOGIES[j] for j in range(14) if labels[j] > 0.5]
        path_str = ", ".join(active_pathologies) if active_pathologies else "None"

        display_text = (
            f"REPORT TEXT:\n{report_text[:250]}...\n\n"
            f"ACTIVE PATHOLOGIES (14-class):\n{path_str}\n\n"
            f"HAS GROUND-TRUTH BBOX: {bool(sample['has_bbox'].item())}\n"
            f"IMAGE TENSOR SHAPE: {tuple(image_tensor.shape)}"
        )

        ax_txt.text(
            0.05, 0.5, display_text,
            fontsize=10, verticalalignment="center", wrap=True,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#1a365d", alpha=0.9)
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Data sanity check figure saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--output", type=str, default="outputs/figures/data_sanity_check.png")
    args = parser.parse_args()
    run_data_sanity_check(args.num_samples, args.output)
