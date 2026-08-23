import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

CHEXPERT_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
    "No Finding",
]

class RadiologyReportDataset(Dataset):
    """
    Unified Radiology Report Dataset Schema across all data sources (IU X-Ray, MIMIC-CXR).
    """

    def __init__(
        self,
        samples: list,
        transform=None,
        tokenizer=None,
        max_text_len: int = 256,
        image_size: int = 224,
    ):
        self.samples = samples
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def _generate_synthetic_image_if_missing(self, path):
        """Helper to return a dummy PIL image if file is unavailable during testing."""
        img = Image.new("RGB", (self.image_size, self.image_size), color=(128, 128, 128))
        return img

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item.get("image_path", "")

        # Load image safely
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                image = self._generate_synthetic_image_if_missing(img_path)
        else:
            image = self._generate_synthetic_image_if_missing(img_path)

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = image

        report_text = item.get("report_text", item.get("findings", "") + " " + item.get("impression", "")).strip()

        # Tokenize report if tokenizer provided
        token_data = {}
        if self.tokenizer is not None:
            try:
                encoded = self.tokenizer(
                    report_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_text_len,
                    return_tensors="pt",
                )
                token_data["input_ids"] = encoded["input_ids"].squeeze(0)
                token_data["attention_mask"] = encoded["attention_mask"].squeeze(0)
            except Exception:
                pass

        if "input_ids" not in token_data or token_data["input_ids"].numel() == 0:
            # Fallback fixed-length token tensor
            words = report_text.lower().split()
            ids = [(abs(hash(w)) % 50000) + 1 for w in words[:self.max_text_len]]
            if len(ids) < self.max_text_len:
                ids += [0] * (self.max_text_len - len(ids))
            token_data["input_ids"] = torch.tensor(ids, dtype=torch.long)
            token_data["attention_mask"] = torch.tensor([1 if x != 0 else 0 for x in ids], dtype=torch.long)

        pathology_labels = item.get("pathology_labels", [0.0] * 14)
        if len(pathology_labels) != 14:
            pathology_labels = pathology_labels[:14] + [0.0] * (14 - len(pathology_labels))
        pathology_tensor = torch.tensor(pathology_labels, dtype=torch.float32)

        # Bounding box mask processing (e.g. 14x14 visual patch grid mask for supervised explainability)
        bbox_mask = torch.zeros((14, 14), dtype=torch.float32)
        has_bbox = False

        bbox_regions = item.get("bbox_regions", None)
        if bbox_regions:
            has_bbox = True
            for box_info in bbox_regions:
                # box format: [ymin, xmin, ymax, xmax] in normalized [0, 1] coordinates
                b = box_info.get("bbox", [0.2, 0.2, 0.8, 0.8])
                ymin, xmin, ymax, xmax = b[0], b[1], b[2], b[3]
                r1, r2 = int(ymin * 14), int(ymax * 14)
                c1, c2 = int(xmin * 14), int(xmax * 14)
                r1, r2 = max(0, r1), min(14, max(r1 + 1, r2))
                c1, c2 = max(0, c1), min(14, max(c1 + 1, c2))
                bbox_mask[r1:r2, c1:c2] = 1.0

        return {
            "patient_id": str(item.get("patient_id", "unknown")),
            "study_id": str(item.get("study_id", "unknown")),
            "image": image_tensor,
            "report_text": report_text,
            "input_ids": token_data["input_ids"],
            "attention_mask": token_data["attention_mask"],
            "pathology_labels": pathology_tensor,
            "bbox_mask": bbox_mask,
            "has_bbox": torch.tensor(1.0 if has_bbox else 0.0, dtype=torch.float32),
            "raw_bbox_regions": json.dumps(bbox_regions or []),
        }
