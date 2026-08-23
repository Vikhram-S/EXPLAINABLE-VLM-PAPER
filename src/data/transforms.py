import torchvision.transforms as T
from PIL import Image

# BioViL-T / Standard Medical Image Normalization Parameters
BIOVIL_MEAN = [0.485, 0.456, 0.406]
BIOVIL_STD = [0.229, 0.224, 0.225]

def get_transforms(split: str = "train", image_size: int = 224):
    """
    Returns image transformation pipeline.
    CRITICAL: Horizontal flip is strictly prohibited for chest radiographs
    to preserve left/right laterality for conditions like cardiomegaly.
    """
    if split == "train":
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomRotation(degrees=5),  # Small safe rotation
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=BIOVIL_MEAN, std=BIOVIL_STD),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=BIOVIL_MEAN, std=BIOVIL_STD),
        ])
