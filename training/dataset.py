"""
PyTorch Dataset & DataLoader for Image Quality Assessment.

What is a PyTorch Dataset?
    Think of it like a MongoDB collection. It stores all your data and knows
    how to fetch 1 item by index: dataset[42] returns (image, label).

What is a DataLoader?
    Think of it like pagination. Instead of loading all 5,000 images into RAM
    at once, it feeds them to the model in small shuffled batches of 32.

This file:
    1. Reads labels.csv to know which filename maps to which quality issue.
    2. Loads each image from disk, resizes to 224x224, normalizes pixel values.
    3. Converts the text label ("blur") into a one-hot vector ([1,0,0,0,0,0,0,0,0,0]).
    4. Splits data into 80% training and 20% validation sets.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# ============================================================
# LABEL DEFINITIONS
# ============================================================

# These must match EXACTLY the labels in labels.csv (same order matters!)
LABEL_NAMES = [
    "blur",
    "darkness",
    "overexposure",
    "low_resolution",
    "glare",
    "noise",
    "motion_artifacts",
    "occlusion",
    "poor_framing",
    "clean",
]

NUM_CLASSES = len(LABEL_NAMES)  # 10


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

# Why these specific transforms?
# 1. Resize(224, 224): ResNet-18 was trained on 224x224 images. We must match this.
# 2. ToTensor(): Converts PIL Image (0-255 integers) to PyTorch Tensor (0.0-1.0 floats).
#    Also rearranges dimensions from (Height, Width, Channels) to (Channels, Height, Width).
#    PyTorch expects [C, H, W] format, not [H, W, C] like PIL/NumPy.
# 3. Normalize(): Subtracts ImageNet mean and divides by ImageNet std deviation.
#    This puts pixel values in the same range that ResNet-18 was originally trained on.
#    Without this, the pretrained weights would receive unexpected input values.

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Randomly flip image left-right (data augmentation)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet RGB channel means
        std=[0.229, 0.224, 0.225],   # ImageNet RGB channel standard deviations
    ),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # No random flip for validation — we want consistent, repeatable evaluation
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# PYTORCH DATASET CLASS
# ============================================================

class QualityDataset(Dataset):
    """
    Custom PyTorch Dataset for image quality assessment.

    When PyTorch needs image #42, it calls dataset[42], which triggers __getitem__(42).
    This method:
        1. Reads the filename and label from the CSV data.
        2. Opens the image file from disk.
        3. Applies transforms (resize, normalize).
        4. Converts the text label to a one-hot tensor.
        5. Returns (image_tensor, label_tensor).
    """

    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path:  Path to labels.csv (maps filenames to quality labels).
            image_dir: Path to the folder containing distorted images.
            transform: Image preprocessing pipeline (resize, normalize, etc.).
        """
        self.image_dir = image_dir
        self.transform = transform

        # Read the CSV into a pandas DataFrame
        # DataFrame is like a table: columns = ["filename", "label"]
        self.data = pd.read_csv(csv_path)

        print(f"Loaded dataset: {len(self.data)} images, {NUM_CLASSES} classes")
        print(f"Label distribution:\n{self.data['label'].value_counts().to_string()}")

    def __len__(self):
        """Returns total number of images in dataset. PyTorch needs this."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Fetches a single (image, label) pair by index.
        
        This is called automatically by DataLoader during training.
        Think of it like: dataset[42] returns the 42nd image and its label.
        """
        # Step 1: Get filename and label text from CSV row
        row = self.data.iloc[index]
        filename = row["filename"]
        label_text = row["label"]  # e.g., "blur"

        # Step 2: Load image from disk
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # Step 3: Apply transforms (resize to 224x224, normalize)
        if self.transform:
            image = self.transform(image)

        # Step 4: Convert text label to one-hot vector
        # Example: "blur" → [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #          "clean" → [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        label_index = LABEL_NAMES.index(label_text)
        label_tensor = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        label_tensor[label_index] = 1.0  # Set the correct position to 1

        return image, label_tensor


# ============================================================
# DATA LOADING HELPER FUNCTION
# ============================================================

def get_dataloaders(csv_path="data/distorted/labels.csv",
                    image_dir="data/distorted",
                    batch_size=32,
                    train_split=0.8):
    """
    Creates train and validation DataLoaders.
    
    Args:
        csv_path:    Path to labels.csv.
        image_dir:   Path to distorted images folder.
        batch_size:  How many images per batch (32 is standard for CPU training).
        train_split: Fraction of data for training (0.8 = 80% train, 20% val).
    
    Returns:
        train_loader: DataLoader for training (shuffled).
        val_loader:   DataLoader for validation (not shuffled).
    
    Why split into train and validation?
        - Training data: The model learns patterns from this data.
        - Validation data: We test the model on data it has NEVER seen during training.
          This tells us if the model actually learned general patterns,
          or just memorized the training images (overfitting).
    """
    # Create full dataset with training transforms
    full_dataset = QualityDataset(csv_path, image_dir, transform=train_transforms)

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size

    print(f"\nSplitting dataset: {train_size} training, {val_size} validation")

    # random_split randomly assigns each image to either train or val set
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Override validation transforms (no random flipping)
    # Note: This is a simplified approach. In production, you'd use separate Dataset instances.
    val_dataset.dataset.transform = val_transforms

    # Create DataLoaders (the "pagination" layer)
    # shuffle=True for training: randomizes batch order each epoch so the model
    # doesn't memorize the order of images (prevents overfitting).
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"Val batches:   {len(val_loader)} (batch_size={batch_size})")

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test: load dataset and print one batch shape
    train_loader, val_loader = get_dataloaders()

    # Grab one batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"\nBatch image tensor shape: {images.shape}")
    # Expected: [32, 3, 224, 224] = 32 images, 3 RGB channels, 224x224 pixels
    print(f"Batch label tensor shape: {labels.shape}")
    # Expected: [32, 10] = 32 images, 10 quality labels each
    print(f"Sample label: {labels[0]} → {LABEL_NAMES[labels[0].argmax().item()]}")
