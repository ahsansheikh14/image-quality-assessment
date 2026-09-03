"""
Training Pipeline for Image Quality Assessment Model.

Training Loop Overview:
    1. DataLoader feeds batches of [32, 3, 224, 224] images and [32, 10] labels.
    2. Forward pass: Model outputs raw logits [32, 10].
    3. Loss Calculation: BCEWithLogitsLoss compares predicted logits with ground truth.
    4. Backward pass: Backpropagation computes gradients (dLoss/dw).
    5. Optimizer step: Adam updates the weights of our custom classification head.
    6. Validation: Every epoch, evaluate accuracy & loss on unseen validation data.
    7. Save: Saves the best performing model weights to models/quality_model.pt.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from app.model import ImageQualityModel
from training.dataset import get_dataloaders, LABEL_NAMES, NUM_CLASSES


def calculate_metrics(outputs, targets, threshold=0.5):
    """
    Computes multi-label accuracy.
    A prediction is correct if predicted probability >= threshold matches ground truth.
    """
    probs = torch.sigmoid(outputs)
    preds = (probs >= threshold).float()
    
    # Check exact match across all labels for each sample
    # Or calculate average per-class accuracy
    correct = (preds == targets).float().mean()
    return correct.item()


def train_model(epochs=10, batch_size=32, lr=0.001, device="cpu"):
    """
    Main training function.
    """
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"🚀 Training on device: {device}")

    # 2. Get DataLoaders
    train_loader, val_loader = get_dataloaders(
        csv_path="data/distorted/labels.csv",
        image_dir="data/distorted",
        batch_size=batch_size
    )

    # 3. Instantiate Model
    model = ImageQualityModel(num_classes=NUM_CLASSES, freeze_backbone=True)
    model.to(device)

    # 4. Define Loss Function & Optimizer
    # BCEWithLogitsLoss combines Sigmoid + Binary Cross-Entropy in a numerically stable way
    criterion = nn.BCEWithLogitsLoss()
    
    # We only pass trainable parameters (the custom classification head) to Adam
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr)

    # 5. Tracking history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/quality_model.pt"

    print("\n--- Starting Training ---")
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # ----------------------------------------------------
        # TRAINING PHASE
        # ----------------------------------------------------
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Reset gradients from previous batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass (compute gradients)
            loss.backward()

            # Update weights
            optimizer.step()

            # Track statistics
            train_loss += loss.item() * images.size(0)
            train_acc += calculate_metrics(outputs, labels) * images.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_acc / len(train_loader.dataset)

        # ----------------------------------------------------
        # VALIDATION PHASE
        # ----------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_acc += calculate_metrics(outputs, labels) * images.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_acc / len(val_loader.dataset)

        # Record history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")

        # Save model if validation loss improves
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  ⭐ Saved best model checkpoint to {model_save_path} (Val Loss: {best_val_loss:.4f})")

    total_time = time.time() - start_total_time
    print(f"\n Training complete in {total_time / 60:.2f} minutes!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    return history


if __name__ == "__main__":
    # Train for 5 epochs (takes ~1-2 minutes on CPU with frozen backbone)
    train_model(epochs=5, batch_size=32, lr=0.001, device="cpu")
