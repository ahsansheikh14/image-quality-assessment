"""
Training Pipeline for Image Quality Assessment Model.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam

from app.model import ImageQualityModel
from training.dataset import get_dataloaders, NUM_CLASSES


def calculate_metrics(outputs, targets, threshold=0.5):
    """
    Computes multi-label accuracy.
    A prediction is correct if predicted probability >= threshold matches ground truth.
    """
    probs = torch.sigmoid(outputs)
    preds = (probs >= threshold).float()
    correct = (preds == targets).float().mean()
    return correct.item()


def train_model(epochs=5, batch_size=32, lr=0.001, device="cpu"):
    """
    Main training routine for ResNet-18 multi-label quality classifier.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Training on device: {device}")

    train_loader, val_loader = get_dataloaders(
        csv_path="data/distorted/labels.csv",
        image_dir="data/distorted",
        batch_size=batch_size
    )

    model = ImageQualityModel(num_classes=NUM_CLASSES, freeze_backbone=True)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/quality_model.pt"

    print("\nStarting training loop...")
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_acc += calculate_metrics(outputs, labels) * images.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_acc / len(train_loader.dataset)

        # Validation Phase
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

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc * 100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  Saved best model checkpoint to {model_save_path} (Val Loss: {best_val_loss:.4f})")

    total_time = time.time() - start_total_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    return history


if __name__ == "__main__":
    train_model(epochs=5, batch_size=32, lr=0.001, device="cpu")
