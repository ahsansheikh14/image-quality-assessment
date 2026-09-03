"""
Model Architecture for Image Quality Assessment.

Uses Transfer Learning:
    - Backbone: Pretrained ResNet-18 (frozen feature extractor)
    - Head: Custom linear layer outputting 10 quality issue logits
"""

import torch
import torch.nn as nn
from torchvision import models


class ImageQualityModel(nn.Module):
    """
    ResNet-18 based Multi-Label Image Quality Assessment Model.
    
    Predicts 10 quality issue scores:
    ['blur', 'darkness', 'overexposure', 'low_resolution', 'glare', 
     'noise', 'motion_artifacts', 'occlusion', 'poor_framing', 'clean']
    """
    def __init__(self, num_classes=10, freeze_backbone=True):
        super(ImageQualityModel, self).__init__()
        
        # Load ResNet-18 with pretrained ImageNet weights
        weights = models.ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=weights)
        
        # Optionally freeze all convolutional backbone layers
        # Freezing means these weights will not be updated during training (fast training)
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Replace the final fully-connected (fc) layer
        # Original: Linear(in_features=512, out_features=1000)
        # Custom: Linear(in_features=512, out_features=10)
        in_features = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),  # 20% dropout to prevent overfitting
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape [batch_size, 3, 224, 224]
            
        Returns:
            logits: Tensor of shape [batch_size, num_classes] (raw unbounded scores)
        """
        return self.resnet(x)


def load_trained_model(model_path="models/quality_model.pt", num_classes=10, device="cpu"):
    """
    Utility to instantiate the model and load trained weights.
    """
    model = ImageQualityModel(num_classes=num_classes, freeze_backbone=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
