"""
Image Preprocessing for Inference.

Converts any incoming PIL image into a normalized PyTorch tensor
ready to be passed to the ImageQualityModel.
"""

from torchvision import transforms

# Preprocessing transforms for inference (must match training normalization!)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_image(pil_image):
    """
    Prepares a single PIL image for model inference.
    
    Args:
        pil_image: PIL Image in RGB mode.
        
    Returns:
        Tensor of shape [1, 3, 224, 224] with added batch dimension.
    """
    img_rgb = pil_image.convert("RGB")
    tensor = inference_transforms(img_rgb)
    # Add batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
    return tensor.unsqueeze(0)
