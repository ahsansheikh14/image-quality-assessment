"""
Inference & Decision Engine for Image Quality Assessment.

Evaluates an input image against 10 quality metrics and decides whether
the image is suitable for downstream Computer Vision processing.
"""

import torch
from PIL import Image
from app.preprocess import preprocess_image
from training.dataset import LABEL_NAMES


def assess_image_quality(pil_image, model, device="cpu", threshold=0.40):
    """
    Assesses the quality of an image and returns individual scores and CV suitability decision.
    
    Args:
        pil_image: PIL Image object.
        model: Loaded ImageQualityModel instance.
        device: 'cpu' or 'cuda'.
        threshold: Confidence score above which a quality defect is flagged.
        
    Returns:
        dict: Detailed quality report with scores, detected issues, and recommendation.
    """
    # 1. Preprocess
    img_tensor = preprocess_image(pil_image).to(device)
    
    # 2. Forward pass
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0)  # Shape: [10]
        
    # 3. Format scores into a readable dictionary
    scores = {}
    detected_issues = []
    
    for idx, label in enumerate(LABEL_NAMES):
        score_val = round(probabilities[idx].item(), 4)
        scores[label] = score_val
        
        # Flag issues that cross the threshold (ignoring 'clean' label)
        if label != "clean" and score_val >= threshold:
            detected_issues.append({
                "issue": label,
                "confidence": score_val
            })
            
    # Sort detected issues by highest confidence first
    detected_issues.sort(key=lambda x: x["confidence"], reverse=True)
    
    # 4. Overall Suitability Decision
    # Suitable if no critical quality defects are detected or clean score is high
    is_suitable = len(detected_issues) == 0 or scores.get("clean", 0.0) >= 0.60
    
    if is_suitable:
        status = "PASSED"
        recommendation = "Image quality is high. Suitable for downstream Computer Vision processing."
    else:
        status = "REJECTED"
        issue_names = [item["issue"] for item in detected_issues]
        recommendation = f"Image rejected due to detected defects: {', '.join(issue_names)}. Retake or enhance before processing."
        
    return {
        "status": status,
        "is_suitable_for_cv": is_suitable,
        "recommendation": recommendation,
        "detected_issues": detected_issues,
        "quality_scores": scores
    }
