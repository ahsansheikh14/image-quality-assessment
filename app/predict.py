"""
Inference & Decision Engine for Image Quality Assessment.

Evaluates an input image against 10 quality metrics and decides whether
the image is suitable for downstream Computer Vision processing.
"""

import torch
from PIL import Image
from app.preprocess import preprocess_image
from training.dataset import LABEL_NAMES


def assess_image_quality(pil_image, model, device="cpu", threshold=0.35):
    """
    Assesses the quality of an image and returns individual scores and CV suitability decision.
    
    Args:
        pil_image: PIL Image object.
        model: Loaded ImageQualityModel instance.
        device: 'cpu' or 'cuda'.
        threshold: Confidence score above which a quality defect is flagged (default 0.35).
        
    Returns:
        dict: Detailed quality report with scores, detected issues, and recommendation.
    """
    # 1. Preprocess image into normalized 4D tensor [1, 3, 224, 224]
    img_tensor = preprocess_image(pil_image).to(device)
    
    # 2. Forward pass through trained model
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0)  # Shape: [10]
        
    # 3. Extract scores and detect defects
    scores = {}
    detected_issues = []
    
    for idx, label in enumerate(LABEL_NAMES):
        score_val = round(probabilities[idx].item(), 4)
        scores[label] = score_val
        
        # Flag any defect (excluding the 'clean' baseline) that exceeds threshold
        if label != "clean" and score_val >= threshold:
            detected_issues.append({
                "issue": label,
                "confidence": score_val
            })
            
    # Sort detected issues by highest confidence first
    detected_issues.sort(key=lambda x: x["confidence"], reverse=True)
    
    # 4. Strict Production Suitability Rule:
    # An image is SUITABLE only if NO defects are detected AND clean score is strong
    is_suitable = (len(detected_issues) == 0) and (scores.get("clean", 0.0) >= 0.35)
    
    if is_suitable:
        status = "PASSED"
        recommendation = "Image quality is high. Suitable for downstream Computer Vision processing."
    else:
        status = "REJECTED"
        if detected_issues:
            issue_descriptions = [f"{item['issue'].replace('_', ' ').title()} ({item['confidence']*100:.1f}%)" for item in detected_issues]
            recommendation = f"Image rejected due to detected defects: {', '.join(issue_descriptions)}."
        else:
            recommendation = "Image rejected: Quality score is too low or uncertain for reliable CV processing."
        
    return {
        "status": status,
        "is_suitable_for_cv": is_suitable,
        "recommendation": recommendation,
        "detected_issues": detected_issues,
        "quality_scores": scores
    }
