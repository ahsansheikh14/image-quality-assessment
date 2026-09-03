"""
Inference & Decision Engine for Image Quality Assessment.

Evaluates an input image against 10 quality metrics and decides whether
the image is suitable for downstream Computer Vision processing.
"""

import torch
from PIL import Image
from app.preprocess import preprocess_image
from training.dataset import LABEL_NAMES


def assess_image_quality(pil_image, model, device="cpu"):
    """
    Assesses the quality of an image and returns individual scores and CV suitability decision.
    
    Decision Rules:
    - REJECT if any single defect >= 50% (0.50)
    - REJECT if 2 or more defects >= 35% (0.35)
    - PASS otherwise (if clean score is reasonable and no threshold criteria met)
    
    Args:
        pil_image: PIL Image object.
        model: Loaded ImageQualityModel instance.
        device: 'cpu' or 'cuda'.
        
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
    
    severe_defects = []    # Defects >= 50%
    moderate_defects = []  # Defects >= 35%
    
    for idx, label in enumerate(LABEL_NAMES):
        score_val = round(probabilities[idx].item(), 4)
        scores[label] = score_val
        
        # Check defects (exclude 'clean' label)
        if label != "clean":
            if score_val >= 0.50:
                severe_defects.append({"issue": label, "confidence": score_val})
                detected_issues.append({"issue": label, "confidence": score_val})
            elif score_val >= 0.35:
                moderate_defects.append({"issue": label, "confidence": score_val})
                detected_issues.append({"issue": label, "confidence": score_val})
            elif score_val >= 0.20:
                # Track minor notices for complete reporting
                detected_issues.append({"issue": label, "confidence": score_val})
            
    # Sort detected issues by highest confidence first
    detected_issues.sort(key=lambda x: x["confidence"], reverse=True)
    
    # 4. Configured Rejection Rule:
    # REJECT if at least 1 severe defect (>= 50%) OR at least 2 moderate defects (>= 35%)
    has_severe = len(severe_defects) >= 1
    has_multiple_moderate = (len(severe_defects) + len(moderate_defects)) >= 2
    
    should_reject = has_severe or has_multiple_moderate
    is_suitable = not should_reject
    
    if is_suitable:
        status = "PASSED"
        if len(detected_issues) > 0:
            top_issue = detected_issues[0]
            recommendation = f"Image quality is acceptable for CV processing (minor {top_issue['issue'].replace('_', ' ')} detected at {top_issue['confidence']*100:.1f}%)."
        else:
            recommendation = "Image quality is high. Suitable for downstream Computer Vision processing."
    else:
        status = "REJECTED"
        flagged = severe_defects + [d for d in moderate_defects if d not in severe_defects]
        issue_descriptions = [f"{item['issue'].replace('_', ' ').title()} ({item['confidence']*100:.1f}%)" for item in flagged]
        
        if has_severe:
            recommendation = f"Image rejected due to severe defect (>= 50%): {', '.join(issue_descriptions)}."
        else:
            recommendation = f"Image rejected due to multiple moderate defects (>= 35%): {', '.join(issue_descriptions)}."
        
    return {
        "status": status,
        "is_suitable_for_cv": is_suitable,
        "recommendation": recommendation,
        "detected_issues": detected_issues,
        "quality_scores": scores
    }
