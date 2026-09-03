"""
FastAPI Application & Interactive Web Interface for Image Quality Assessment.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from io import BytesIO
from PIL import Image
import torch
import os

from app.model import load_trained_model, ImageQualityModel
from app.predict import assess_image_quality
from training.dataset import NUM_CLASSES

app = FastAPI(title="Image Quality Assessment Service", version="1.0")

# Global state
state = {}


@app.on_event("startup")
def startup_event():
    print("Loading Image Quality Model...")
    model_path = "models/quality_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(model_path):
        model = load_trained_model(model_path=model_path, num_classes=NUM_CLASSES, device=device)
        print("✅ Loaded trained quality model weights.")
    else:
        print("⚠️ No trained weights found. Loading baseline model. Run `python -m training.train` to train.")
        model = ImageQualityModel(num_classes=NUM_CLASSES, freeze_backbone=True)
        model.to(device)
        model.eval()
        
    state["model"] = model
    state["device"] = device


@app.get("/health")
def health_check():
    return {"status": "ok", "device": state.get("device", "cpu")}


@app.post("/predict")
async def predict_quality(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns full quality defect scores
    and downstream Computer Vision suitability decision.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse image: {str(e)}")
        
    try:
        result = assess_image_quality(
            pil_image=image,
            model=state["model"],
            device=state["device"]
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/", response_class=HTMLResponse)
def root_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Quality Assessment Tool</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f2f5; margin: 0; padding: 30px 20px; }
            .container { max-width: 900px; margin: 0 auto; background: white; border-radius: 12px; padding: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
            h1 { margin-top: 0; color: #1a202c; font-size: 26px; }
            p.sub { color: #718096; margin-bottom: 25px; }
            .upload-box { border: 2px dashed #cbd5e0; padding: 30px; text-align: center; border-radius: 8px; cursor: pointer; background: #f8fafc; }
            .upload-box:hover { border-color: #3182ce; background: #ebf8ff; }
            .preview-container { display: flex; gap: 30px; margin-top: 30px; flex-wrap: wrap; }
            .image-col { flex: 1; min-width: 280px; text-align: center; }
            .image-col img { max-width: 100%; max-height: 320px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .results-col { flex: 1.3; min-width: 300px; }
            .badge { display: inline-block; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 14px; margin-bottom: 15px; }
            .badge-pass { background: #c6f6d5; color: #22543d; }
            .badge-fail { background: #fed7d7; color: #742a2a; }
            .score-bar-container { margin-bottom: 12px; }
            .score-label { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: #4a5568; margin-bottom: 4px; }
            .bar-bg { background: #edf2f7; border-radius: 6px; height: 10px; overflow: hidden; }
            .bar-fill { height: 100%; border-radius: 6px; transition: width 0.4s ease; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔬 Image Quality Assessment Model</h1>
            <p class="sub">Detect defects (blur, glare, darkness, overexposure, motion artifacts, occlusion, framing, low resolution) to verify CV pipeline suitability.</p>
            
            <div class="upload-box" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleImageUpload(event)">
                <p style="margin: 0; font-size: 16px; color: #4a5568; font-weight: 500;">📁 Click or drag an image here to test quality</p>
            </div>

            <div id="outputSection" class="preview-container" style="display: none;">
                <div class="image-col">
                    <img id="imgPreview" src="" alt="Uploaded image">
                </div>
                <div class="results-col">
                    <div id="statusBadge"></div>
                    <p id="recommendationText" style="color: #4a5568; font-size: 14px; margin-bottom: 20px;"></p>
                    <div id="scoreBars"></div>
                </div>
            </div>
        </div>

        <script>
            async function handleImageUpload(event) {
                const file = event.target.files[0];
                if (!file) return;

                // Preview image
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('imgPreview').src = e.target.result;
                    document.getElementById('outputSection').style.display = 'flex';
                };
                reader.readAsDataURL(file);

                // Send to API
                const formData = new FormData();
                formData.append('file', file);

                document.getElementById('statusBadge').innerHTML = '<span class="badge" style="background:#e2e8f0; color:#4a5568;">Analyzing image quality...</span>';

                try {
                    const response = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await response.json();

                    if (data.error) {
                        alert('Server error: ' + data.error);
                        return;
                    }

                    // Render Badge
                    const badge = document.getElementById('statusBadge');
                    if (data.is_suitable_for_cv) {
                        badge.innerHTML = '<span class="badge badge-pass">✅ PASSED FOR COMPUTER VISION</span>';
                    } else {
                        badge.innerHTML = '<span class="badge badge-fail">❌ REJECTED (DEFECTS DETECTED)</span>';
                    }

                    document.getElementById('recommendationText').innerText = data.recommendation;

                    // Render Bars
                    const barsContainer = document.getElementById('scoreBars');
                    barsContainer.innerHTML = Object.entries(data.quality_scores).map(([label, score]) => {
                        const pct = (score * 100).toFixed(1);
                        let barColor = '#3182ce';
                        if (label === 'clean') {
                            barColor = score >= 0.5 ? '#38a169' : '#e53e3e';
                        } else {
                            barColor = score >= 0.5 ? '#e53e3e' : (score >= 0.35 ? '#dd6b20' : '#3182ce');
                        }
                        return `
                            <div class="score-bar-container">
                                <div class="score-label">
                                    <span>${label.replace('_', ' ').toUpperCase()}</span>
                                    <span>${pct}%</span>
                                </div>
                                <div class="bar-bg">
                                    <div class="bar-fill" style="width: ${pct}%; background: ${barColor};"></div>
                                </div>
                            </div>
                        `;
                    }).join('');

                } catch (err) {
                    alert('Error analyzing image: ' + err.message);
                }
            }
        </script>
    </body>
    </html>
    """
