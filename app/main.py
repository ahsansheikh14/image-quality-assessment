"""
FastAPI Application & Web Interface for Image Quality Assessment.
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

state = {}


@app.on_event("startup")
def startup_event():
    print("Loading image quality model...")
    model_path = "models/quality_model.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(model_path):
        model = load_trained_model(model_path=model_path, num_classes=NUM_CLASSES, device=device)
        print("Loaded trained model weights.")
    else:
        print("No trained weights found. Initializing base model.")
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
    Accepts an uploaded image file and returns quality defect scores
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
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Image Quality Assessment Tool</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 40px 20px; }
            .container { max-width: 900px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 32px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
            h1 { margin-top: 0; font-size: 22px; font-weight: 700; color: #0f172a; }
            p.sub { color: #64748b; font-size: 14px; margin-bottom: 24px; line-height: 1.5; }
            .upload-box { border: 2px dashed #cbd5e1; padding: 36px 20px; text-align: center; border-radius: 6px; cursor: pointer; background: #f8fafc; transition: all 0.2s ease; }
            .upload-box:hover { border-color: #2563eb; background: #eff6ff; }
            .preview-container { display: flex; gap: 32px; margin-top: 32px; flex-wrap: wrap; }
            .image-col { flex: 1; min-width: 280px; text-align: center; }
            .image-col img { max-width: 100%; max-height: 320px; border-radius: 6px; border: 1px solid #e2e8f0; }
            .results-col { flex: 1.2; min-width: 300px; }
            .badge { display: inline-block; padding: 6px 12px; border-radius: 4px; font-weight: 600; font-size: 12px; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 12px; }
            .badge-pass { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
            .badge-fail { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
            .badge-loading { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }
            .score-bar-container { margin-bottom: 12px; }
            .score-label { display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; color: #475569; margin-bottom: 4px; }
            .bar-bg { background: #f1f5f9; border-radius: 4px; height: 8px; overflow: hidden; }
            .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Image Quality Assessment</h1>
            <p class="sub">Automated defect detection pipeline for verifying image suitability prior to downstream Computer Vision processing.</p>
            
            <div class="upload-box" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleImageUpload(event)">
                <p style="margin: 0; font-size: 14px; color: #334155; font-weight: 500;">Click or drag an image here to evaluate quality metrics</p>
            </div>

            <div id="outputSection" class="preview-container" style="display: none;">
                <div class="image-col">
                    <img id="imgPreview" src="" alt="Uploaded input">
                </div>
                <div class="results-col">
                    <div id="statusBadge"></div>
                    <p id="recommendationText" style="color: #475569; font-size: 13px; line-height: 1.5; margin-bottom: 20px;"></p>
                    <div id="scoreBars"></div>
                </div>
            </div>
        </div>

        <script>
            async function handleImageUpload(event) {
                const file = event.target.files[0];
                if (!file) return;

                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('imgPreview').src = e.target.result;
                    document.getElementById('outputSection').style.display = 'flex';
                };
                reader.readAsDataURL(file);

                const formData = new FormData();
                formData.append('file', file);

                document.getElementById('statusBadge').innerHTML = '<span class="badge badge-loading">Analyzing image...</span>';

                try {
                    const response = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await response.json();

                    if (data.error) {
                        alert('Server error: ' + data.error);
                        return;
                    }

                    const badge = document.getElementById('statusBadge');
                    if (data.is_suitable_for_cv) {
                        badge.innerHTML = '<span class="badge badge-pass">Passed For Pipeline</span>';
                    } else {
                        badge.innerHTML = '<span class="badge badge-fail">Rejected (Defects Detected)</span>';
                    }

                    document.getElementById('recommendationText').innerText = data.recommendation;

                    const barsContainer = document.getElementById('scoreBars');
                    barsContainer.innerHTML = Object.entries(data.quality_scores).map(([label, score]) => {
                        const pct = (score * 100).toFixed(1);
                        let barColor = '#2563eb';
                        if (label === 'clean') {
                            barColor = score >= 0.5 ? '#16a34a' : '#dc2626';
                        } else {
                            barColor = score >= 0.5 ? '#dc2626' : (score >= 0.35 ? '#ea580c' : '#2563eb');
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
