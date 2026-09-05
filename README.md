# Image Quality Assessment Model

A deep learning multi-label quality defect detection system built with **PyTorch (ResNet-18)**, **Transfer Learning**, **FastAPI**, and **Docker**.

Evaluates images prior to ingestion into downstream Computer Vision pipelines by scoring 9 defect categories plus a clean baseline.

---

## Defect Categories

| # | Defect Type | Description |
| :--- | :--- | :--- |
| 1 | **Blur** | Out-of-focus or unsharp images |
| 2 | **Darkness** | Low light / underexposed conditions |
| 3 | **Overexposure** | Excessively bright / washed-out lighting |
| 4 | **Low Resolution** | Pixelated or heavily downscaled images |
| 5 | **Glare** | Light reflections and bright hotspots |
| 6 | **Noise** | Camera sensor grain and electronic noise |
| 7 | **Motion Artifacts** | Directional streak blur from moving camera/subject |
| 8 | **Occlusion** | Objects or opaque blocks obstructing view |
| 9 | **Poor Framing** | Bad composition / cut-off subjects |
| 10 | **Clean** | High-quality baseline suitable for downstream processing |

---

## Benchmarks & Evaluation

Trained on 5,000 synthetically distorted images (4,000 train / 1,000 validation split):

* **Validation Accuracy**: **98.74%**
* **Validation Loss (BCE)**: **0.0468**
* **Inference Latency**: **~140 ms per image on CPU**
* **Decision Rules**:
  * Reject if any single defect confidence $\ge 50\%$.
  * Reject if 2 or more defects confidence $\ge 35\%$.
  * Pass otherwise.

---

## Automated Testing & Verification Suite

The repository includes an automated test script in `test_quality_pipeline.py`.

### Run Automated Tests:

```powershell
python test_quality_pipeline.py
```

### Verified Test Results:

```text
======================================================================
TEST & VERIFICATION SUMMARY REPORT
======================================================================
Test Case                        | Latency   | Decision   | Result
----------------------------------------------------------------------
Clean Source Image               | 273.51 ms | PASSED     | PASS
Heavily Blurred Image (Defect)   |  99.57 ms | REJECTED   | PASS
Severely Dark Image (Defect)     |  97.89 ms | REJECTED   | PASS
Overexposed Image (Defect)       |  95.07 ms | REJECTED   | PASS
======================================================================
Average Inference Latency: 141.51 ms per image on CPU
OVERALL QUALITY PIPELINE STATUS: PASSED
======================================================================
```

---

## Repository Structure

```
image-quality-assessment/
│
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application & web UI
│   ├── model.py         # ResNet-18 architecture with custom head
│   ├── predict.py       # Inference and CV suitability decision engine
│   └── preprocess.py    # Image transforms and normalization
│
├── training/
│   ├── __init__.py
│   ├── generate_data.py # Synthetic distortion generator (5,000 images)
│   ├── dataset.py       # PyTorch Dataset and DataLoader
│   └── train.py         # Training loop with BCEWithLogitsLoss & Adam
│
├── data/
│   ├── clean/           # Source clean images
│   └── distorted/       # 5,000 synthetic distorted images + labels.csv
│
├── models/
│   └── quality_model.pt # Trained model weights checkpoint (98.74% accuracy)
│
├── test_quality_pipeline.py # Automated verification test script
├── Dockerfile           # Production container configuration
├── .dockerignore        # Build exclusion rules
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## Step-by-Step Execution Guide

### 1. Generate Synthetic Dataset (5,000 Images)

```powershell
python -m training.generate_data
```

---

### 2. Train the Model

```powershell
python -m training.train
```

* Saves the best checkpoint to `models/quality_model.pt`.

---

### 3. Start the Web Service

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

Access the user interface at:
`http://localhost:8000`

---

### 4. Run with Docker

```bash
# Build image
docker build -t image-quality-assessment:v1 .

# Run container
docker run -p 8000:8000 image-quality-assessment:v1
```
