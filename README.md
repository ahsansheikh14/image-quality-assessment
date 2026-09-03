# 🔬 Image Quality Assessment Model

A deep learning multi-label quality defect detection system built with **PyTorch (ResNet-18)**, **Transfer Learning**, **FastAPI**, and **Docker**.

Predicts whether an image is suitable for downstream Computer Vision pipelines by scoring 9 defect categories plus a clean baseline.

---

## 📌 Detected Quality Defect Categories

| # | Quality Defect | Description |
| :--- | :--- | :--- |
| 1 | **Blur** | Out-of-focus or unsharp images |
| 2 | **Darkness** | Low light / underexposed conditions |
| 3 | **Overexposure** | Excessively bright / washed-out lighting |
| 4 | **Low Resolution** | Pixelated or heavily downscaled images |
| 5 | **Glare** | Light reflections and bright hotspots |
| 6 | **Noise** | Camera sensor grain and electronic noise |
| 7 | **Motion Artifacts** | Directional streak blur from moving camera |
| 8 | **Occlusion** | Objects or opaque blocks obstructing view |
| 9 | **Poor Framing** | Bad composition / cut-off subjects |
| 10 | **Clean** | High-quality baseline suitable for downstream CV |

---

## 📁 Repository Structure

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
│   └── quality_model.pt # Trained model weights checkpoint
│
├── Dockerfile           # Production container configuration
├── .dockerignore        # Build exclusion rules
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## 🚀 Step-by-Step Guide: How to Run the Project

### 1. Generate Synthetic Training Data (5,000 Images)

```powershell
.\venv\Scripts\python.exe -m training.generate_data
```

---

### 2. Train the ResNet-18 Model

```powershell
.\venv\Scripts\python.exe -m training.train
```

* Trains for 5 epochs using transfer learning on CPU in ~1–2 minutes.
* Automatically saves the best model checkpoint to `models/quality_model.pt`.

---

### 3. Launch the Web Application & Live UI

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Open your browser and navigate to:
👉 **`http://localhost:8000`**

Upload any image from your computer to see real-time defect diagnosis and CV pipeline suitability decision!

---

### 4. Run with Docker

```bash
# Build image
docker build -t image-quality-assessment:v1 .

# Run container
docker run -p 8000:8000 image-quality-assessment:v1
```
