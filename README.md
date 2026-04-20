# Real-Time Aggressive Driving Detection System
### 23PT26 — Raksha P A

A deep learning system that detects unsafe driving behavior from dashcam footage
using a **MobileNetV2 + LSTM** architecture, with real-time overlay, drowsiness
detection, Grad-CAM explainability, and a Streamlit dashboard.

---

## Project structure

```
driving_detection/
├── app.py                   ← Streamlit web dashboard
├── requirements.txt
├── src/
│   ├── preprocess.py        ← Video → frame sequences (.npy)
│   ├── model.py             ← CNN-LSTM architecture
│   ├── train.py             ← Two-phase training loop
│   ├── inference.py         ← Real-time detection + overlay + voice alert
│   ├── gradcam.py           ← Grad-CAM explainability heatmaps
│   └── evaluate.py          ← Confusion matrix + per-class accuracy
├── models/                  ← Saved .h5 model files (created after training)
├── data/
│   ├── raw_videos/
│   │   ├── normal/          ← .mp4 / .avi clips of normal driving
│   │   ├── distracted/      ← .mp4 / .avi clips of distracted driving
│   │   └── phone_usage/     ← .mp4 / .avi clips of phone usage
│   └── sequences/           ← Preprocessed .npy files (created by preprocess.py)
└── outputs/                 ← Charts, reports, annotated videos
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

Organise your videos into class folders:
```
data/raw_videos/
    normal/        video1.mp4  video2.mp4 ...
    distracted/    video3.mp4  ...
    phone_usage/   video5.mp4  ...
```

**Recommended datasets:**
- [DAD Dataset](https://github.com/okankop/Driver-Anomaly-Detection) — normal vs anomalous
- [DMD Dataset](https://dmd.vicomtech.org/) — fine-grained distraction classes

---

## Running the pipeline

### Step 1 — Preprocess videos into sequences
```bash
python src/preprocess.py --data_dir data/raw_videos --out_dir data/sequences
```
Outputs: `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`

### Step 2 — Train the model
```bash
python src/train.py --epochs 30 --batch 8
```
- Phase 1 (10 epochs): backbone frozen, trains LSTM + head only
- Phase 2 (20 epochs): top 20 backbone layers unfrozen for fine-tuning
- Best model saved to `models/cnn_lstm_best.h5`
- Training curves saved to `outputs/training_curves.png`

### Step 3 — Evaluate
```bash
python src/evaluate.py --seq_dir data/sequences --model models/cnn_lstm_best.h5
```
Outputs confusion matrix and per-class accuracy charts to `outputs/`.

### Step 4a — Real-time inference (webcam or video)
```bash
# Webcam
python src/inference.py --source 0

# Video file
python src/inference.py --source data/raw_videos/test.mp4

# Save annotated video
python src/inference.py --source test.mp4 --save outputs/annotated.mp4
```
Press **Q** to quit. Session report chart saved to `outputs/session_report.png`.

### Step 4b — Streamlit dashboard
```bash
streamlit run app.py
```
Upload a video and view the full analysis in your browser.

### Step 5 — Grad-CAM explainability
```bash
python src/gradcam.py --video data/raw_videos/test.mp4 \
                      --model models/cnn_lstm_best.h5  \
                      --out   outputs/gradcam.png
```

---

## Model architecture

```
Input: (16, 224, 224, 3)            ← 16-frame sequence
    │
    ├─ TimeDistributed(MobileNetV2)  ← feature extraction per frame
    │    pretrained on ImageNet
    │    output: (16, 1280)
    │
    ├─ LSTM(256)                     ← temporal behavior modeling
    │    output: (256,)
    │
    ├─ BatchNormalization
    ├─ Dense(128, relu)
    ├─ Dropout(0.5)
    └─ Dense(3, softmax)             ← Normal / Distracted / Phone Usage
```

---

## Extra features included

| Feature | File | Description |
|---------|------|-------------|
| Drowsiness detection | `inference.py` | Eye Aspect Ratio via MediaPipe |
| Voice alerts | `inference.py` | pyttsx3 spoken warnings |
| Grad-CAM heatmaps | `gradcam.py` | Visual explainability for viva |
| Two-phase training | `train.py` | Frozen backbone → fine-tune |
| Data augmentation | `train.py` | Random flip + brightness jitter |
| Class weight balancing | `train.py` | Handles dataset imbalance |
| Session report chart | `inference.py` | Post-drive risk timeline |
| Streamlit dashboard | `app.py` | Upload video → full analysis |

---


