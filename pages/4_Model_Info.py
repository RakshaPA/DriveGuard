"""
pages/4_Model_Info.py  —  Architecture, dataset, and technology details
"""

import streamlit as st
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.styles import inject_css, page_header, sidebar_branding

st.set_page_config(page_title="Model Info · DriveGuard", layout="wide", page_icon="🧠")
inject_css()
sidebar_branding()
st.sidebar.page_link("app.py",                    label="Home",            icon="🏠")
st.sidebar.page_link("pages/1_Live_Detection.py", label="Live Detection",  icon="📹")
st.sidebar.page_link("pages/2_Analytics.py",      label="Analytics",       icon="📊")
st.sidebar.page_link("pages/3_Alert_Panel.py",    label="Alert Panel",     icon="🚨")
st.sidebar.page_link("pages/4_Model_Info.py",     label="Model Info",      icon="🧠")
st.sidebar.page_link("pages/5_Report.py",         label="Generate Report", icon="📄")

page_header("Model Information", "Architecture, technologies, dataset, and design decisions")

# ── Architecture diagram (text-based) ─────────────────────────────────────────
st.markdown("### Model architecture")

st.markdown("""
<div style="background:#131929;border:1px solid rgba(0,200,150,0.18);border-radius:14px;
            padding:1.5rem;font-family:'Share Tech Mono',monospace;font-size:0.82rem;
            color:#E8EDF5;line-height:2.2;">
  <div style="color:#6B7A99;font-size:0.7rem;letter-spacing:0.15em;margin-bottom:1rem;">
    PIPELINE ARCHITECTURE
  </div>
  <div style="display:flex;align-items:center;gap:0;flex-wrap:wrap;">
    <div style="background:#1C2438;border:1px solid #00C89633;border-radius:8px;
                padding:8px 14px;color:#00C896;">Input</div>
    <div style="color:#6B7A99;padding:0 8px;">→</div>
    <div style="background:#1C2438;border:1px solid #00C89633;border-radius:8px;
                padding:8px 14px;color:#00C896;">TimeDistributed(MobileNetV2)</div>
    <div style="color:#6B7A99;padding:0 8px;">→</div>
    <div style="background:#1C2438;border:1px solid #00C89633;border-radius:8px;
                padding:8px 14px;color:#00C896;">LSTM(256)</div>
    <div style="color:#6B7A99;padding:0 8px;">→</div>
    <div style="background:#1C2438;border:1px solid #00C89633;border-radius:8px;
                padding:8px 14px;color:#00C896;">Dense(128)</div>
    <div style="color:#6B7A99;padding:0 8px;">→</div>
    <div style="background:#1C2438;border:1px solid #00C89633;border-radius:8px;
                padding:8px 14px;color:#00C896;">Softmax(3)</div>
  </div>
  <div style="margin-top:1rem;color:#6B7A99;font-size:0.75rem;">
    Input shape: (16, 224, 224, 3) &nbsp;·&nbsp;
    Feature vector: 1280-d per frame &nbsp;·&nbsp;
    Classes: Normal / Distracted / Phone Usage
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tech cards ─────────────────────────────────────────────────────────────────
st.markdown("### Technologies used")
tech_cols = st.columns(3)

techs = [
    ("MobileNetV2", "#00C896", [
        "Lightweight CNN backbone",
        "Pretrained on ImageNet (1.4M images)",
        "Global Average Pooling → 1280-d feature",
        "Top 20 layers fine-tuned on driving data",
        "Designed for mobile/edge inference",
    ]),
    ("LSTM Network", "#5B8FFF", [
        "Long Short-Term Memory architecture",
        "256 hidden units",
        "Processes 16-frame sequences",
        "Captures temporal driving patterns",
        "Prevents forgetting with gating mechanism",
    ]),
    ("MediaPipe", "#FFA502", [
        "Google's real-time face mesh",
        "468 facial landmark points",
        "Eye Aspect Ratio (EAR) computation",
        "Drowsiness detected at EAR < 0.22",
        "Runs in parallel with CNN-LSTM",
    ]),
]

for col, (name, color, points) in zip(tech_cols, techs):
    with col:
        bullets = "".join([
            f'<li style="margin-bottom:4px;">{p}</li>' for p in points
        ])
        st.markdown(f"""
        <div style="background:#131929;border:1px solid {color}33;border-radius:14px;
                    padding:1.3rem;height:100%;">
          <div style="font-weight:700;font-size:1rem;color:{color};margin-bottom:0.8rem;">
            {name}
          </div>
          <ul style="color:#6B7A99;font-size:0.82rem;line-height:1.7;
                     padding-left:1.2rem;margin:0;">
            {bullets}
          </ul>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tech_cols2 = st.columns(3)
techs2 = [
    ("Grad-CAM", "#FF4757", [
        "Gradient-weighted Class Activation Maps",
        "Shows what CNN region triggered prediction",
        "Applied to last Conv2D layer",
        "Heatmap overlaid on original frame",
        "Key for model explainability at viva",
    ]),
    ("pyttsx3 Voice Alerts", "#00C896", [
        "Text-to-speech alert system",
        "Spoken warning when risk exceeds threshold",
        "Throttled to once every 5 seconds",
        "Runs on background thread (no lag)",
        "Works offline — no internet required",
    ]),
    ("OpenCV", "#5B8FFF", [
        "Video capture and frame extraction",
        "Frame resizing and colour conversion",
        "Real-time overlay rendering",
        "Confidence bar and status text overlay",
        "Annotated video saving (.mp4)",
    ]),
]
for col, (name, color, points) in zip(tech_cols2, techs2):
    with col:
        bullets = "".join([f'<li style="margin-bottom:4px;">{p}</li>' for p in points])
        st.markdown(f"""
        <div style="background:#131929;border:1px solid {color}33;border-radius:14px;
                    padding:1.3rem;height:100%;">
          <div style="font-weight:700;font-size:1rem;color:{color};margin-bottom:0.8rem;">{name}</div>
          <ul style="color:#6B7A99;font-size:0.82rem;line-height:1.7;padding-left:1.2rem;margin:0;">
            {bullets}
          </ul>
        </div>""", unsafe_allow_html=True)

# ── Dataset info ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Dataset")

d1, d2 = st.columns(2)
with d1:
    st.markdown("""
    <div style="background:#131929;border:1px solid rgba(0,200,150,0.18);border-radius:14px;padding:1.4rem;">
      <div style="font-weight:700;color:#00C896;margin-bottom:0.8rem;">
        DAD — Driver Anomaly Detection Dataset
      </div>
      <ul style="color:#6B7A99;font-size:0.85rem;line-height:1.8;padding-left:1.2rem;margin:0;">
        <li>Frontal + overhead dashboard camera views</li>
        <li>Normal vs anomalous driving clips</li>
        <li>Multiple drivers in varied conditions</li>
        <li>Publicly available on GitHub</li>
        <li>Used for: normal / distracted classification</li>
      </ul>
    </div>""", unsafe_allow_html=True)

with d2:
    st.markdown("""
    <div style="background:#131929;border:1px solid rgba(0,200,150,0.18);border-radius:14px;padding:1.4rem;">
      <div style="font-weight:700;color:#00C896;margin-bottom:0.8rem;">
        DMD — Driver Monitoring Dataset
      </div>
      <ul style="color:#6B7A99;font-size:0.85rem;line-height:1.8;padding-left:1.2rem;margin:0;">
        <li>Fine-grained distraction labels</li>
        <li>Includes phone usage, eating, radio</li>
        <li>RGB + depth + IR camera streams</li>
        <li>32 drivers × multiple sessions</li>
        <li>Used for: phone usage classification</li>
      </ul>
    </div>""", unsafe_allow_html=True)

# ── Training details ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Training configuration")

params = {
    "Input shape":         "(16, 224, 224, 3)",
    "Sequence length":     "16 frames",
    "Frame step":          "Every 2nd frame",
    "Batch size":          "8",
    "Optimizer":           "Adam (lr=1e-4 → 1e-5 phase 2)",
    "Loss function":       "Sparse Categorical Crossentropy",
    "Phase 1 epochs":      "10 (backbone frozen)",
    "Phase 2 epochs":      "20 (top 20 layers unfrozen)",
    "Regularisation":      "Dropout 0.5 + BatchNorm",
    "Data augmentation":   "Random flip + brightness jitter",
    "Class balancing":     "Sklearn compute_class_weight",
    "Early stopping":      "Patience = 7 (val_accuracy)",
}

p_cols = st.columns(2)
items  = list(params.items())
half   = len(items) // 2
for col, chunk in zip(p_cols, [items[:half], items[half:]]):
    with col:
        rows = "".join([
            f"""<tr>
              <td style="color:#6B7A99;padding:6px 0;font-size:0.82rem;">{k}</td>
              <td style="color:#E8EDF5;padding:6px 0;font-size:0.82rem;
                         font-family:'Share Tech Mono',monospace;text-align:right;">{v}</td>
            </tr>"""
            for k, v in chunk
        ])
        st.markdown(f"""
        <div style="background:#131929;border:1px solid rgba(0,200,150,0.15);
                    border-radius:12px;padding:1rem 1.2rem;">
          <table style="width:100%;">{rows}</table>
        </div>""", unsafe_allow_html=True)

# ── Model file status ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Model files")
for fname, desc in [
    ("models/cnn_lstm_best.h5", "Best checkpoint (highest val_accuracy)"),
    ("models/cnn_lstm.h5",      "Final model after all epochs"),
    ("outputs/training_curves.png", "Training accuracy + loss plot"),
    ("outputs/confusion_matrix.png","Confusion matrix"),
]:
    exists = os.path.exists(fname)
    color  = "#00C896" if exists else "#FF4757"
    symbol = "✓" if exists else "✗"
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                background:#131929;border-radius:8px;padding:0.6rem 1rem;margin-bottom:6px;
                border:1px solid {color}22;">
      <span style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:#E8EDF5;">
        {fname}
      </span>
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="color:#6B7A99;font-size:0.78rem;">{desc}</span>
        <span style="color:{color};font-size:1rem;">{symbol}</span>
      </div>
    </div>""", unsafe_allow_html=True)
