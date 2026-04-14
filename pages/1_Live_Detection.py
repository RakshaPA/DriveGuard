"""
pages/1_Live_Detection.py  —  Upload video or webcam, run inference frame by frame
"""

import streamlit as st
import cv2, numpy as np, collections, tempfile, os, time, sys, json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.styles import inject_css, page_header, sidebar_branding, risk_gauge_html, alert_card_html

st.set_page_config(page_title="Live Detection · DriveGuard", layout="wide", page_icon="📹")
inject_css()
sidebar_branding()
st.sidebar.page_link("app.py",                    label="Home",            icon="🏠")
st.sidebar.page_link("pages/1_Live_Detection.py", label="Live Detection",  icon="📹")
st.sidebar.page_link("pages/2_Analytics.py",      label="Analytics",       icon="📊")
st.sidebar.page_link("pages/3_Alert_Panel.py",    label="Alert Panel",     icon="🚨")
st.sidebar.page_link("pages/4_Model_Info.py",     label="Model Info",      icon="🧠")
st.sidebar.page_link("pages/5_Report.py",         label="Generate Report", icon="📄")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Settings")
model_path      = st.sidebar.text_input("Model path", "models/cnn_lstm_best.h5")
conf_threshold  = st.sidebar.slider("Min confidence", 0.40, 0.99, 0.60, step=0.05)
risk_threshold  = st.sidebar.slider("Alert threshold", 30, 90, 65)
driver_name     = st.sidebar.text_input("Driver name", "Driver 1")

SEQ_LEN  = 16
IMG_SIZE = 224
CLASSES  = ["Normal", "Distracted", "Phone Usage"]
RISK_MAP = {"Normal": 8, "Distracted": 72, "Phone Usage": 90}

page_header("Live Detection", "Upload a dashcam video and analyse driver behavior in real time")

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(path)

def predict(model, buf):
    import tensorflow as tf
    seq = np.expand_dims(np.array(buf), 0)
    probs = model.predict(seq, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs.tolist()

def compute_risk(label, conf):
    return int(RISK_MAP[label] * conf + (1 - conf) * 10)

def annotate_frame(frame, label, conf, risk):
    h, w = frame.shape[:2]
    color_map = {
        "Normal":      (80,  220, 140),
        "Distracted":  (30,  180, 255),
        "Phone Usage": (60,  60,  255),
        "Initialising":(200, 200, 200),
        "Detecting...": (200, 200, 200),
        "Uncertain":   (200, 200, 200),
    }
    color = color_map.get(label, (200, 200, 200))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (10, 15, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, label.upper(), (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Conf: {conf*100:.0f}%  Risk: {risk}/100",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 170, 190), 1)
    cv2.rectangle(frame, (0, 0), (w, h), color, 2)
    bar_x, bar_w = w - 170, 150
    cv2.rectangle(frame, (bar_x, 8), (bar_x + bar_w, 20), (30, 40, 55), -1)
    cv2.rectangle(frame, (bar_x, 8), (bar_x + int(bar_w * risk / 100), 20), color, -1)
    return frame

model = load_model(model_path)
if model is None:
    st.error(f"Model not found at `{model_path}`. Train first: `python src/train.py`")
    st.stop()

tab1, tab2 = st.tabs(["  Upload video  ", "  Webcam capture  "])

with tab1:
    uploaded = st.file_uploader("Upload dashcam video", type=["mp4", "avi", "mov"],
                                 label_visibility="collapsed")

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        left, right = st.columns([3, 1])

        with left:
            st.markdown("#### Processed output")
            frame_placeholder = st.empty()

        with right:
            st.markdown("#### Live metrics")
            risk_placeholder   = st.empty()
            status_placeholder = st.empty()
            st.markdown("---")
            st.markdown("#### Incident log")
            log_placeholder = st.empty()

        progress = st.progress(0, text="Initialising...")

        cap   = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        buf   = collections.deque(maxlen=SEQ_LEN)

        risk_log, label_log, conf_log, incidents = [], [], [], []
        frame_idx = 0
        label, conf, risk = "Initialising", 0.0, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
            buf.append(resized)

            if len(buf) == SEQ_LEN:
                raw_label, raw_conf, probs = predict(model, buf)

                # Confidence gate — don't trust low-confidence predictions
                if raw_conf < conf_threshold:
                    label = "Uncertain"
                    conf  = raw_conf
                else:
                    label = raw_label
                    conf  = raw_conf

                risk = compute_risk(label if label in RISK_MAP else "Normal", conf)
                risk_log.append(risk)
                label_log.append(label)
                conf_log.append(conf)

                if label not in ("Normal", "Uncertain", "Initialising") and conf >= conf_threshold:
                    ts = f"{frame_idx // 30 // 60:02d}:{frame_idx // 30 % 60:02d}"
                    incidents.append({"time": ts, "label": label,
                                      "risk": risk, "conf": conf})

            ann = annotate_frame(
                cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (640, 360)),
                label, conf, risk
            )
            frame_placeholder.image(
                cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                width=640
            )

            risk_placeholder.markdown(risk_gauge_html(risk), unsafe_allow_html=True)

            badge_colors = {
                "Normal":       "#16a34a",
                "Distracted":   "#d97706",
                "Phone Usage":  "#dc2626",
                "Uncertain":    "#94a3b8",
                "Initialising": "#94a3b8",
            }
            bc = badge_colors.get(label, "#94a3b8")
            status_placeholder.markdown(f"""
            <div style="background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid {bc};
                        border-radius:10px;padding:0.9rem 1rem;margin-top:0.5rem;">
              <div style="font-size:0.95rem;font-weight:600;color:{bc};">
                {label}
              </div>
              <div style="font-size:0.78rem;color:#94a3b8;margin-top:3px;">
                {conf*100:.0f}% confidence
              </div>
            </div>""", unsafe_allow_html=True)

            log_html = "".join([
                alert_card_html(i["time"], i["label"], i["risk"], i["conf"])
                for i in incidents[-5:]
            ]) or "<div style='color:#94a3b8;font-size:0.82rem;'>No incidents yet</div>"
            log_placeholder.markdown(log_html, unsafe_allow_html=True)

            frame_idx += 1
            progress.progress(min(frame_idx / total, 1.0),
                              text=f"Frame {frame_idx}/{total} · {label}")

        cap.release()
        os.unlink(tmp_path)
        progress.empty()

        # ── Save session for analytics ─────────────────────────────────────────
        clean_labels = [l for l in label_log if l in RISK_MAP]
        session = {
            "driver":            driver_name,
            "date":              datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total_sequences":   len(label_log),
            "normal":            label_log.count("Normal"),
            "distracted":        label_log.count("Distracted"),
            "phone_usage":       label_log.count("Phone Usage"),
            "max_risk":          int(max(risk_log)) if risk_log else 0,
            "avg_risk":          int(np.mean(risk_log)) if risk_log else 0,
            "incidents":         incidents,
            "risk_log":          risk_log,
            "label_log":         label_log,
        }
        os.makedirs("outputs", exist_ok=True)
        session_file = f"outputs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(session_file, "w") as f:
            json.dump(session, f)
        st.session_state["last_session"] = session

        st.success(f"Analysis complete — {len(incidents)} incidents detected. "
                   f"Session saved to `{session_file}`.")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total sequences", len(label_log))
        c2.metric("Normal",          label_log.count("Normal"))
        c3.metric("Distracted",      label_log.count("Distracted"))
        c4.metric("Phone usage",     label_log.count("Phone Usage"))
        c5.metric("Max risk",        f"{max(risk_log) if risk_log else 0}/100")

      

with tab2:
    st.markdown("""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
                padding:1.5rem;text-align:center;">
      <div style="font-size:0.8rem;font-weight:600;color:#2563eb;margin-bottom:0.5rem;">
        WEBCAM MODE
      </div>
      <p style="color:#64748b;font-size:0.88rem;margin:0 0 1rem;">
        For live webcam inference, run the command-line tool which provides
        lower-latency real-time detection:
      </p>
      <code style="background:#f1f5f9;color:#1e293b;padding:8px 20px;border-radius:8px;
                   font-size:0.85rem;display:inline-block;border:1px solid #e2e8f0;">
        python src/inference.py --source 0
      </code>
      <p style="color:#64748b;font-size:0.8rem;margin:1rem 0 0;">
        Replace <code style="background:#f1f5f9;padding:2px 6px;border-radius:4px;">0</code>
        with your camera index. Press <strong style="color:#1e293b;">Q</strong> to quit.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    img = st.camera_input("Or capture a single frame for quick analysis")
    if img and model:
        arr     = np.frombuffer(img.getvalue(), np.uint8)
        frame   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        seq     = np.stack([resized] * SEQ_LEN, axis=0)
        label, conf, probs = predict(model, seq)
        risk = compute_risk(label, conf)
        ann  = annotate_frame(cv2.resize(frame, (640, 360)), label, conf, risk)
        st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                 caption=f"{label} · {conf*100:.0f}% confidence",
                 width=640)
        st.markdown(risk_gauge_html(risk), unsafe_allow_html=True)