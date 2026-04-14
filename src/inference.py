"""
inference.py  —  Real-time driving behavior detection on video / webcam
────────────────────────────────────────────────────────────────────────
Usage:
    # Run on a video file:
    python src/inference.py --source data/raw_videos/test.mp4

    # Run on webcam (index 0):
    python src/inference.py --source 0

    # Save annotated output:
    python src/inference.py --source test.mp4 --save outputs/annotated.mp4

Features:
    • Per-frame behavior classification (Normal / Distracted / Phone Usage)
    • Rolling risk score (0–100)
    • Confidence bar overlay
    • Voice alert via pyttsx3 (throttled to once per 5 s)
    • Drowsiness detection via MediaPipe (Eye Aspect Ratio)
    • Post-session timeline chart saved automatically
"""

import cv2
import numpy as np
import time
import argparse
import os
import collections
import threading

import tensorflow as tf
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional voice alert (graceful fallback if pyttsx3 not available)
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 160)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────
SEQ_LEN    = 16
IMG_SIZE   = 224
CLASSES    = ["Normal", "Distracted", "Phone Usage"]
RISK_MAP   = {"Normal": 5, "Distracted": 72, "Phone Usage": 90}

# Overlay colors (BGR)
COLOR_SAFE    = (80, 200, 80)
COLOR_WARN    = (30, 180, 255)
COLOR_DANGER  = (40,  40, 220)
COLOR_BG      = (20,  20,  20)

EAR_THRESHOLD   = 0.22   # below this → drowsy
EAR_CONSEC_FRAMES = 20   # consecutive frames below threshold → alert

# ── MediaPipe face mesh ────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
# Left eye landmark indices (MediaPipe 468-point model)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
           for i in eye_indices]
    # Vertical distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C + 1e-6)


# ── Voice alert (runs in background thread) ────────────────────────────────────
_last_alert_time = 0
_alert_lock = threading.Lock()

def speak(text):
    global _last_alert_time
    if not TTS_AVAILABLE:
        return
    now = time.time()
    with _alert_lock:
        if now - _last_alert_time < 5:
            return
        _last_alert_time = now
    def _run():
        _tts_engine.say(text)
        _tts_engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()


# ── Overlay drawing helpers ────────────────────────────────────────────────────

def draw_overlay(frame, label, confidence, risk, ear, drowsy):
    h, w = frame.shape[:2]

    # Choose color by label
    if label == "Normal":
        color = COLOR_SAFE
    elif label == "Distracted":
        color = COLOR_WARN
    else:
        color = COLOR_DANGER

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Label + confidence
    cv2.putText(frame, f"Status: {label.upper()}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame, f"Conf: {confidence*100:.1f}%",
                (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1)

    # Risk score bar (top-right)
    bar_x, bar_y, bar_w, bar_h = w - 180, 10, 160, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    fill = int(bar_w * risk / 100)
    bar_color = COLOR_SAFE if risk < 30 else (COLOR_WARN if risk < 65 else COLOR_DANGER)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                  bar_color, -1)
    cv2.putText(frame, f"Risk: {risk}/100",
                (bar_x, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_color, 1)

    # EAR / drowsiness indicator (bottom-left)
    ear_color = (40, 40, 220) if drowsy else (180, 180, 180)
    drowsy_text = "DROWSY!" if drowsy else f"EAR: {ear:.2f}"
    cv2.putText(frame, drowsy_text,
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)

    # Alert flash when dangerous
    if label != "Normal":
        cv2.rectangle(frame, (0, 0), (w, h), color, 3)

    return frame


# ── Post-session timeline chart ───────────────────────────────────────────────

def save_timeline(risk_log, label_log, out_path="outputs/session_report.png"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    times = np.arange(len(risk_log))

    # Risk curve
    axes[0].fill_between(times, risk_log, alpha=0.3, color="tomato")
    axes[0].plot(times, risk_log, color="tomato", linewidth=1.5)
    axes[0].axhline(65, color="orange", linestyle="--", linewidth=1, label="High risk threshold")
    axes[0].set_ylabel("Risk score"); axes[0].set_ylim(0, 105)
    axes[0].legend(loc="upper right"); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Driving session analysis")

    # Label timeline
    label_colors = {"Normal": "green", "Distracted": "orange", "Phone Usage": "red"}
    label_nums   = {"Normal": 0, "Distracted": 1, "Phone Usage": 2}
    nums = [label_nums.get(l, 0) for l in label_log]
    scatter_colors = [label_colors.get(l, "gray") for l in label_log]
    axes[1].scatter(times, nums, c=scatter_colors, s=8, zorder=3)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["Normal", "Distracted", "Phone Usage"])
    axes[1].set_xlabel("Sequence index (time →)")
    axes[1].grid(True, alpha=0.3)

    # Summary stats
    total = len(label_log)
    dist_pct = label_log.count("Distracted") / max(total, 1) * 100
    phone_pct = label_log.count("Phone Usage") / max(total, 1) * 100
    fig.text(0.01, 0.01,
             f"Distractions: {dist_pct:.1f}%  |  Phone usage: {phone_pct:.1f}%"
             f"  |  Max risk: {max(risk_log, default=0)}/100"
             f"  |  Avg risk: {int(np.mean(risk_log) if risk_log else 0)}/100",
             fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"✓  Session report saved → {out_path}")


# ── Main inference loop ────────────────────────────────────────────────────────

def run(source, model_path, save_path=None):
    # Load model
    print(f"Loading model from {model_path} …")
    model = tf.keras.models.load_model(model_path)

    # Open video source
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (fw, fh))

    # Rolling frame buffer
    frame_buffer = collections.deque(maxlen=SEQ_LEN)

    # Logs for post-session chart
    risk_log, label_log = [], []

    # Drowsiness state
    ear_counter = 0
    drowsy = False

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    label, confidence, risk = "Initializing…", 0.0, 0
    print("Running inference — press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Drowsiness detection ───────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        ear = 0.35  # default (no face detected)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  fw, fh)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, fw, fh)
            ear = (left_ear + right_ear) / 2.0
            if ear < EAR_THRESHOLD:
                ear_counter += 1
                if ear_counter >= EAR_CONSEC_FRAMES:
                    drowsy = True
                    speak("Drowsiness detected! Please take a break.")
            else:
                ear_counter = 0
                drowsy = False

        # ── Behavior classification ────────────────────────────────────────
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        frame_buffer.append(resized)

        if len(frame_buffer) == SEQ_LEN:
            seq = np.expand_dims(np.array(frame_buffer), axis=0)  # (1,16,224,224,3)
            probs = model.predict(seq, verbose=0)[0]
            idx = int(np.argmax(probs))
            label = CLASSES[idx]
            confidence = float(probs[idx])

            # Compute rolling risk score
            base_risk = RISK_MAP[label]
            risk = int(base_risk * confidence + (1 - confidence) * 10)
            if drowsy:
                risk = min(100, risk + 20)

            risk_log.append(risk)
            label_log.append(label)

            # Voice alerts
            if label == "Phone Usage":
                speak("Warning! Put down your phone.")
            elif label == "Distracted":
                speak("Stay focused on the road.")
            elif drowsy:
                speak("Drowsiness detected!")

        # ── Draw overlay and show ──────────────────────────────────────────
        frame = draw_overlay(frame, label, confidence, risk, ear, drowsy)

        if writer:
            writer.write(frame)
        cv2.imshow("Driving Behavior Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    if risk_log:
        save_timeline(risk_log, label_log)
        print(f"\n── Session Summary ─────────────────────────────")
        print(f"  Total sequences analyzed : {len(label_log)}")
        print(f"  Normal   : {label_log.count('Normal')}")
        print(f"  Distracted: {label_log.count('Distracted')}")
        print(f"  Phone Usage: {label_log.count('Phone Usage')}")
        print(f"  Max risk : {max(risk_log)}/100")
        print(f"  Avg risk : {int(np.mean(risk_log))}/100")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",     default="0",
                    help="Video path or webcam index (0)")
    ap.add_argument("--model_path", default="models/cnn_lstm_best.h5")
    ap.add_argument("--save",       default=None,
                    help="Optional path to save annotated video")
    args = ap.parse_args()
    run(args.source, args.model_path, args.save)
