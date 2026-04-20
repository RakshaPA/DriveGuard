"""
inference.py  —  Real-time driving behavior detection
Run: python src/inference.py --source your_video.mp4
     python src/inference.py --source 0   (webcam)
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

mp_face = None
try:
    mp_face = mp.solutions.face_mesh
except AttributeError:
    mp_face = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyttsx3
    _tts = pyttsx3.init()
    _tts.setProperty("rate", 160)
    TTS_OK = True
except Exception:
    TTS_OK = False

SEQ_LEN   = 16
IMG_SIZE  = 224
CLASSES   = ["Normal", "Distracted", "Phone Usage"]
RISK_MAP  = {"Normal": 5, "Distracted": 72, "Phone Usage": 90}

COLOR_SAFE   = (80,  200, 80)
COLOR_WARN   = (30,  180, 255)
COLOR_DANGER = (40,  40,  220)

EAR_THRESH = 0.22
EAR_FRAMES = 20

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


def eye_aspect_ratio(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
           for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C + 1e-6)


_last_alert = 0
_lock = threading.Lock()

def speak(text):
    global _last_alert
    if not TTS_OK:
        return
    now = time.time()
    with _lock:
        if now - _last_alert < 5:
            return
        _last_alert = now
    threading.Thread(
        target=lambda: (_tts.say(text), _tts.runAndWait()),
        daemon=True
    ).start()


def load_model_auto(model_path):
    """Try .keras first, then .h5 fallback."""
    keras_path = model_path.replace(".h5", ".keras")
    if os.path.exists(keras_path):
        print(f"Loading model: {keras_path}")
        return tf.keras.models.load_model(keras_path)
    elif os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        return tf.keras.models.load_model(model_path)
    else:
        print(f"[ERROR] Model not found: {keras_path} or {model_path}")
        print("  Train first: python src/train.py --epochs 30 --batch 8")
        return None


def predict_from_buffer(model, frame_buffer):
    seq   = np.expand_dims(np.array(frame_buffer), axis=0)
    probs = model.predict(seq, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs


def predict_single_frame(model, frame_rgb):
    resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    seq     = np.stack([resized] * SEQ_LEN, axis=0)
    seq     = np.expand_dims(seq, axis=0)
    probs   = model.predict(seq, verbose=0)[0]
    idx     = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs


def draw_overlay(frame, label, confidence, risk, ear, drowsy):
    h, w = frame.shape[:2]
    color = (COLOR_SAFE   if label == "Normal" else
             COLOR_WARN   if label == "Distracted" else
             COLOR_DANGER)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (10, 14, 26), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"STATUS: {label.upper()}",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
                (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 190, 210), 1)

    bx, by, bw, bh = w - 190, 12, 165, 16
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 50, 65), -1)
    fill = int(bw * min(risk, 100) / 100)
    bar_c = (COLOR_SAFE if risk < 30 else (COLOR_WARN if risk < 65 else COLOR_DANGER))
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), bar_c, -1)
    cv2.putText(frame, f"Risk: {risk}/100",
                (bx, by + bh + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, bar_c, 1)

    ear_color = (40, 40, 220) if drowsy else (160, 170, 180)
    ear_text  = "DROWSY! TAKE A BREAK" if drowsy else f"EAR: {ear:.2f}  Alert"
    cv2.putText(frame, ear_text,
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, ear_color, 2)

    if label != "Normal" or drowsy:
        cv2.rectangle(frame, (0, 0), (w, h), color, 3)
    return frame


def save_timeline(risk_log, label_log, path="outputs/session_report.png"):
    if not risk_log:
        return
    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.patch.set_facecolor("#0B0F1A")
    t = np.arange(len(risk_log))

    axes[0].set_facecolor("#131929")
    axes[0].fill_between(t, risk_log, alpha=0.25, color="#FF4757")
    axes[0].plot(t, risk_log, color="#FF4757", lw=1.5)
    axes[0].axhline(65, color="#FFA502", linestyle="--", lw=1, label="High risk")
    axes[0].axhline(30, color="#00C896", linestyle="--", lw=1, label="Moderate")
    axes[0].set_ylabel("Risk score", color="#6B7A99")
    axes[0].set_ylim(0, 108)
    axes[0].tick_params(colors="#6B7A99")
    axes[0].legend(facecolor="#1C2438", labelcolor="#E8EDF5", fontsize=8)
    axes[0].set_title("Driving session", color="#E8EDF5")
    for sp in axes[0].spines.values(): sp.set_edgecolor("#1C2438")

    lmap  = {"Normal": 0, "Distracted": 1, "Phone Usage": 2}
    cmap  = {"Normal": "#00C896", "Distracted": "#FFA502", "Phone Usage": "#FF4757"}
    axes[1].set_facecolor("#131929")
    axes[1].scatter(t, [lmap.get(l, 0) for l in label_log],
                    c=[cmap.get(l, "#6B7A99") for l in label_log], s=10)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["Normal", "Distracted", "Phone Usage"], color="#6B7A99")
    axes[1].set_xlabel("Sequence →", color="#6B7A99")
    axes[1].tick_params(colors="#6B7A99")
    for sp in axes[1].spines.values(): sp.set_edgecolor("#1C2438")

    total = len(label_log)
    fig.text(0.01, 0.01,
             f"Distracted: {label_log.count('Distracted')/max(total,1)*100:.1f}%  |  "
             f"Phone: {label_log.count('Phone Usage')/max(total,1)*100:.1f}%  |  "
             f"Max risk: {max(risk_log)}/100  |  Avg: {int(np.mean(risk_log))}/100",
             fontsize=8, color="#6B7A99")
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓  Session report → {path}")


def run(source, model_path, save_path=None):
    model = load_model_auto(model_path)
    if model is None:
        return

    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {fw}x{fh} @ {fps:.1f} FPS  |  Press Q to quit")

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh)
        )

    frame_buffer = collections.deque(maxlen=SEQ_LEN)
    risk_log, label_log = [], []
    ear_counter = 0
    drowsy      = False
    label, confidence, risk = "Loading...", 0.0, 0
    frame_count = 0

    face_mesh = None
    if mp_face is not None:
        face_mesh = mp_face.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
    else:
        print("[WARN] Mediapipe face mesh not available. Drowsiness detection disabled.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Drowsiness
        ear = 0.0
        if face_mesh is not None:
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lm    = result.multi_face_landmarks[0].landmark
                l_ear = eye_aspect_ratio(lm, LEFT_EYE,  fw, fh)
                r_ear = eye_aspect_ratio(lm, RIGHT_EYE, fw, fh)
                ear   = (l_ear + r_ear) / 2.0
                if ear < EAR_THRESH:
                    ear_counter += 1
                    if ear_counter >= EAR_FRAMES:
                        drowsy = True
                        speak("Drowsiness detected! Please take a break.")
                else:
                    ear_counter = 0
                    drowsy      = False
            else:
                ear_counter = 0
                drowsy      = False
        else:
            drowsy = False

        # Preprocess and buffer
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        frame_buffer.append(resized)

        # Predict
        if frame_count < SEQ_LEN:
            label, confidence, _ = predict_single_frame(model, rgb)
        else:
            label, confidence, _ = predict_from_buffer(model, frame_buffer)

        risk = int(RISK_MAP.get(label, 5) * confidence + (1 - confidence) * 10)
        if drowsy:
            risk = min(100, risk + 20)

        risk_log.append(risk)
        label_log.append(label)

        if label == "Phone Usage" and confidence > 0.65:
            speak("Warning! Put down your phone.")
        elif label == "Distracted" and confidence > 0.65:
            speak("Stay focused on the road.")

        frame = draw_overlay(frame, label, confidence, risk, ear, drowsy)

        if writer:
            writer.write(frame)
        cv2.imshow("DriveGuard AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    if face_mesh is not None:
        face_mesh.close()

    if risk_log:
        save_timeline(risk_log, label_log)
        total = len(label_log)
        print("\n── Session Summary ──────────────────────────────")
        print(f"  Frames      : {frame_count}")
        print(f"  Normal      : {label_log.count('Normal')} ({label_log.count('Normal')/total*100:.1f}%)")
        print(f"  Distracted  : {label_log.count('Distracted')} ({label_log.count('Distracted')/total*100:.1f}%)")
        print(f"  Phone Usage : {label_log.count('Phone Usage')} ({label_log.count('Phone Usage')/total*100:.1f}%)")
        print(f"  Max risk    : {max(risk_log)}/100")
        print(f"  Avg risk    : {int(np.mean(risk_log))}/100")
        print("─────────────────────────────────────────────────")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",     default="0")
    ap.add_argument("--model_path", default="models/cnn_lstm_best.keras")
    ap.add_argument("--save",       default=None)
    args = ap.parse_args()
    run(args.source, args.model_path, args.save)