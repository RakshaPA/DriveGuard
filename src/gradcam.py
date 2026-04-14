"""
gradcam.py  —  Grad-CAM heatmap visualisation
───────────────────────────────────────────────
Shows which regions of a frame the CNN is attending to when it makes
a prediction — great for your viva demo!

Usage:
    python src/gradcam.py --video data/raw_videos/test.mp4 \
                          --model models/cnn_lstm_best.h5  \
                          --out   outputs/gradcam.png
"""

import cv2
import numpy as np
import argparse
import os
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE = 224
SEQ_LEN  = 16
CLASSES  = ["Normal", "Distracted", "Phone Usage"]


def get_gradcam_heatmap(model, frame_seq, class_idx=None):
    """
    Compute Grad-CAM for the last conv layer of MobileNetV2.

    Parameters
    ----------
    frame_seq : np.ndarray  shape (1, SEQ_LEN, 224, 224, 3)
    class_idx : int | None  — if None, uses predicted class

    Returns
    -------
    heatmap : np.ndarray  shape (7, 7)  values in [0, 1]
    pred_class : int
    probs : np.ndarray
    """
    # Build a sub-model that outputs the last conv layer activations
    # and the final predictions
    backbone = model.get_layer("feature_extractor").layer
    last_conv = None
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    if last_conv is None:
        raise ValueError("Could not find a Conv2D layer in backbone.")

    grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[backbone.get_layer(last_conv).output, backbone.output]
    )

    # Use only the middle frame of the sequence for the heatmap
    mid = SEQ_LEN // 2
    single_frame = frame_seq[0, mid:mid+1]  # (1, 224, 224, 3)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(single_frame)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)           # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
    conv_outputs = conv_outputs[0]                       # (h, w, C)

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), class_idx, predictions.numpy()[0]


def overlay_heatmap(frame_bgr, heatmap, alpha=0.45):
    """Resize heatmap to frame size and overlay as a coloured mask."""
    h, w = frame_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, colored, alpha, 0)


def run(video_path, model_path, out_path, n_frames=8):
    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    while len(raw_frames) < SEQ_LEN * 2:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        raw_frames.append(resized)
    cap.release()

    if len(raw_frames) < SEQ_LEN:
        raise ValueError("Video too short for Grad-CAM analysis.")

    # Build sequence
    rgb_seq = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
               for f in raw_frames[:SEQ_LEN]]
    seq = np.expand_dims(np.array(rgb_seq), axis=0)  # (1,16,224,224,3)

    heatmap, pred_idx, probs = get_gradcam_heatmap(model, seq)
    label = CLASSES[pred_idx]
    conf  = probs[pred_idx]

    # Visualise: original | heatmap overlay
    mid_frame = raw_frames[SEQ_LEN // 2]
    overlay   = overlay_heatmap(mid_frame, heatmap)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original frame")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Grad-CAM  →  {label}  ({conf*100:.1f}%)")
    axes[1].axis("off")

    # Confidence bar
    ax_bar = fig.add_axes([0.1, 0.02, 0.8, 0.04])
    ax_bar.barh([0, 1, 2], probs, color=["green", "orange", "red"], height=0.6)
    ax_bar.set_yticks([0, 1, 2])
    ax_bar.set_yticklabels(CLASSES, fontsize=9)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Confidence")

    plt.suptitle("Grad-CAM Explainability", fontsize=13, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓  Grad-CAM saved → {out_path}")
    print(f"   Prediction: {label}  |  Confidence: {conf*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  required=True)
    ap.add_argument("--model",  default="models/cnn_lstm_best.h5")
    ap.add_argument("--out",    default="outputs/gradcam.png")
    args = ap.parse_args()
    run(args.video, args.model, args.out)
