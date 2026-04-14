"""
evaluate.py  —  Evaluate trained model on validation set
─────────────────────────────────────────────────────────
Usage:
    python src/evaluate.py --seq_dir data/sequences --model models/cnn_lstm_best.h5
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)

CLASSES = ["Normal", "Distracted", "Phone Usage"]
os.makedirs("outputs", exist_ok=True)


def evaluate(seq_dir, model_path):
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    X_val = np.load(os.path.join(seq_dir, "X_val.npy"))
    y_val = np.load(os.path.join(seq_dir, "y_val.npy"))
    print(f"Validation set: {X_val.shape[0]} sequences")

    # Predict in batches to avoid OOM
    probs = model.predict(X_val, batch_size=4, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    # ── Classification report ──────────────────────────────────────────────
    print("\n── Classification Report ─────────────────────────────")
    print(classification_report(y_val, y_pred, target_names=CLASSES))

    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(
            tf.keras.utils.to_categorical(y_val, len(CLASSES)),
            probs, multi_class="ovr"
        )
        print(f"  ROC-AUC (OvR): {auc:.4f}")
    except Exception as e:
        print(f"  ROC-AUC not computed: {e}")

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_val, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
    axes[0].set_title("Confusion matrix (counts)")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
    axes[1].set_title("Confusion matrix (normalised)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    out_path = "outputs/confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n✓  Confusion matrix saved → {out_path}")

    # ── Per-class accuracy bar chart ───────────────────────────────────────
    per_class_acc = cm_norm.diagonal()
    fig2, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASSES, per_class_acc,
                  color=["#4caf50", "#ff9800", "#f44336"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy"); ax.set_title("Per-class accuracy")
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val*100:.1f}%", ha="center", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out2 = "outputs/per_class_accuracy.png"
    plt.savefig(out2, dpi=150)
    print(f"✓  Per-class accuracy saved → {out2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", default="data/sequences")
    ap.add_argument("--model",   default="models/cnn_lstm_best.h5")
    args = ap.parse_args()
    evaluate(args.seq_dir, args.model)
