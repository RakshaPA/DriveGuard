"""
train.py  —  Train the CNN-LSTM driving behavior model
───────────────────────────────────────────────────────
Usage:
    python src/train.py --seq_dir data/sequences --epochs 30 --batch 8

Two-phase training:
    Phase 1  — Backbone frozen,   train LSTM + head  (fast, 10 epochs)
    Phase 2  — Top layers unfrozen, fine-tune end-to-end (slow, remaining epochs)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from model import build_cnn_lstm, CLASSES

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ── Data loader ───────────────────────────────────────────────────────────────

def load_data(seq_dir):
    X_train = np.load(os.path.join(seq_dir, "X_train.npy"))
    y_train = np.load(os.path.join(seq_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(seq_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(seq_dir, "y_val.npy"))
    print(f"Train: {X_train.shape}  |  Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


def make_tf_dataset(X, y, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── Augmentation (applied per-frame during training) ──────────────────────────

def augment_sequence(seq, label):
    """Random horizontal flip + brightness jitter applied to the whole sequence."""
    if tf.random.uniform(()) > 0.5:
        seq = tf.image.flip_left_right(seq)
    seq = tf.image.random_brightness(seq, max_delta=0.1)
    seq = tf.clip_by_value(seq, 0.0, 1.0)
    return seq, label


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(
            "models/cnn_lstm_best.h5",
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger("outputs/training_log.csv"),
    ]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_history(history, save_path="outputs/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val accuracy")
    axes[0].set_title("Accuracy over epochs")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train loss")
    axes[1].plot(history.history["val_loss"], label="Val loss")
    axes[1].set_title("Loss over epochs")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✓  Training curves saved → {save_path}")


# ── Main training loop ────────────────────────────────────────────────────────

def train(seq_dir, epochs, batch_size, phase1_epochs):
    X_train, y_train, X_val, y_val = load_data(seq_dir)

    # Class weights to handle imbalance
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(weights))
    print(f"Class weights: { {CLASSES[k]: round(v,2) for k,v in class_weight.items()} }")

    # Build augmented training dataset
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train))
        .map(augment_sequence, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = make_tf_dataset(X_val, y_val, batch_size)

    model = build_cnn_lstm(fine_tune_layers=0)  # backbone fully frozen

    # ── Phase 1: train head + LSTM only ───────────────────────────────────────
    print(f"\n{'='*50}")
    print(f" Phase 1: training head ({phase1_epochs} epochs, backbone frozen)")
    print(f"{'='*50}")
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=phase1_epochs,
        class_weight=class_weight,
        callbacks=get_callbacks()
    )

    # ── Phase 2: fine-tune top MobileNetV2 layers ─────────────────────────────
    remaining = epochs - phase1_epochs
    if remaining > 0:
        print(f"\n{'='*50}")
        print(f" Phase 2: fine-tuning top 20 layers ({remaining} epochs)")
        print(f"{'='*50}")
        # Unfreeze top 20 backbone layers
        for layer in model.get_layer("feature_extractor").layer.layers[-20:]:
            layer.trainable = True
        # Lower LR for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        hist2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=remaining,
            initial_epoch=phase1_epochs,
            class_weight=class_weight,
            callbacks=get_callbacks()
        )
        # Merge histories for plotting
        for k in hist1.history:
            hist1.history[k].extend(hist2.history.get(k, []))

    model.save("models/cnn_lstm.h5")
    print("\n✓  Final model saved → models/cnn_lstm.h5")

    plot_history(hist1)

    # Final evaluation
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\n✓  Final val accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir",      default="data/sequences")
    ap.add_argument("--epochs",       type=int, default=30)
    ap.add_argument("--batch",        type=int, default=8)
    ap.add_argument("--phase1",       type=int, default=10,
                    help="Epochs for frozen-backbone phase")
    args = ap.parse_args()
    train(args.seq_dir, args.epochs, args.batch, args.phase1)
