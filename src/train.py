"""
train.py  —  Two-phase CNN-LSTM training
Run: python src/train.py --epochs 30 --batch 8
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

CLASSES = ["Normal", "Distracted", "Phone Usage"]


def normalize_images(X):
    return X.astype(np.float32) / 255.0


def load_data(seq_dir):
    X_train = np.load(os.path.join(seq_dir, "X_train.npy"))
    y_train = np.load(os.path.join(seq_dir, "y_train.npy"))
    X_val   = np.load(os.path.join(seq_dir, "X_val.npy"))
    y_val   = np.load(os.path.join(seq_dir, "y_val.npy"))
    X_train = normalize_images(X_train)
    X_val   = normalize_images(X_val)
    print(f"Train: {X_train.shape}  |  Val: {X_val.shape}  |  dtype: {X_train.dtype}")
    return X_train, y_train, X_val, y_val


def build_model(seq_len=16, img_size=224, n_classes=3,
                lstm_units=256, dropout=0.5, fine_tune=0):
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    for layer in backbone.layers[:-fine_tune] if fine_tune > 0 else backbone.layers:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(seq_len, img_size, img_size, 3))
    x = tf.keras.layers.TimeDistributed(backbone, name="feature_extractor")(inputs)
    x = tf.keras.layers.LSTM(lstm_units, name="lstm")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def repeat_image_to_sequence(image, label):
    seq = tf.repeat(tf.expand_dims(image, axis=0), repeats=16, axis=0)
    return seq, label


def augment(image, label):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def plot_history(history, extra_history=None, path="outputs/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0B0F1A")
    for ax in axes:
        ax.set_facecolor("#131929")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1C2438")
        ax.tick_params(colors="#6B7A99")

    acc  = history.history["accuracy"]
    vacc = history.history["val_accuracy"]
    loss  = history.history["loss"]
    vloss = history.history["val_loss"]

    if extra_history:
        acc  += extra_history.history["accuracy"]
        vacc += extra_history.history["val_accuracy"]
        loss  += extra_history.history["loss"]
        vloss += extra_history.history["val_loss"]

    axes[0].plot(acc,  color="#00C896", label="Train accuracy")
    axes[0].plot(vacc, color="#FF4757", label="Val accuracy", linestyle="--")
    axes[0].set_title("Accuracy", color="#E8EDF5")
    axes[0].set_ylabel("Accuracy", color="#6B7A99")
    axes[0].legend(facecolor="#1C2438", labelcolor="#E8EDF5")
    axes[0].grid(True, alpha=0.15)

    axes[1].plot(loss,  color="#00C896", label="Train loss")
    axes[1].plot(vloss, color="#FF4757", label="Val loss", linestyle="--")
    axes[1].set_title("Loss", color="#E8EDF5")
    axes[1].set_ylabel("Loss", color="#6B7A99")
    axes[1].legend(facecolor="#1C2438", labelcolor="#E8EDF5")
    axes[1].grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓  Training curves saved → {path}")


def train(seq_dir, epochs, batch_size, phase1_epochs):
    X_train, y_train, X_val, y_val = load_data(seq_dir)

    # Auto-fix batch size if dataset too small
    if len(X_train) < batch_size:
        batch_size = max(1, len(X_train))
        print(f"[INFO] Batch size reduced to {batch_size} (small dataset)")

    if len(X_val) == 0:
        print("[ERROR] Validation set is empty. Re-run preprocess.py")
        return

    # Class weights
    unique = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=unique, y=y_train)
    class_weight = {int(k): float(v) for k, v in zip(unique, weights)}
    print(f"Class weights: { {CLASSES[k]: round(v,2) for k,v in class_weight.items()} }")

    # Datasets
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train))
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .map(repeat_image_to_sequence, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .map(repeat_image_to_sequence, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/cnn_lstm_best.keras",
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.CSVLogger("outputs/training_log.csv"),
    ]

    # ── Phase 1: frozen backbone ───────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f" Phase 1: training head ({phase1_epochs} epochs, backbone frozen)")
    print(f"{'='*50}")
    model = build_model(fine_tune=0)
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # ── Phase 2: fine-tune top layers ─────────────────────────────────────
    remaining = epochs - phase1_epochs
    hist2 = None
    if remaining > 0:
        print(f"\n{'='*50}")
        print(f" Phase 2: fine-tuning top 20 layers ({remaining} epochs)")
        print(f"{'='*50}")
        # Unfreeze top 20 backbone layers
        feature_extractor = model.get_layer("feature_extractor").layer
        for layer in feature_extractor.layers[-20:]:
            layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=remaining,
            initial_epoch=phase1_epochs,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )

    # Save final model
    model.save("models/cnn_lstm_best.keras")
    print("\n✓  Model saved → models/cnn_lstm_best.keras")

    plot_history(hist1, hist2)

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\n✓  Final val accuracy : {acc*100:.2f}%")
    print(f"   Final val loss     : {loss:.4f}")

    # ── Minimum accuracy warning ───────────────────────────────────────────
    if acc < 0.5:
        print("\n[WARN] Accuracy is low. Possible reasons:")
        print("  1. Too few images — need 200+ per class ideally")
        print("  2. Images not sorted correctly — check data/raw_images/ folders")
        print("  3. Run: python src/preprocess.py to rebuild sequences")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir",  default="data/sequences")
    ap.add_argument("--epochs",   type=int, default=30)
    ap.add_argument("--batch",    type=int, default=8)
    ap.add_argument("--phase1",   type=int, default=10)
    args = ap.parse_args()
    train(args.seq_dir, args.epochs, args.batch, args.phase1)