"""
model.py  —  MobileNetV2 + LSTM driving behavior classifier
────────────────────────────────────────────────────────────
Architecture:
    Input (16, 224, 224, 3)
        └─ TimeDistributed(MobileNetV2)  →  (16, 1280)
        └─ LSTM(256)                     →  (256,)
        └─ Dense(128, relu) + Dropout
        └─ Dense(3, softmax)             →  class probabilities
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

SEQ_LEN    = 16
IMG_SIZE   = 224
N_CLASSES  = 3
CLASSES    = ["Normal", "Distracted", "Phone Usage"]


def build_cnn_lstm(seq_len=SEQ_LEN, img_size=IMG_SIZE, n_classes=N_CLASSES,
                   lstm_units=256, dropout=0.5, fine_tune_layers=20):
    """
    Returns compiled CNN-LSTM model.

    Parameters
    ----------
    fine_tune_layers : int
        Number of top MobileNetV2 layers to unfreeze for fine-tuning.
        Set to 0 to freeze the entire backbone.
    """
    # ── Backbone ──────────────────────────────────────────────────────────────
    backbone = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"        # GlobalAveragePooling → (1280,)
    )
    # Freeze all except the top fine_tune_layers
    for layer in backbone.layers[:-fine_tune_layers]:
        layer.trainable = False

    # ── Full model ────────────────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=(seq_len, img_size, img_size, 3),
                            name="frame_sequence")

    # Apply backbone to every frame independently
    x = layers.TimeDistributed(backbone, name="feature_extractor")(inputs)
    # x shape: (batch, seq_len, 1280)

    # Capture temporal patterns across the sequence
    x = layers.LSTM(lstm_units, return_sequences=False, name="lstm")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="DrivingBehaviorCNN_LSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def load_model(path="models/cnn_lstm.h5"):
    return tf.keras.models.load_model(path)


if __name__ == "__main__":
    model = build_cnn_lstm()
    model.summary()
    print(f"\nTotal params: {model.count_params():,}")
