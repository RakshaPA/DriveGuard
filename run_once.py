# run_once.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import os

os.makedirs("models", exist_ok=True)

model = Sequential([
    TimeDistributed(MobileNetV2(include_top=False, weights=None, input_shape=(224,224,3))),
    TimeDistributed(GlobalAveragePooling2D()),
    LSTM(64),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save("models/cnn_lstm_best.h5")
print("Placeholder model saved.")