"""
preprocess.py  —  Image dataset → sequences for CNN-LSTM
Reads from: data/raw_images/normal/  distracted/  phone_usage/
Saves to:   data/sequences/
Run: python src/preprocess.py
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
SEQ_LEN  = 16
DATA_DIR = "data/raw_images"
OUT_DIR  = "data/sequences"

CLASS_MAP = {
    "normal":      0,
    "distracted":  1,
    "phone_usage": 2,
}
CLASSES = ["Normal", "Distracted", "Phone Usage"]


def load_and_build(data_dir):
    """
    Each image is stored as a single frame. The 16-frame sequence is created
    later during training/evaluation by repeating the image.
    Augmentations are still applied at the image level.
    """
    images, labels = [], []

    for folder, label in CLASS_MAP.items():
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"[WARN] {folder_path} not found — skipping")
            continue

        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if len(files) == 0:
            print(f"[WARN] {folder} has 0 images — check your sort_dataset.py")
            continue

        print(f"  {folder}: {len(files)} images → class '{CLASSES[label]}'")

        for fname in tqdm(files, desc=folder, leave=False):
            img_path = os.path.join(folder_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(label)

            flipped = np.fliplr(img)
            images.append(flipped)
            labels.append(label)

            bright = np.clip(img * np.random.uniform(0.75, 1.25), 0, 1)
            images.append(bright)
            labels.append(label)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def build():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nLoading images from '{DATA_DIR}' ...")
    X, y = load_and_build(DATA_DIR)

    if len(X) == 0:
        print("\n[ERROR] No images loaded!")
        print("  1. Run sort_dataset.py first")
        print("  2. Check that data/raw_images/normal/ distracted/ phone_usage/ have images")
        return

    total = len(X)
    print(f"\nTotal images built    : {total}")
    print(f"  (3x augmentation applied — original + flip + brightness)")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {int((y == i).sum())} images")

    if total < 10:
        print("\n[ERROR] Too few sequences. Need at least 10 total (preferably 100+).")
        print("  Make sure your image folders have images.")
        return

    # Shuffle
    idx = np.random.permutation(total)
    X, y = X[idx], y[idx]

    # Stratified 80/20 split
    # If too few samples, fall back to simple split
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        print("[WARN] Not enough samples per class for stratified split — using random split")
        split = max(1, int(0.8 * total))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

    np.save(f"{OUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUT_DIR}/X_val.npy",   X_val)
    np.save(f"{OUT_DIR}/y_val.npy",   y_val)

    print(f"\n✓ Saved to '{OUT_DIR}/'")
    print(f"  Train sequences : {len(X_train)}")
    print(f"  Val sequences   : {len(X_val)}")
    print(f"  Sequence shape  : {X_train[0].shape}")
    print(f"\nNext step → python src/train.py --epochs 30 --batch 8")


if __name__ == "__main__":
    build()