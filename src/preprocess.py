"""
preprocess.py  —  Extract frame sequences from driving videos
─────────────────────────────────────────────────────────────
Usage:
    python src/preprocess.py --data_dir data/raw_videos --out_dir data/sequences

Expected folder layout:
    data/raw_videos/
        normal/         video1.mp4  video2.mp4 ...
        distracted/     ...
        phone_usage/    ...
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

IMG_SIZE  = 224   # MobileNetV2 input
SEQ_LEN   = 16    # frames per sequence
FRAME_STEP = 2    # sample every Nth frame
CLASSES   = ["normal", "distracted", "phone_usage"]


def extract_frames(video_path):
    """Read all frames, resize to 224x224, normalize to [0,1]."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()
    return frames


def frames_to_sequences(frames):
    """Sliding window → shape (N, SEQ_LEN, 224, 224, 3)."""
    sampled = frames[::FRAME_STEP]
    seqs = []
    for i in range(0, len(sampled) - SEQ_LEN + 1, SEQ_LEN // 2):
        seq = sampled[i : i + SEQ_LEN]
        if len(seq) == SEQ_LEN:
            seqs.append(seq)
    return np.array(seqs, dtype=np.float32)


def build_dataset(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_sequences, all_labels = [], []

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Folder not found: {class_dir} — skipping.")
            continue

        videos = [f for f in os.listdir(class_dir)
                  if f.lower().endswith((".mp4", ".avi", ".mov"))]
        print(f"\n[{class_name}]  {len(videos)} videos found")

        for vf in tqdm(videos, desc=class_name):
            frames = extract_frames(os.path.join(class_dir, vf))
            if len(frames) < SEQ_LEN:
                continue
            seqs = frames_to_sequences(frames)
            all_sequences.append(seqs)
            all_labels.extend([label_idx] * len(seqs))

    if not all_sequences:
        print("[ERROR] No sequences built. Check folder structure.")
        return

    X = np.concatenate(all_sequences, axis=0)
    y = np.array(all_labels, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # 80/20 split
    split = int(0.8 * len(X))
    np.save(os.path.join(out_dir, "X_train.npy"), X[:split])
    np.save(os.path.join(out_dir, "y_train.npy"), y[:split])
    np.save(os.path.join(out_dir, "X_val.npy"),   X[split:])
    np.save(os.path.join(out_dir, "y_val.npy"),   y[split:])

    print(f"\n✓  Saved to '{out_dir}'")
    print(f"   Train: {split}  |  Val: {len(X)-split}  |  Shape: {X[0].shape}")
    print(f"   Class counts: { {CLASSES[i]: int((y==i).sum()) for i in range(len(CLASSES))} }")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw_videos")
    ap.add_argument("--out_dir",  default="data/sequences")
    args = ap.parse_args()
    build_dataset(args.data_dir, args.out_dir)
