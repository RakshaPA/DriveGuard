import os
import shutil

SOURCE_DIR = r"C:\Users\Raksha\Downloads\dl_package-images\Distracted Driving.v1i.coco\train"
DEST_BASE  = "data/raw_images"

DIRS = {
    "normal":      os.path.join(DEST_BASE, "normal"),
    "distracted":  os.path.join(DEST_BASE, "distracted"),
    "phone_usage": os.path.join(DEST_BASE, "phone_usage"),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

NORMAL_KW     = ["safe", "normal", "c0", "forward", "attentive"]
PHONE_KW      = ["phone", "call", "c2", "c4", "talking-phone",
                  "talkingphone", "calling"]
DISTRACTED_KW = ["distract", "text", "yawn", "sleep", "drowsy",
                  "drink", "eat", "radio", "reach", "hair",
                  "makeup", "passenger", "talk", "c1", "c3",
                  "c5", "c6", "c7", "c8", "c9", "fatigue",
                  "tired", "looking", "away", "inattentive"]

count     = {"normal": 0, "distracted": 0, "phone_usage": 0, "skipped": 0}
skipped   = []

all_files = [f for f in os.listdir(SOURCE_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"Total images found: {len(all_files)}")

for file in all_files:
    fname = file.lower()
    dest_key = None

    if any(kw in fname for kw in PHONE_KW):
        dest_key = "phone_usage"
    elif any(kw in fname for kw in NORMAL_KW):
        dest_key = "normal"
    elif any(kw in fname for kw in DISTRACTED_KW):
        dest_key = "distracted"
    else:
        count["skipped"] += 1
        skipped.append(file)
        continue

    shutil.copy(
        os.path.join(SOURCE_DIR, file),
        os.path.join(DIRS[dest_key], file)
    )
    count[dest_key] += 1

print("\n✅ Sorting complete!")
print(f"  Normal      : {count['normal']}")
print(f"  Distracted  : {count['distracted']}")
print(f"  Phone Usage : {count['phone_usage']}")
print(f"  Skipped     : {count['skipped']}")

if skipped:
    print(f"\nSkipped sample names (first 15):")
    for f in skipped[:15]:
        print("  ", f)
    print("\n→ Add keywords from skipped names to fix them")