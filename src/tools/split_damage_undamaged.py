import os
import shutil
import csv
import random

# Define datapaths
SRC_ROOT = "data/original_data/data1a"
DST_ROOT = "data/car_classification_resnet_data"

SPLITS = ["training", "validation", "testing"]

SRC_CLASSES = {
    "00-damage": 1,
    "01-whole": 0
}

DST_CLASSES = {
    1: "damaged",
    0: "undamaged"
}

CSV_PATH = os.path.join(DST_ROOT, "labels.csv")
RANDOM_SEED = 1234

random.seed(RANDOM_SEED)

for split in SPLITS:
    for cls in DST_CLASSES.values():
        os.makedirs(os.path.join(DST_ROOT, split, cls), exist_ok=True)


# Collect all images
all_items = []

for orig_split in os.listdir(SRC_ROOT):
    split_path = os.path.join(SRC_ROOT, orig_split)
    if not os.path.isdir(split_path):
        continue

    for src_class, label in SRC_CLASSES.items():
        class_dir = os.path.join(split_path, src_class)
        if not os.path.exists(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                all_items.append({
                    "src_path": os.path.join(class_dir, fname),
                    "label": label,
                    "class": src_class
                })

print(f"Total images found: {len(all_items)}")


# Splitting proces
random.shuffle(all_items)

n_total = len(all_items)
n_train = int(0.7 * n_total)
n_val = int(0.2 * n_total)

splits = {
    "training": all_items[:n_train],
    "validation": all_items[n_train:n_train + n_val],
    "testing": all_items[n_train + n_val:]
}


# Build the CSV
rows = [("image_path", "data_split_to", "label")]

for split, items in splits.items():
    for item in items:
        label = item["label"]
        cls_name = DST_CLASSES[label]

        fname = os.path.basename(item["src_path"])
        new_name = f"{split}_{item['class']}_{fname}"

        dst_path = os.path.join(
            DST_ROOT,
            split,
            cls_name,
            new_name
        )

        shutil.copy2(item["src_path"], dst_path)

        dst_path = dst_path.replace("\\", "/")
        rows.append((f"../../{dst_path}", split, label))


with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)


print("Processing complete.")
print(f"Dataset saved to: {DST_ROOT}")
print(f"CSV saved to: {CSV_PATH}")
print(f"Split ratio → Train: {len(splits['training'])}, "
      f"Val: {len(splits['validation'])}, "
      f"Test: {len(splits['testing'])}")
