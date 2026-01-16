import os
import shutil
import csv

# Define paths
BASE_DIR = "data/original_data"

IMAGE_DIR = os.path.join(BASE_DIR, "Roboflow_annotation/data/images")
LABEL_DIR = os.path.join(BASE_DIR, "Roboflow_annotation/data/labels")

DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
CSV_DIR = os.path.join(DATA_DIR, "dataset_csv_file")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

CSV_PATH = os.path.join(CSV_DIR, "dataset.csv")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# Define the Yolo class names for splitting
CLASS_NAMES = [
    'back_high', 'back_low', 'back_medium',
    'front_high', 'front_low', 'front_medium',
    'rear-left_high', 'rear-left_low', 'rear-left_medium',
    'rear-right_high', 'rear-right_low', 'rear-right_medium'
]


# Collect all image paths
def collect_images(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file.lower())
            if ext in IMG_EXT:
                image_paths.append(os.path.join(root, file))
    return image_paths

all_images = sorted(collect_images(IMAGE_DIR))
print(f"Total images found: {len(all_images)}")


# Main procesing
counter = 0

with open(CSV_PATH, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "filepath",
        "damage_location",
        "damage_severity",
        "caption"
    ])

    for img_path in all_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABEL_DIR, base_name + ".txt")

        # Skip if label missing
        if not os.path.exists(label_path):
            print(f"[WARN] Missing label for {base_name}, skipping")
            continue

        # Read YOLO label (first object only)
        with open(label_path, "r") as f:
            line = f.readline().strip()

        if not line:
            print(f"[WARN] Empty label file: {label_path}")
            continue

        class_id = int(line.split()[0])
        class_name = CLASS_NAMES[class_id]

        # Split label name
        damage_location, damage_severity = class_name.split("_")

        # Generate caption
        caption = (
            f"A car with {damage_location} damage "
            f"of {damage_severity} severity"
        )

        # Copy image
        _, ext = os.path.splitext(img_path)
        new_name = f"{counter:04d}{ext}"
        final_img_path = os.path.join(IMG_DIR, new_name)

        shutil.copy2(img_path, final_img_path)

        final_img_path = final_img_path.replace("\\", "/")

        # Write CSV row
        writer.writerow([
            f"../../{final_img_path}",
            damage_location,
            damage_severity,
            caption
        ])

        counter += 1

print("Done!")
print(f"{counter} images copied into {IMG_DIR}")
print(f"CSV saved at {CSV_PATH}")
