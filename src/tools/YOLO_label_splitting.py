import os
import shutil
import yaml
from pathlib import Path
from collections import Counter
import numpy as np

# Define datapaths
SRC_ROOT = "data/dataset_yaml_file"
SRC_IMAGES = os.path.join(SRC_ROOT, "images")
SRC_LABELS = os.path.join(SRC_ROOT, "labels")

DST_LOCATION = "data/YOLO_DATASET_LOCATION"
DST_SEVERITY = "data/YOLO_DATASET_SEVERITY"

SPLITS = ["train", "validation", "test"]

LOCATION_NAMES = ["front", "back", "rear-left", "rear-right"]
SEVERITY_NAMES = ["low", "medium", "high"]

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# define some helper functions
def make_dirs(root):
    for split in SPLITS:
        os.makedirs(f"{root}/images/{split}", exist_ok=True)
        os.makedirs(f"{root}/labels/{split}", exist_ok=True)


def compute_class_weights(label_dir: Path, num_classes: int):
    counter = Counter()

    for txt in label_dir.glob("*.txt"):
        with open(txt) as f:
            for line in f:
                cls = int(line.strip().split()[0])
                counter[cls] += 1

    freqs = np.zeros(num_classes, dtype=np.float32)
    for cls_id, count in counter.items():
        freqs[cls_id] = count

    # Avoid zero division
    freqs[freqs == 0] = 1.0

    mean_freq = freqs.mean()
    weights = mean_freq / freqs

    # Normalize for stability
    weights = weights / weights.mean()

    return weights.round(4).tolist()


def add_class_weights_to_yaml(yaml_path: str, weights: list):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    data["class_weights"] = weights

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"✅ Added class_weights to {yaml_path}")
    print(f"   Weights: {weights}")


make_dirs(DST_LOCATION)
make_dirs(DST_SEVERITY)


# Process the images and the labels
for split in SPLITS:
    label_dir = Path(SRC_LABELS) / split
    image_dir = Path(SRC_IMAGES) / split

    for label_file in label_dir.glob("*.txt"):
        with open(label_file) as f:
            lines = f.readlines()

        loc_lines, sev_lines = [], []

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 6:
                continue

            loc_cls, sev_cls, xc, yc, bw, bh = parts
            loc_lines.append(f"{loc_cls} {xc} {yc} {bw} {bh}\n")
            sev_lines.append(f"{sev_cls} {xc} {yc} {bw} {bh}\n")

        # Write labels
        (Path(DST_LOCATION) / "labels" / split / label_file.name).write_text("".join(loc_lines))
        (Path(DST_SEVERITY) / "labels" / split / label_file.name).write_text("".join(sev_lines))

        # Copy image
        for ext in IMAGE_EXTS:
            img_path = image_dir / f"{label_file.stem}{ext}"
            if img_path.exists():
                shutil.copy(img_path, Path(DST_LOCATION) / "images" / split / img_path.name)
                shutil.copy(img_path, Path(DST_SEVERITY) / "images" / split / img_path.name)
                break


# Create the YAML files
location_yaml_path = f"{DST_LOCATION}/data_location.yaml"
severity_yaml_path = f"{DST_SEVERITY}/data_severity.yaml"

location_yaml = {
    "path": f"../../{DST_LOCATION}",
    "train": "images/train",
    "val": "images/validation",
    "test": "images/test",
    "nc": len(LOCATION_NAMES),
    "names": LOCATION_NAMES
}

severity_yaml = {
    "path": f"../../{DST_SEVERITY}",
    "train": "images/train",
    "val": "images/validation",
    "test": "images/test",
    "nc": len(SEVERITY_NAMES),
    "names": SEVERITY_NAMES
}

with open(location_yaml_path, "w") as f:
    yaml.dump(location_yaml, f, sort_keys=False)

with open(severity_yaml_path, "w") as f:
    yaml.dump(severity_yaml, f, sort_keys=False)


# Compute the class weights and add it to the YAML file
location_weights = compute_class_weights(
    Path(DST_LOCATION) / "labels" / "train",
    num_classes=len(LOCATION_NAMES)
)

severity_weights = compute_class_weights(
    Path(DST_SEVERITY) / "labels" / "train",
    num_classes=len(SEVERITY_NAMES)
)

add_class_weights_to_yaml(location_yaml_path, location_weights)
add_class_weights_to_yaml(severity_yaml_path, severity_weights)


print("✅ YOLO datasets successfully created with class weights")
print(f"📁 Location dataset: {DST_LOCATION}")
print(f"📁 Severity dataset: {DST_SEVERITY}")
