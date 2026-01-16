import os
import shutil
import random
from pathlib import Path
from PIL import Image
import yaml


# Define the configurations
ORIG_IMG_DIR = "data/original_data/Roboflow_annotation/data/images"
ORIG_LBL_DIR = "data/original_data/Roboflow_annotation/data/labels"

TEMP_RENAMED_DIR = "data/misc/renamed_dataset"
FINAL_OUTPUT_DIR = "data/dataset_yaml_file"
MISC_DIR = "data/misc"

random.seed(1234)


# Define classes and helper function to encode label to number and sanitize YOLO labels
COMBINED_NAMES = [
    'back_high', 'back_low', 'back_medium',
    'front_high', 'front_low', 'front_medium',
    'rear-left_high', 'rear-left_low', 'rear-left_medium',
    'rear-right_high', 'rear-right_low', 'rear-right_medium'
]

LOCATION_NAMES = ['front', 'back', 'rear-left', 'rear-right']
SEVERITY_NAMES = ['low', 'medium', 'high']

LOC_MAP = {name: idx for idx, name in enumerate(LOCATION_NAMES)}
SEV_MAP = {name: idx for idx, name in enumerate(SEVERITY_NAMES)}

def decode_combined_class(cls_id: int):
    name = COMBINED_NAMES[cls_id]
    loc, sev = name.split("_")
    return LOC_MAP[loc], SEV_MAP[sev]

def sanitize_label(parts, img_w, img_h):
    try:
        nums = list(map(float, parts))
    except ValueError:
        return None

    if len(nums) < 5:
        return None

    cls = int(nums[0])
    xc, yc, w, h = nums[-4:]

    if xc > 1: xc /= img_w
    if yc > 1: yc /= img_h
    if w > 1:  w  /= img_w
    if h > 1:  h  /= img_h

    if w <= 0 or h <= 0:
        return None
    if not (0 <= xc <= 1 and 0 <= yc <= 1):
        return None

    return cls, xc, yc, w, h


# Define the helper functions
def rename_files():
    shutil.rmtree(TEMP_RENAMED_DIR, ignore_errors=True)
    os.makedirs(f"{TEMP_RENAMED_DIR}/images", exist_ok=True)
    os.makedirs(f"{TEMP_RENAMED_DIR}/labels", exist_ok=True)

    imgs = sorted(Path(ORIG_IMG_DIR).glob("*.jpg"))
    print(f"Found {len(imgs)} images")

    for idx, img_path in enumerate(imgs):
        shutil.copy(img_path, f"{TEMP_RENAMED_DIR}/images/{idx:04d}.jpg")

        lbl_path = Path(ORIG_LBL_DIR) / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.copy(lbl_path, f"{TEMP_RENAMED_DIR}/labels/{idx:04d}.txt")


def split_and_rewrite():
    shutil.rmtree(FINAL_OUTPUT_DIR, ignore_errors=True)

    for split in ["train", "validation", "test"]:
        os.makedirs(f"{FINAL_OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{FINAL_OUTPUT_DIR}/labels/{split}", exist_ok=True)

    imgs = sorted(Path(f"{TEMP_RENAMED_DIR}/images").glob("*.jpg"))
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(0.7 * n)
    n_val = int(0.2 * n)

    split_map = {
        "train": imgs[:n_train],
        "validation": imgs[n_train:n_train + n_val],
        "test": imgs[n_train + n_val:]
    }

    for split, items in split_map.items():
        for img_path in items:
            shutil.copy(
                img_path,
                f"{FINAL_OUTPUT_DIR}/images/{split}/{img_path.name}"
            )

            lbl_src = Path(TEMP_RENAMED_DIR) / "labels" / f"{img_path.stem}.txt"
            if not lbl_src.exists():
                continue

            img = Image.open(img_path)
            w, h = img.size

            out_lbl = Path(FINAL_OUTPUT_DIR) / "labels" / split / f"{img_path.stem}.txt"

            with open(lbl_src) as f_in, open(out_lbl, "w") as f_out:
                for line_idx, raw in enumerate(f_in):
                    parts = raw.strip().split()
                    fixed = sanitize_label(parts, w, h)

                    if fixed is None:
                        print(
                            f"⚠ [SKIP] {img_path.name} line {line_idx}: "
                            f"malformed input → '{raw.strip()}'"
                        )
                        continue

                    cls, xc, yc, bw, bh = fixed
                    loc_id, sev_id = decode_combined_class(cls)

                    output_fields = [
                        loc_id,
                        sev_id,
                        xc,
                        yc,
                        bw,
                        bh
                    ]

                    if len(output_fields) != 6:
                        print(
                            f"🚨 [WARNING] {img_path.name} line {line_idx}: "
                            f"expected 6 fields, got {len(output_fields)} → {output_fields}"
                        )
                        continue

                    f_out.write(
                        f"{loc_id} {sev_id} "
                        f"{xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
                    )

    print("labels rewritten with validation")


def create_yaml_files():
    base = "data/dataset_yaml_file/images"

    location_yaml = {
        "train": f"{base}/train",
        "val": f"{base}/validation",
        "test": f"{base}/test",
        "nc": len(LOCATION_NAMES),
        "names": LOCATION_NAMES
    }

    severity_yaml = {
        "train": f"{base}/train",
        "val": f"{base}/validation",
        "test": f"{base}/test",
        "nc": len(SEVERITY_NAMES),
        "names": SEVERITY_NAMES
    }

    with open(f"{FINAL_OUTPUT_DIR}/data_location.yaml", "w") as f:
        yaml.dump(location_yaml, f, sort_keys=False)

    with open(f"{FINAL_OUTPUT_DIR}/data_severity.yaml", "w") as f:
        yaml.dump(severity_yaml, f, sort_keys=False)

    with open(f"data/misc/renamed_dataset/data_location.yaml", "w") as f:
        yaml.dump(location_yaml, f, sort_keys=False)

    with open(f"data/misc/renamed_dataset/data_severity.yaml", "w") as f:
        yaml.dump(severity_yaml, f, sort_keys=False)

    print("YAML files created")



if __name__ == "__main__":
    rename_files()
    split_and_rewrite()
    create_yaml_files()

    print("\nDataset prepared successfully")
    print(f"Output: {FINAL_OUTPUT_DIR}")

    shutil.rmtree(MISC_DIR, ignore_errors=True)
    print("Temporary misc folder deleted")