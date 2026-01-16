import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import sys

# Define labeling options
location_options = {
    0: "front",
    1: "rear-left",
    2: "rear-right",
    3: "back"
}

severity_options = {
    0: "low",
    1: "medium",
    2: "high"
}

valid_ext = (".jpg", ".jpeg", ".JPEG", ".JPG", ".png", ".PNG")

# CSV paths
source_csv = "data/dataset_csv_file/dataset_filepath.csv"
output_csv = "data/dataset_csv_file/dataset.csv"

# Load source file list
try:
    df = pd.read_csv(source_csv)
except FileNotFoundError as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# If dataset.csv exists, load progress
if os.path.exists(output_csv):
    print("Found existing labeled dataset — merging...")
    labeled_df = pd.read_csv(output_csv)
    df = df.merge(labeled_df, on="filepath", how="left")  # preserve old progress
else:
    df["damage_location"] = ""
    df["damage_severity"] = ""

# Ask for range
print(f"Dataset contains {len(df)} images total.")
start_index = int(input("Enter START index: "))
end_index = int(input("Enter END index (exclusive): "))

# Labeling loop
for i in range(start_index, min(end_index, len(df))):

    # Skip if already labeled
    if pd.notna(df.loc[i, "damage_location"]) and df.loc[i, "damage_location"] != "":
        print(f"Skipping already labeled image [{i}] -> {df.loc[i, 'filepath']}")
        continue

    img_path = df.loc[i, "filepath"]

    if not img_path.endswith(valid_ext):
        print(f"Unsupported file type: {img_path}")
        continue

    try:
        with Image.open(img_path) as img:
            plt.imshow(img)
            plt.axis("off")
            plt.show()
    except Exception as e:
        print(f"Error opening {img_path}: {e}")
        continue

    print(f"\nLabeling Image [{i}] -> {img_path}")

    # Damage location
    print("\nDamage Location Options:")
    print(location_options)
    loc = input("Enter damage location number (or x to exit): ")
    if str(loc).lower() == "x":
        break

    loc = int(loc)

    # Damage severity
    print("\nDamage Severity Options:")
    print(severity_options)
    sev = input("Enter damage severity number (or x to exit): ")
    if str(sev).lower() == "x":
        break

    sev = int(sev)

    # Save into dataframe
    df.loc[i, "damage_location"] = location_options[loc]
    df.loc[i, "damage_severity"] = severity_options[sev]

    # Save immediately (prevent data loss)
    df.to_csv(output_csv, index=False)
    print("✔ Saved!\n----------------------")

print("\n✔ All progress saved.")