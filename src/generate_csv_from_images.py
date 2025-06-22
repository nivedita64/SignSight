import os
import pandas as pd
from tqdm import tqdm

DATASET_PATH = "../dataset/processed_combine_asl_dataset"
OUTPUT_CSV = "../dataset/asl_combined_AtoL.csv"
INCLUDE_LABELS = list("abcdefghijkl")  # Only A–L

data = []

for label in INCLUDE_LABELS:
    folder_path = os.path.join(DATASET_PATH, label)
    print(f"Processing: {label.upper()}")
    if not os.path.exists(folder_path):
        continue
    for file_name in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, file_name)
        if os.path.isfile(img_path):
            data.append({"image_path": img_path, "label": label.upper()})

df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ CSV saved at {OUTPUT_CSV}")