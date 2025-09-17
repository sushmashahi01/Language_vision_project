import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

# === Config ===
INPUT_FILE = "manynames-en.tsv"  # replace with your file
OUTPUT_DIR = Path("vg_images")  # folder to save images
#MAX_ROWS = 2000  # limit to 2000 rows

# Create output folder if not exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(INPUT_FILE, sep="\t")  # use sep="," if it's a CSV with commas
MAX_ROWS = len(df)
print(MAX_ROWS)
# Filter first 2000 rows
subset = df.head(MAX_ROWS)

# Iterate and download
for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Downloading"):
    img_id = row["vg_image_id"]
    url = row["link_mn"]
    save_path = OUTPUT_DIR / f"{img_id}.png"

    # Skip if already downloaded
    if save_path.exists():
        continue

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        print(f"Failed to download {url} (id {img_id}): {e}")
