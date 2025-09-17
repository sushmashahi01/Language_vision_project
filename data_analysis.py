from pathlib import Path
from PIL import Image
import pandas as pd

# Path to your images
IMAGE_FOLDER = Path("./vg_images")

# List to store data
data = []

# Iterate over all PNG images
for img_path in IMAGE_FOLDER.glob("*.png"):
    # Read the image
    img = Image.open(img_path)

    # Get image ID (filename without .png)
    image_id = img_path.stem

    # Append to list
    data.append({"vg_image_id": image_id})

# Create DataFrame
df_sampled = pd.DataFrame(data)
df_all = pd.read_csv("manynames-en.tsv", sep="\t")
df_all["vg_image_id"]= df_all["vg_image_id"].astype(int)
df_sampled["vg_image_id"]= df_sampled["vg_image_id"].astype(int)
# Step 3: Merge DataFrames based on vg_image_id
df_merged = pd.merge(df_sampled, df_all, on="vg_image_id", how="inner")

df_merged.to_csv("sampled_images_metadata.csv",index=False)
# Show first few rows
print(df_merged.head())
print(df_merged["topname"].nunique())
print(df_merged["topname"].value_counts())

print(df_merged["domain"].nunique())
print(df_merged["domain"].value_counts())
