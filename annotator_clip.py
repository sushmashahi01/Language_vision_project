import torch
from PIL import Image
from torchvision import transforms
import clip
from pathlib import Path
import pandas as pd

# === Config ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FOLDER = Path("./vg_images")  # folder with images
OUTPUT_FILE = "annotated_with_objects_clip.csv"

# Candidate boxes (example: you can use full image or precomputed boxes)
# For simplicity, we'll use the full image as one box here
def get_default_box(image):
    width, height = image.size
    return [[0, 0, width, height]]
# Load candidate names dynamically
df_gt = pd.read_csv("manynames-en.tsv", sep="\t")
object_names = sorted(df_gt["topname"].dropna().unique().tolist())
print("total_object name " + str(len(object_names)))
# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# Convert image path to tensor
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor_transform = transforms.ToTensor()
    return tensor_transform(image), image

# Detect object name
def detect_object_name(model, processor, image_pt, boxes, object_names, device=DEVICE):
    import torch.nn.functional as F
    height, width = image_pt.shape[-2:]
    max_similarity = -float('inf')
    detected_object = None

    # Encode all object names
    tokenized_queries = clip.tokenize(object_names).to(device)
    text_features = model.encode_text(tokenized_queries)
    norm_text_features = F.normalize(text_features, p=2, dim=-1)

    model.eval()

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        x_max, y_max = min(width, x_max), min(height, y_max)
        if x_min > x_max or y_min > y_max:
            continue
        cropped_image = image_pt[:, y_min:y_max+1, x_min:x_max+1]
        if not torch.prod(torch.tensor(cropped_image.shape)):
            continue
        pil_crop = transforms.functional.to_pil_image(cropped_image)
        processed_crop = processor(pil_crop).unsqueeze(0).to(device)
        image_features = model.encode_image(processed_crop)
        norm_image_features = F.normalize(image_features, p=2, dim=-1)
        similarity = (norm_text_features @ norm_image_features.T).squeeze()
        top_idx = similarity.argmax().item()
        if similarity[top_idx] > max_similarity:
            max_similarity = similarity[top_idx]
            detected_object = object_names[top_idx]

    return detected_object

# Process all images in folder
results = []

for image_path in IMAGE_FOLDER.glob("*.*"):  # handles png, jpg, jpeg etc.
    try:
        image_pt, pil_image = load_image(image_path)
        boxes = get_default_box(pil_image)  # using full image as one box
        obj_name = detect_object_name(model, preprocess, image_pt.to(DEVICE), boxes, object_names, device=DEVICE)
        results.append({"vg_image_id": image_path.stem, "topname_predicted": obj_name})
        print(f"Processed {image_path.name} -> {obj_name}")
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved detected objects to {OUTPUT_FILE}")