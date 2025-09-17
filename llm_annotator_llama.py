import os
import sys
import pandas as pd
from pathlib import Path
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import subprocess

# === Config ===
IMAGE_FOLDER = Path("./vg_images")  # folder with images
OUTPUT_FILE = "annotated_with_objects_llama.csv"
#MAX_IMAGES = 500
MODEL_NAME = "llama3.2-vision:90b"  # replace with a vision-capable model (e.g., llava:13b, qwen-vl, gemma2-vision)

# --- Check if model exists in Ollama ---
try:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if MODEL_NAME not in result.stdout:
        print(f"❌ Error: Model '{MODEL_NAME}' is not installed in Ollama.")
        sys.exit(1)
except FileNotFoundError:
    print("❌ Error: Ollama is not installed or not in PATH.")
    sys.exit(1)

# Initialize LLM
llm = Ollama(
    model=MODEL_NAME,
    #temperature=0.7,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# Function to generate and print response for a given prompt
def generate_response(prompt):
    response = llm.invoke(prompt)
    return response.strip()

df_gt = pd.read_csv("manynames-en.tsv", sep="\t")
object_names = sorted(df_gt["topname"].dropna().unique().tolist())
# Function to query LLM with an image
def describe_image(image_path: str):
    prompt = (
        f"You are an picture annotator. Your task is to identify the object highlighted in the given picture.\n"
        f"Possible objects: {object_names}\n"
        f"tell me which object do you see in this picture? {image_path}"
        f"Format your answer exactly as follows:\n"
        f"Object: <object name>\n"
        f"Explanation: <short explanation>"
    )

    return generate_response(prompt)


# Collect images
image_files = list(IMAGE_FOLDER.glob("*.png"))
#[:MAX_IMAGES]
#file= "./vg_images/2403043.png"
#describe_image(file)

# Build DataFrame to store results
results = []
for img_path in image_files:
    print(img_path.stem)
    print(f"Processing {img_path.name}...")
    desc = describe_image(img_path)
    # Parse response into object + explanation
    obj, expl = None, None
    if "Object:" in desc:
        parts = desc.split("\n")
        for p in parts:
            if p.lower().startswith("object:"):
                obj = p.split(":", 1)[1].strip()
            elif p.lower().startswith("explanation:"):
                expl = p.split(":", 1)[1].strip()
    results.append({"vg_image_id": img_path.stem, "topname_predicted": obj, "explanation": expl})

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV
df_results.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved results to {OUTPUT_FILE}")
