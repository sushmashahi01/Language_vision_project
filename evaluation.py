import pandas as pd

# === Input files ===
files = [
    "annotated_with_objects_clip.csv",
    "annotated_with_objects_llama.csv",
    "annotated_with_objects_llava.csv",
    "annotated_with_objects_gemma.csv"
]
ground_truth_file = "manynames-en.tsv"  # has: vg_object_id, vg_image_id, topname

# Load ground truth once
df_gt = pd.read_csv(ground_truth_file, sep="\t")
df_gt["vg_image_id"] = df_gt["vg_image_id"].astype(str)
df_gt_unique = df_gt.drop_duplicates(subset=["vg_image_id"])[["vg_image_id", "topname"]]

# Store accuracies here
results = []

for gen_file in files:
    print(f"🔎 Processing {gen_file} ...")

    # Load predictions
    df_gen = pd.read_csv(gen_file)
    df_gen["vg_image_id"] = df_gen["vg_image_id"].astype(str)

    # Merge with ground truth
    df_merged = df_gen.merge(
        df_gt_unique,
        on="vg_image_id",
        how="left",
        suffixes=("", "_gt")
    )

    # Normalize case/whitespace
    df_merged["topname_predicted"] = (
        df_merged["topname_predicted"].astype(str).str.strip().str.lower()
    )
    df_merged["topname_gt"] = (
        df_merged["topname"].astype(str).str.strip().str.lower()
    )

    # Remove rows where prediction is NaN or empty
    df_merged = df_merged[df_merged["topname_predicted"].notna()]
    df_merged = df_merged[df_merged["topname_predicted"] != "nan"]
    df_merged = df_merged[df_merged["topname_predicted"] != ""]

    # Compare predictions vs ground truth
    df_merged["match"] = df_merged["topname_predicted"] == df_merged["topname_gt"]

    # Counts
    total = len(df_merged)
    correct = df_merged["match"].sum()
    accuracy = correct / total if total > 0 else 0

    print(f"✅ {gen_file}: {correct}/{total} correct ({accuracy:.2%})")

    # Save per-file merged results
    out_name = f"comparison_results_{gen_file.replace('.csv', '')}.csv"
    df_merged.to_csv(out_name, index=False)

    # Append to results
    results.append({
        "file": gen_file,
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    })

# Save summary of accuracies
df_results = pd.DataFrame(results)
df_results.to_csv("result.csv", index=False)

print("\n📊 Accuracy summary saved to result.csv")
print(df_results)
