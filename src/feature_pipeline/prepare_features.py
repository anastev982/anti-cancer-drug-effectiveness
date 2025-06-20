import os
import pandas as pd
from pathlib import Path
from src.config import RAW_PROCESSED_DIR

# Set path to cleaned data folder
CLEANED_DIR = Path("./cleaned")

# Load the cleaned datasets
drug_df = pd.read_parquet(RAW_PROCESSED_DIR / "merged_clean_data.parquet")
drug_df = drug_df.rename(columns={"SANGER_MODEL_ID": "model_id"})

gene_df = pd.read_parquet(CLEANED_DIR / "cleaned_gene_data.parquet")

print("Merged drug_df columns:")
print(drug_df.columns.tolist())

# Check basic structure
print(f"Drug data shape: {drug_df.shape}")
print(f"Gene data shape: {gene_df.shape}")

print("Available gene_df columns:")
print(gene_df.columns.tolist())

# Step 2: Filter useful columns from gene data
important_gene_cols = ["model_id", "effect", "gene_symbol", "vaf"]
gene_features = gene_df[important_gene_cols].copy()

# Drop rows with missing model_id or gene_symbol
gene_features.dropna(subset=["model_id", "gene_symbol"], inplace=True)

# Create binary flag: mutation exists
gene_features["mutated"] = 1

pivot_df = gene_features.pivot_table(
    index="model_id",
    columns="gene_symbol",
    values="mutated",
    aggfunc="max",
    fill_value=0,
)

# Clean column names
pivot_df.columns = pivot_df.columns.str.replace(r"[^\w]", "_", regex=True)
pivot_df.columns = pivot_df.columns.str.strip().str.lower()

# Optional: sort columns alphabetically
pivot_df = pivot_df[sorted(pivot_df.columns)]

print("Pivoted mutation matrix shape:", pivot_df.shape)
print(pivot_df.head())

# Sanity check
print("Filtered gene data shape:", gene_features.shape)
print("Sample gene mutations:")
print(gene_features.head())

# One-hot encode mutation effects
effect_ohe = pd.get_dummies(
    gene_features[["model_id", "effect"]], columns=["effect"], prefix="effect"
)

# Group by model_id to aggregate one-hot encoded effects
effect_encoded = effect_ohe.groupby("model_id").sum()

# Preview
print("Effect-encoded matrix shape:", effect_encoded.shape)
print(effect_encoded.head())

# MERGING
print("Drug DataFrame columns:")
print(drug_df.columns.tolist())

print("Reducing drug_df to only model_id for merge...")
drug_model_df = drug_df[["model_id"]].drop_duplicates()

# Merge with gene mutation matrix
merged_gene_features = drug_model_df.merge(pivot_df, on="model_id", how="left")

# Merge with one-hot encoded mutation effects
merged_gene_features = merged_gene_features.merge(
    effect_encoded, on="model_id", how="left"
)

# Fill NaNs (assume no mutation)
merged_gene_features = merged_gene_features.fillna(0)

# Save to disk
os.makedirs("final_features", exist_ok=True)
print("Saving gene feature matrix...")
merged_gene_features.to_parquet("final_features/model_gene_features.parquet")

# Print final shape and sample
print(" Final feature matrix:", merged_gene_features.shape)
print(merged_gene_features.head())
