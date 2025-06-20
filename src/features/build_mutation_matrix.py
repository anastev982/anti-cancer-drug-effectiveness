import pandas as pd
from pathlib import Path

# === Load mutation-related data ===

# Binary mutation matrix (rows: model_id, columns: genes)
mutation_matrix = pd.read_parquet("final_features/mutation_matrix.parquet")

# Full mutation annotations (gene-level info)
mutations = pd.read_parquet("cleaned/mutations.parquet")

# Gene ↔ model ID mapping
gene_map = pd.read_parquet("cleaned/cleaned_gene_data.parquet")

# === Load drug response data ===
drug_response = pd.read_excel("data/backup/GDSC2_fitted_dose_response_27Oct23.xlsx")

# === Optional: Merge mutation + gene_map (useful if rebuilding matrix) ===
merged_mutation = mutations.merge(
    gene_map[["gene_id", "model_id"]], on="gene_id", how="inner"
)

# === Merge drug response data with mutation matrix ===
# mutation_matrix index: model_id
# drug_response column: SANGER_MODEL_ID
merged = pd.merge(
    drug_response,
    mutation_matrix,
    how="inner",
    left_on="SANGER_MODEL_ID",
    right_index=True,
)

print(" Merged drug response with mutation matrix")
print("Shape:", merged.shape)
print("Columns:", merged.columns.tolist())
print(merged.head())

# === Extract features (X) and target (y) ===

# Choose prediction target: LN_IC50
y = merged["LN_IC50"]

# Drop non-feature columns to create input matrix
X = merged.drop(
    columns=[
        "AUC",
        "LN_IC50",
        "SANGER_MODEL_ID",
        "CELL_LINE_NAME",
        "DRUG_NAME",
        "PUTATIVE_TARGET",
        "PATHWAY_NAME",
        "COMPANY_ID",
        "WEBRELEASE",
        "MIN_CONC",
        "MAX_CONC",
        "RMSE",
        "Z_SCORE",
        "NLME_RESULT_ID",
        "NLME_CURVE_ID",
        "COSMIC_ID",
        "TCGA_DESC",
        "DATASET",
        "DRUG_ID",
    ],
    errors="ignore",
)

# === Save final data for modeling ===
X.to_parquet("final_features/X.parquet")
y.to_frame().to_parquet("final_features/y.parquet")

print(" Feature matrix (X) and target (y) saved")
print("X shape:", X.shape)
print("y shape:", y.shape)
