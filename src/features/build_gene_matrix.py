import pandas as pd

# Load merged data (with model_id and gene symbol)
df = pd.read_parquet("cleaned/cleaned_gene_data.parquet")

# Sanity check
print("Shape before filtering:", df.shape)
print("Columns:", df.columns.tolist())

# Drop rows without model_id or gene symbol
df = df.dropna(subset=["model_id", "gene_symbol"])

# Create binary flag for presence of mutation
df["mutated"] = 1

# Pivot to wide format: model_id as rows, genes as columns
pivot_df = df.pivot_table(
    index="model_id",  # each row is a model/sample
    columns="gene_symbol",  # each column is a gene
    values="mutated",  # values = 1 if mutated
    aggfunc="max",  # in case of duplicates, take max (i.e. 1)
    fill_value=0,  # fill non-mutations with 0
)

# Clean column names: lowercase, no special chars
pivot_df.columns = pivot_df.columns.str.replace(r"[^\w]", "_", regex=True).str.lower()

# Optional: sort gene columns alphabetically for consistency
pivot_df = pivot_df[sorted(pivot_df.columns)]

# Print shape and sample
print("Gene mutation matrix shape:", pivot_df.shape)
print(pivot_df.head())

pivot_df.to_parquet("final_features/mutation_matrix.parquet")
print(" Saved matrix to final_features/mutation_matrix.parquet")
