import pandas as pd
from collections import Counter

# Load the CSV
df = pd.read_csv("outputs/gene_combination_mse.csv")

# Sort by MSE (best-performing combinations first)
df_sorted = df.sort_values("mse")

# Select top rows (adjust here for 50 or 100)
top_n_rows = df_sorted.head(50)  # or .head(100)

# Split the 'genes' column into lists
top_n_rows["genes"] = top_n_rows["genes"].apply(lambda x: x.split(","))

# Flatten the gene lists
all_genes = [gene.strip() for gene_list in top_n_rows["genes"] for gene in gene_list]

# Count frequency
gene_counts = Counter(all_genes)

# Convert to DataFrame for easier handling
top_genes_df = pd.DataFrame(gene_counts.items(), columns=["gene", "count"]).sort_values(
    "count", ascending=False
)

# Optionally: extract just the top N genes
top_50_genes = top_genes_df.head(50)["gene"].tolist()
top_100_genes = top_genes_df.head(100)["gene"].tolist()

# Preview
print("Top 10 genes from top 50 combinations:", top_50_genes[:10])
