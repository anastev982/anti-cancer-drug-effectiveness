import pandas as pd
from collections import Counter

# Load your combined chunk results
df = pd.read_csv("outputs/top_combos_per_chunk.csv")

# Drop rows with missing genes (if any)
df = df.dropna(subset=["genes"])

# Split gene strings into lists
df["gene_list"] = df["genes"].apply(lambda x: x.split(","))

# Flatten all gene names across all chunks
all_genes = [gene.strip() for sublist in df["gene_list"] for gene in sublist]

# Count how often each gene appears
gene_counts = Counter(all_genes)

# Turn into a DataFrame
top_genes_df = pd.DataFrame(gene_counts.items(), columns=["gene", "count"])
top_genes_df = top_genes_df.sort_values("count", ascending=False).reset_index(drop=True)

# Show top 30
print(top_genes_df.head(30))

# Save to CSV if needed
top_genes_df.head(30).to_csv("outputs/top_30_genes_across_chunks.csv", index=False)
