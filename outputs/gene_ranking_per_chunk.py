import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load the top combo results
df = pd.read_csv("outputs/top_combos_per_chunk.csv")

# Just keep the top N per chunk (e.g. best 3 per chunk)
top_n_per_chunk = df.groupby("chunk").apply(lambda g: g.nsmallest(3, "mse")).reset_index(drop=True)

# Flatten all gene names
all_genes = []
for genes in top_n_per_chunk["genes"]:
    all_genes.extend(genes.split(","))

# Count frequency of each gene
gene_counts = Counter(all_genes)

# Convert to DataFrame and sort
gene_df = pd.DataFrame(gene_counts.items(), columns=["gene", "count"]).sort_values("count", ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(gene_df["gene"], gene_df["count"], color="lightgreen")
plt.gca().invert_yaxis()
plt.xlabel("Appearances in Top Combos")
plt.title("Top Genes Appearing in Best Combos Across Chunks")
plt.tight_layout()
plt.show()
