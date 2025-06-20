import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt

# Load and parse CSV
df = pd.read_csv("outputs/gene_combination_mse.csv")

# Parse the stringified list into actual list
df["genes"] = df["genes"].str.split(",")


# Get top 20 performing combinations
top_combos = df.sort_values("mse").head(20)

# Count genes across these top combinations
all_genes = [gene for gene_list in top_combos["genes"] for gene in gene_list]
gene_counts = Counter(all_genes)

# Convert to DataFrame
top_genes_df = pd.DataFrame(gene_counts.items(), columns=["gene", "count"]).sort_values(
    "count", ascending=False
)

# Plot
top_genes_df.head(10).plot.bar(x="gene", y="count", legend=False)
plt.title("Most Common Genes in Top 20 Combinations")
plt.ylabel("Appearances")
plt.tight_layout()
plt.savefig("outputs/top_genes_in_best_combos_FIXED.png")
plt.show()
