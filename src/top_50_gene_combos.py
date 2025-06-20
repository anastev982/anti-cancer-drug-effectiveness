import pandas as pd
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the top combos per chunk
df = pd.read_csv("outputs/top_combos_per_chunk.csv")

# Count frequency of each combination
combo_counts = df["genes"].value_counts().reset_index()
combo_counts.columns = ["genes", "count"]

# Merge with MSE info
merged = df.merge(combo_counts, on="genes")

# Drop duplicates and sort by count, then MSE
top_combos = (
    merged.drop_duplicates("genes")
    .sort_values(by=["count", "mse"], ascending=[False, True])
    .head(50)
)

# Extract top genes from top combos
top_combos["genes_list"] = top_combos["genes"].apply(lambda x: x.split(","))
all_genes = [gene for sublist in top_combos["genes_list"] for gene in sublist]
top_genes_df = pd.DataFrame(
    Counter(all_genes).items(), columns=["gene", "count"]
).sort_values("count", ascending=False)

# Visualize combo sizes vs MSE
top_combos["num_genes"] = top_combos["genes_list"].apply(len)
plt.figure(figsize=(10, 6))
sns.boxplot(data=top_combos, x="num_genes", y="mse")
plt.title("MSE Distribution by Number of Genes in Combo")
plt.xlabel("Number of Genes in Combination")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Save top gene combos for future testing
# Save or print the result
top_combos.to_csv("outputs/top_50_gene_combos.csv", index=False)
print("Saved top 50 combos to outputs/top_50_gene_combos.csv")
