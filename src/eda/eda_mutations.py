import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_parquet("cleaned/mutations.parquet")

# Basic info
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Sample data:")
print(df.head())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Top mutated genes
top_genes = df["symbol"].value_counts().head(10)
print("\nTop mutated genes:")
print(top_genes)

# Mutation effects
effect_counts = df["effect"].value_counts()
print("\nMutation effects:")
print(effect_counts)

# Plot top mutated genes
top_genes.plot(kind="barh", title="Top Mutated Genes", figsize=(8, 5))
plt.xlabel("Mutation Count")
plt.ylabel("Gene")
plt.tight_layout()
plt.show()

print("\nBuilding mutation feature matrix...")

mutation_matrix = df.pivot_table(
    index="gene_id", columns="effect", aggfunc="size", fill_value=0
)

mutation_matrix.columns = mutation_matrix.columns.str.lower()

# Save to disk
mutation_matrix.to_parquet("final_features/mutation_features_by_gene.parquet")

print("Mutation matrix shape:", mutation_matrix.shape)
print(mutation_matrix.head())
