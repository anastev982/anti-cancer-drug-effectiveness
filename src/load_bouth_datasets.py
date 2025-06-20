import pandas as pd

drug_df = pd.read_parquet("merged_drug_data.parquet")
gene_df = pd.read_parquet("merged_gene_data.parquet")

print("Files loaded.")
print(f"Drug Data: {drug_df.shape}")
print(f"Gene Data: {gene_df.shape}")
# Drug data
print("\nDrug Data Overview:")
print(drug_df.info())
print(drug_df.describe(include="all").T)


# Gene data
print("\nGene Data Overview:")
print(gene_df.info())
print(gene_df.describe(include="all").T)
