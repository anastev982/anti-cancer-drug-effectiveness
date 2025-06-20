# build_full_model_input.py

import pandas as pd

# Load drug response data (Excel)
drug_response_df = pd.read_excel("data/backup/GDSC2_fitted_dose_response_27Oct23.xlsx")

# Load mutation matrix
mutation_df = pd.read_parquet("final_features/mutation_matrix.parquet")

# Merge on 'SANGER_MODEL_ID' from drug_response ↔ index of mutation_df
merged = pd.merge(
    drug_response_df,
    mutation_df,
    how="inner",
    left_on="SANGER_MODEL_ID",
    right_index=True,
)

# Save to disk so training can use it later
merged.to_parquet("final_features/full_model_input.parquet")

print("Merged dataset saved as full_model_input.parquet")
print("Shape:", merged.shape)
