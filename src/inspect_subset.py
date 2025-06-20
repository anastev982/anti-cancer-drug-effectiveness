import pandas as pd

# Define a small list of columns to load
columns_to_load = [
    "LN_IC50",
    "a1bg",
    "tp53",
    "egfr",
    "brca1",
    "brca2",  # example gene columns
]

# Load just those columns
df = pd.read_parquet(
    "final_features/full_model_input.parquet", columns=columns_to_load, engine="pyarrow"
)
print("Subset loaded successfully!")
print(df.head())
