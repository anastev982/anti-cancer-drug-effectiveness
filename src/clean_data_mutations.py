import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw_data/processed/mutations.csv")
CLEANED_PATH = Path("cleaned/mutations.parquet")

# Load raw CSV
df = pd.read_csv(RAW_PATH)

print(" Raw shape:", df.shape)
print(" Columns:", df.columns.tolist())

# Clean basic issues
df = df.dropna(subset=["symbol", "effect"])  # Drop rows missing key values
df = df.drop_duplicates()

# Save to Parquet
CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(CLEANED_PATH)
print(f" Saved cleaned data to {CLEANED_PATH}")
