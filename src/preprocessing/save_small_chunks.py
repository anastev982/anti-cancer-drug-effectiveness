import pandas as pd
import os

# Load full dataset
df = pd.read_parquet("final_features/subset_genes_only.parquet")

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Chunk size
chunk_size = 50
chunk_dir = "final_chunks"
os.makedirs(chunk_dir, exist_ok=True)

# Save chunks
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i : i + chunk_size]
    chunk_path = os.path.join(chunk_dir, f"chunk_{i//chunk_size}.parquet")
    chunk.to_parquet(chunk_path)
    print(f"Saved: {chunk_path}")

print("All chunks saved.")
