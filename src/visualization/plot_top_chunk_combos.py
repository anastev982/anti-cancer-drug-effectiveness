import pandas as pd
import matplotlib.pyplot as plt

# Load the evaluated combinations from all chunks
df = pd.read_csv("outputs/top_combos_per_chunk.csv")

# Sort by lowest MSE
top_10 = df.sort_values("mse").head(10)

# Set up the plot
plt.figure(figsize=(12, 6))
bars = plt.barh(range(len(top_10)), top_10["mse"], color="skyblue")

# Add gene combo + chunk as labels
labels = [
    f"{row['genes']} ({row['chunk'].replace('.parquet', '')})"
    for _, row in top_10.iterrows()
]
plt.yticks(range(len(top_10)), labels)

plt.xlabel("MSE (Lower is Better)")
plt.title("Top 10 Gene Combinations by MSE Across Chunks")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
