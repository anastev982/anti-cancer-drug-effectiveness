import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
RAW_PATH = Path("data/raw_data/processed/row_data_drugs.csv")

# Load data
df = pd.read_csv(RAW_PATH)

# Show shape and columns
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Show head
print("Sample data:")
print(df.head())

# Clean column names (strip spaces)
df.columns = df.columns.str.strip()

# Check missing values
print("\nMissing values:")
print(df.isna().sum())

# Most common targets
print("\nTop drug targets:")
print(df["Targets"].value_counts().head(10))

# Most common pathways
print("\nTop target pathways:")
print(df["Target pathway"].value_counts().head(10))

# Plot top 10 drug targets
top_targets = df["Targets"].value_counts().head(10)
top_targets.plot(kind="barh")
plt.title("Top 10 Drug Targets")
plt.xlabel("Number of Drugs")
plt.ylabel("Target")
plt.tight_layout()
plt.show()
