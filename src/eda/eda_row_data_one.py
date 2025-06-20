import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Load cleaned file
CLEANED_PATH = Path("cleaned/cleaned_row_data_one.parquet")
df = pd.read_parquet(CLEANED_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Sample data:")
print(df.head())

# Number of unique drugs
print("\nUnique drugs:", df["Drug name"].nunique())

# Top 10 most common features
print("\nMost common features:")
print(df["Feature Name"].value_counts().head(10))

# IC50 effect size stats
print("\nIC50 effect size summary:")
print(df["ic50_effect_size"].describe())

# Plot histogram of IC50 effect size
plt.hist(df["ic50_effect_size"].dropna(), bins=50, color="skyblue", edgecolor="black")
plt.title("Distribution of IC50 Effect Size")
plt.xlabel("IC50 Effect Size")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
