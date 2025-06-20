# src/interpretability/permutation_importance.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Load the same chunk
df = pd.read_parquet("final_chunks/chunk_14.parquet")
genes = ["lrrc37a2", "mstn"]
X = df[genes]
y = df["LN_IC50"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Run permutation importance
results = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Plot
sorted_idx = results.importances_mean.argsort()
plt.barh(range(len(genes)), results.importances_mean[sorted_idx])
plt.yticks(range(len(genes)), [genes[i] for i in sorted_idx])
plt.xlabel("Mean Decrease in Performance (MSE)")
plt.title("Permutation Importance")
plt.tight_layout()
plt.show()
