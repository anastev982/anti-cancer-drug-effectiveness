import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load the filtered dataset ===
df = pd.read_parquet("final_features/subset_genes_only.parquet")
top_genes = [
    "mstn",
    "bambi",
    "tmt1a",
    "sfmbt1",
    "fcrl6",
    "aifm2",
    "pfn2",
    "fyn",
    "lrrc37a2",
    "osbp2",
]

results = []

for r in range(2, len(top_genes) + 1):
    for combo in combinations(top_genes, r):
        X = df[list(combo)]
        y = df["LN_IC50"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        results.append({"genes": ",".join(combo), "num_genes": r, "mse": mse})
        logger.info(f"MSE for {combo}: {mse:.4f}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/gene_combination_mse.csv", index=False)

# Plot
avg_mse = results_df.groupby("num_genes")["mse"].mean().reset_index()
plt.figure(figsize=(8, 5))
plt.plot(avg_mse["num_genes"], avg_mse["mse"], marker="o")
plt.title("Average MSE vs Number of Genes Used")
plt.xlabel("Number of Genes")
plt.ylabel("Average MSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/gene_combination_mse_plot.png")
plt.close()

logger.info(
    "Finished. Results saved to 'outputs/gene_combination_mse.csv' and plot to 'gene_combination_mse_plot.png'."
)
