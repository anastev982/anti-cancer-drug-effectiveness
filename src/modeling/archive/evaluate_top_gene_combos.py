import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import logging

# === Setup ===
logging.basicConfig(level=logging.INFO, format="%(message)s")

TOP_GENES = [
    "fcrl6",
    "osbp2",
    "bambi",
    "lrrc37a2",
    "fyn",
    "tmt1a",
    "aifm2",
    "mstn",
    "sfmbt1",
    "pfn2",
]
DATA_PATH = "final_features/subset_genes_only.parquet"
OUTPUT_CSV = "outputs/top_gene_combos_mse.csv"

# === Load features and target ===
df = pd.read_parquet(DATA_PATH)
y = df["LN_IC50"]
X = df.drop(columns=["LN_IC50"])  # keep only features

results = []

# === Loop through 2- to 10-gene combinations ===
for k in range(2, len(TOP_GENES) + 1):
    logging.info(f"Running {k}-gene combinations...")
    for combo in combinations(TOP_GENES, k):
        X_subset = X[list(combo)]
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        results.append({"genes": ",".join(combo), "num_genes": k, "mse": mse})

# === Save results ===
results_df = pd.DataFrame(results)
os.makedirs("outputs", exist_ok=True)
results_df.to_csv(OUTPUT_CSV, index=False)
logging.info(f"Saved results to {OUTPUT_CSV}")
