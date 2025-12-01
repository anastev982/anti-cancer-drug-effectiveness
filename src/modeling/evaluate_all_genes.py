import logging
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "final_features/subset_preview.parquet"
OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "all_genes_mse.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "all_genes_mse.png")
TARGET_COL = "LN_IC50"
TOP_N = 20  # Top N genes to show in plot


def load_data(path: str) -> pd.DataFrame:
    """Load feature matrix with LN_IC50 target."""
    logger.info(f"Loading dataset from {path} ...")
    df = pd.read_parquet(path)
    logger.info(f"Loaded shape: {df.shape}")
    return df


def evaluate_single_genes(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    For each gene (column) compute MSE of a RandomForestRegressor
    using only that gene as a predictor.
    """
    results = []

    for gene in df.columns:
        if gene.lower() == target_col.lower():
            continue

        X = df[[gene]]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"MSE for {gene}: {mse:.4f}")
        results.append({"gene": gene, "mse": mse})

    return pd.DataFrame(results)


def plot_mse(df: pd.DataFrame, output_path: str, top_n: int = TOP_N) -> None:
    """Plot TOP_N genes with lowest MSE."""
    df_sorted = df.sort_values("mse").head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(df_sorted["gene"], df_sorted["mse"])
    plt.xlabel("MSE")
    plt.ylabel("Gene")
    plt.title(f"Top {top_n} genes by lowest MSE")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved plot to {output_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    results_df = evaluate_single_genes(df, TARGET_COL)
    results_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"MSE results saved to {OUTPUT_CSV}")

    plot_mse(results_df, OUTPUT_PLOT)
