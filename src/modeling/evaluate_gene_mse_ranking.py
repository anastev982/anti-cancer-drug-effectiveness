import logging
import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ============================
# Logging configuration
# ============================
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================
# Constants
# ============================
INPUT_PATH = "final_features/subset_genes_only.parquet"
OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "gene_mse_ranking.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "gene_mse_top20.png")

TARGET_COL = "LN_IC50"
TOP_N = 20

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200


def evaluate_gene_mse(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    For each gene (column) in the dataframe, train a RandomForestRegressor
    using only that gene as input and compute the MSE against the target.

    Returns a DataFrame with columns: ["gene", "mse"].
    """
    results = []

    for gene in df.columns:
        if gene.lower() == target_col.lower():
            # Skip the target column itself
            continue

        X = df[[gene]]
        y = df[target_col]

        # Skip non-informative features (constant values)
        if X[gene].nunique() <= 1:
            logger.info(f"Skipping {gene} (non-informative / constant).")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        model = RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_estimators=N_ESTIMATORS,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        results.append((gene, mse))

        logger.info(f"MSE for {gene}: {mse:.4f}")

    return pd.DataFrame(results, columns=["gene", "mse"])


def plot_top_genes(
    df: pd.DataFrame,
    top_n: int = TOP_N,
    output_path: str = OUTPUT_PLOT,
) -> None:
    """
    Plot the top_n genes with the lowest MSE values.
    """
    top_df = df.sort_values("mse").head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top_df["gene"], top_df["mse"])
    plt.xlabel("MSE (lower is better)")
    plt.title(f"Top {top_n} genes by lowest MSE")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Top {top_n} gene plot saved to {output_path}")


def main() -> None:
    logger.info("Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Evaluating each gene...")
    results_df = evaluate_gene_mse(df, TARGET_COL)

    results_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Results saved to {OUTPUT_CSV}")

    plot_top_genes(results_df, TOP_N)


if __name__ == "__main__":
    main()
