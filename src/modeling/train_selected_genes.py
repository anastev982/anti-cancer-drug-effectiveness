import argparse
import logging
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Logging configuration

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Constants

DEFAULT_DATA_PATH = "final_features/subset_genes_only.parquet"
DEFAULT_PLOT_PATH = "outputs/feature_importance_selected.png"
TARGET_COL = "LN_IC50"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
OUTPUT_DIR = "outputs"


def load_data(path: str) -> pd.DataFrame:
    """Load subset data from parquet file."""
    logger.info(f"Loading dataset from {path} ...")
    df = pd.read_parquet(path)
    logger.info(f"Loaded shape: {df.shape}")
    return df


def train_model(df: pd.DataFrame, genes: List[str]) -> Tuple[float, RandomForestRegressor]:
    """
    Train a RandomForestRegressor using the specified genes as features.
    Returns MSE on a held-out test set and the trained model.
    """
    missing = [g for g in genes if g not in df.columns]
    if missing:
        raise ValueError(f"Missing genes in dataset: {missing}")

    X = df[genes]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_estimators=N_ESTIMATORS,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Model MSE using genes {genes}: {mse:.4f}")
    return mse, model


def plot_importance(
    model: RandomForestRegressor,
    feature_names: List[str],
    output_path: str = DEFAULT_PLOT_PATH,
) -> None:
    """
    Plot feature importances of the trained RandomForest model.
    """
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(8, 5))
    plt.title("Feature Importance (Selected Genes)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right",
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Feature importance plot saved to {output_path}")


def compare_genes_individually(df: pd.DataFrame, genes: List[str]) -> pd.DataFrame:
    """
    Train one-gene models for each gene and compute MSE individually.
    Returns a DataFrame with MSE per gene and saves plots + CSV.
    """
    results = []

    for gene in genes:
        if gene not in df.columns:
            logger.warning(f"Skipping {gene} (not in dataset).")
            continue

        mse, _ = train_model(df, [gene])
        results.append({"Gene": gene, "MSE": mse})

    if not results:
        logger.warning("No valid genes to compare.")
        return pd.DataFrame(columns=["Gene", "MSE"])

    results_df = pd.DataFrame(results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "gene_mse_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Comparison results saved to {csv_path}")

    # Plot unsorted MSEs
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Gene"], results_df["MSE"])
    plt.ylabel("Mean Squared Error")
    plt.title("MSE per Gene")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "gene_mse_comparison.png")
    plt.savefig(png_path)
    plt.close()
    logger.info(f"Comparison plot saved to {png_path}")

    # Plot sorted MSEs
    results_sorted = results_df.sort_values("MSE")
    plt.figure(figsize=(8, 5))
    plt.bar(results_sorted["Gene"], results_sorted["MSE"])
    plt.ylabel("Mean Squared Error")
    plt.title("MSE per Gene (Sorted)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    sorted_png_path = os.path.join(OUTPUT_DIR, "gene_mse_comparison_sorted.png")
    plt.savefig(sorted_png_path)
    plt.close()
    logger.info(f"Sorted comparison plot saved to {sorted_png_path}")

    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RandomForest models on selected genes and optionally compare them individually."
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        required=True,
        help="List of gene names to use as features",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to the input parquet dataset",
    )
    parser.add_argument(
        "--plot",
        default=DEFAULT_PLOT_PATH,
        help="Output path for feature importance plot",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="If set, evaluate each gene individually instead of a single multi-gene model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_data(args.data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.compare:
        logger.info("Running per-gene comparison...")
        compare_genes_individually(df, args.genes)
    else:
        logger.info("Training multi-gene model...")
        mse, model = train_model(df, args.genes)
        logger.info(f"Final MSE on test set: {mse:.4f}")
        plot_importance(model, args.genes, args.plot)


if __name__ == "__main__":
    main()
