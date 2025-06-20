import argparse
import logging
from unittest import result
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path):
    logger.info("Loading subset...")
    return pd.read_parquet(path)


def train_model(df, genes):
    if not all(g in df.columns for g in genes):
        missing = [g for g in genes if g not in df.columns]
        raise ValueError(f"Missing genes in dataset: {missing}")

    X = df[genes]
    y = df["LN_IC50"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    return mse, model


def plot_importance(model, feature_names, output_path):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(8, 5))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(
        range(len(importances)), [feature_names[i] for i in indices], rotation=45
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--genes", nargs="+", required=True, help="List of gene names to use"
    )
    parser.add_argument(
        "--data",
        default="final_features/subset_preview.parquet",
        help="Path to dataset",
    )
    parser.add_argument(
        "--plot",
        default="outputs/feature_importance_selected.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare each gene individually"
    )
    args = parser.parse_args()

    df = load_data(args.data)

if args.compare:
    results = []
    for gene in args.genes:
        mse, _ = train_model(df, [gene])
        logger.info(f"MSE for {gene}: {mse:.4f}")
        results.append({"Gene": gene, "MSE": mse})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/gene_mse_comparison.csv", index=False)
    logger.info("Comparison results saved to outputs/gene_mse_comparison.csv")

    # Plot MSEs
    plt.figure(figsize=(8, 5))
    plt.bar([r["Gene"] for r in results], [r["MSE"] for r in results])
    plt.ylabel("Mean Squared Error")
    plt.title("MSE per Gene")
    plt.tight_layout()
    plt.savefig("outputs/gene_mse_comparison.png")
    logger.info("Comparison plot saved to outputs/gene_mse_comparison.png")

    # Plot sorted MSEs for comparison
    results_sorted = results_df.sort_values("MSE")
    plt.figure(figsize=(8, 5))
    plt.bar(results_sorted["Gene"], results_sorted["MSE"])
    plt.ylabel("Mean Squared Error")
    plt.title("MSE per Gene (Sorted)")
    plt.tight_layout()
    plt.savefig("outputs/gene_mse_comparison_sorted.png")
    logger.info(
        "Sorted comparison plot saved to outputs/gene_mse_comparison_sorted.png"
    )
