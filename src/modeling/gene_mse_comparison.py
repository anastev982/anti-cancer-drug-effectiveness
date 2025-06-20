import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging


# === Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    logger.info("Loading subset...")
    return pd.read_parquet("final_features/subset_preview.parquet")


def train_and_evaluate(df, gene):
    X = df[[gene]]
    y = df["LN_IC50"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--genes", nargs="*", help="List of genes to evaluate", required=True
    )
    args = parser.parse_args()

    df = load_data()

    results = []
    for gene in args.genes:
        if gene not in df.columns:
            logger.warning(f"Gene {gene} not found in dataset. Skipping.")
            continue
        mse = train_and_evaluate(df, gene)
        logger.info(f"MSE for {gene}: {mse:.4f}")
        results.append((gene, mse))

    # Save results table
    df_results = pd.DataFrame(results, columns=["Gene", "MSE"])
    df_results.to_csv("outputs/gene_mse_comparison.csv", index=False)
    logger.info("Saved comparison table to outputs/gene_mse_comparison.csv")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(df_results["Gene"], df_results["MSE"])
    plt.ylabel("MSE")
    plt.title("MSE by Gene")
    plt.tight_layout()
    plt.savefig("outputs/gene_mse_comparison.png")
    logger.info("Saved MSE plot to outputs/gene_mse_comparison.png")


if __name__ == "__main__":
    main()
