import pandas as pd
import logging
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_PATH = "final_features/subset_genes_only.parquet"
OUTPUT_CSV = "outputs/gene_mse_ranking.csv"
OUTPUT_PLOT = "outputs/gene_mse_top20.png"
TOP_N = 20


def evaluate_gene_mse(df, target_col="LN_IC50"):
    results = []
    for gene in df.columns:
        if gene == target_col:
            continue

        X = df[[gene]]
        y = df[target_col]

        if X[gene].nunique() <= 1:
            continue  # skip uninformative

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        results.append((gene, mse))

        logger.info(f"MSE for {gene}: {mse:.4f}")

    return pd.DataFrame(results, columns=["gene", "mse"])


def plot_top_genes(df, top_n=20, output_path=OUTPUT_PLOT):
    top_df = df.sort_values("mse").head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top_df["gene"], top_df["mse"], color="skyblue")
    plt.xlabel("MSE")
    plt.title(f"Top {top_n} Genes by Predictive Power")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Top {top_n} gene plot saved to {output_path}")


def main():
    logger.info("Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)

    logger.info("Evaluating each gene...")
    results_df = evaluate_gene_mse(df)

    results_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Results saved to {OUTPUT_CSV}")

    plot_top_genes(results_df, TOP_N)


if __name__ == "__main__":
    main()
