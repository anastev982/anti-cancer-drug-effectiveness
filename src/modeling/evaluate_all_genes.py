import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "final_features/subset_preview.parquet"
OUTPUT_CSV = "outputs/all_genes_mse.csv"
OUTPUT_PLOT = "outputs/all_genes_mse.png"
TOP_N = 20  # Top N genes to show in plot


def load_data(path):
    logger.info("Loading dataset...")
    return pd.read_parquet(path)


def evaluate_genes(df):
    results = []
    for gene in df.columns:
        if gene.lower() == "ln_ic50":
            continue

        X = df[[gene]]
        y = df["LN_IC50"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"MSE for {gene}: {mse:.4f}")
        results.append({"gene": gene, "mse": mse})

    return pd.DataFrame(results)


def plot_mse(df, output_path):
    df_sorted = df.sort_values("mse").head(TOP_N)

    plt.figure(figsize=(10, 6))
    plt.barh(df_sorted["gene"], df_sorted["mse"], color="skyblue")
    plt.xlabel("MSE")
    plt.ylabel("Gene")
    plt.title(f"Top {TOP_N} Genes by Lowest MSE")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    df = load_data(DATA_PATH)
    results_df = evaluate_genes(df)
    results_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"MSE results saved to {OUTPUT_CSV}")

    plot_mse(results_df, OUTPUT_PLOT)
    logger.info(f"Plot saved to {OUTPUT_PLOT}")
