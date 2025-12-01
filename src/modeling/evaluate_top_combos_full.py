import os
import ast
import logging
from typing import List, Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ============================
# Logging configuration
# ============================
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ============================
# Constants
# ============================
COMBOS_PATH = "outputs/top_50_gene_combos.csv"
FULL_DATA_PATH = "final_features/full_model_input.parquet"

OUTPUT_DIR = "outputs"
BATCH_OUTPUT = os.path.join(OUTPUT_DIR, "eval_top_50_in_batches.csv")
SORTED_OUTPUT = os.path.join(OUTPUT_DIR, "sorted_gene_combos.csv")

TARGET_COL = "LN_IC50"
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_ESTIMATORS = 100
BATCH_SIZE = 10


def load_combos(path: str) -> pd.DataFrame:
    """
    Load the gene combinations file and ensure that the 'genes' column
    is a list of gene names.
    """
    combos_df = pd.read_csv(path)

    # Handle different possible formats for the 'genes' column
    if isinstance(combos_df["genes"].iloc[0], str):
        try:
            combos_df["genes"] = combos_df["genes"].apply(ast.literal_eval)
        except Exception:
            combos_df["genes"] = combos_df["genes"].apply(lambda x: x.split(","))

    return combos_df


def get_all_unique_genes(combos_df: pd.DataFrame) -> List[str]:
    """
    Extract a sorted list of all unique gene names from the 'genes' column.
    """
    all_genes = {g.strip() for genes in combos_df["genes"] for g in genes}
    return sorted(all_genes)


def load_feature_subset(
    data_path: str,
    genes: List[str],
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Load only the columns needed for the evaluation:
    all genes used in any combination + the target column.
    """
    use_columns = genes + [target_col]
    df = pd.read_parquet(data_path, columns=use_columns)
    logger.info(f"Loaded full data subset with shape: {df.shape}")
    return df


def evaluate_combos_in_batches(
    df: pd.DataFrame,
    combos_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Evaluate all gene combinations in batches, training a RandomForestRegressor
    for each combination and computing MSE and R².

    Returns a DataFrame with one row per combination.
    """
    y = df[TARGET_COL]
    all_results: List[dict[str, Any]] = []

    for start in range(0, len(combos_df), batch_size):
        batch = combos_df.iloc[start : start + batch_size]
        batch_id = start // batch_size + 1
        logger.info(f"\n=== Processing batch {batch_id} ({len(batch)} combos) ===")

        for genes in batch["genes"]:
            cols = [g for g in genes if g in df.columns]
            logger.info(f"  Genes: {genes} -> usable: {cols}")

            # Skip if no usable features
            if len(cols) < 2:
                logger.info(f"  Skipping {genes} — not enough usable columns")
                all_results.append(
                    {
                        "genes": ",".join(genes),
                        "num_genes": len(genes),
                        "used_genes": ",".join(cols),
                        "mse": None,
                        "r2": None,
                    }
                )
                continue

            # Train model
            X = df[cols]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

            model = RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            all_results.append(
                {
                    "genes": ",".join(genes),
                    "num_genes": len(genes),
                    "used_genes": ",".join(cols),
                    "mse": mse,
                    "r2": r2,
                }
            )

        # Save after each batch (optional but robust)
        batch_df = pd.DataFrame(all_results)
        batch_df.to_csv(BATCH_OUTPUT, index=False)
        logger.info(f"Batch {batch_id} appended to {BATCH_OUTPUT}")

    return pd.DataFrame(all_results)


def sort_and_save_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the results by MSE (ascending) and R² (descending),
    and save them to SORTED_OUTPUT.
    """
    sorted_df = results_df.sort_values(
        by=["mse", "r2"], ascending=[True, False], na_position="last"
    )
    sorted_df.to_csv(SORTED_OUTPUT, index=False)
    logger.info(f"Saved sorted gene combos to {SORTED_OUTPUT}")
    return sorted_df


def main() -> None:
    logger.info("Loading gene combinations...")
    combos_df = load_combos(COMBOS_PATH)

    logger.info("Extracting unique genes from combinations...")
    all_genes = get_all_unique_genes(combos_df)
    logger.info(f"Total unique genes: {len(all_genes)}")

    logger.info("Loading full dataset (subset of relevant columns)...")
    df = load_feature_subset(FULL_DATA_PATH, all_genes)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Evaluating combinations on full dataset...")
    results_df = evaluate_combos_in_batches(df, combos_df, BATCH_SIZE)
    results_df.to_csv(BATCH_OUTPUT, index=False)
    logger.info(f"Saved batch evaluation results to {BATCH_OUTPUT}")

    sorted_df = sort_and_save_results(results_df)
    logger.info("Top 10 combinations:")
    logger.info(sorted_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
