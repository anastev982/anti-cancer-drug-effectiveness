import os
import logging
from itertools import combinations
from typing import List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Logging configuration

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Constants

CHUNK_DIR = "final_chunks"
OUTPUT_PATH = "outputs/top_combos_per_chunk.csv"

TARGET_COL = "LN_IC50"

TOP_GENES: List[str] = [
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

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200


def evaluate_combinations_in_chunk(df: pd.DataFrame, chunk_name: str) -> List[dict]:
    """
    Evaluate all combinations (size 2 to len(TOP_GENES)) of selected genes
    inside a single chunk. For each combination, train a RandomForestRegressor
    and compute the MSE.

    Returns a list of dicts, each representing a result row.
    """
    results = []
    y = df[TARGET_COL]

    for k in range(2, len(TOP_GENES) + 1):
        logger.info(f"  Checking combinations of size {k} ...")

        for combo in combinations(TOP_GENES, k):
            X = df[list(combo)]

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
                )

                model = RandomForestRegressor(
                    random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)

                results.append(
                    {
                        "chunk": chunk_name,
                        "genes": ",".join(combo),
                        "num_genes": k,
                        "mse": mse,
                    }
                )

            except Exception as e:
                logger.warning(f"    Skipping combo {combo} in {chunk_name}: {e}")

    return results


def process_all_chunks() -> pd.DataFrame:
    """
    Loop over all parquet chunks in CHUNK_DIR,
    evaluate gene combinations for each chunk,
    and aggregate results into a single DataFrame.
    """
    all_results = []

    for fname in sorted(os.listdir(CHUNK_DIR)):
        if fname.endswith(".parquet"):
            chunk_path = os.path.join(CHUNK_DIR, fname)
            logger.info(f"\nProcessing chunk: {fname}")

            df = pd.read_parquet(chunk_path)
            chunk_results = evaluate_combinations_in_chunk(df, fname)
            all_results.extend(chunk_results)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    logger.info("Starting chunk-level evaluation...")

    results_df = process_all_chunks()
    results_df.to_csv(OUTPUT_PATH, index=False)

    logger.info(f"\nSaved all results to: {OUTPUT_PATH}")
    logger.info("Done.")
