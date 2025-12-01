import logging

import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = "final_features/full_model_input.parquet"
OUTPUT_PATH = "final_features/subset_informative.parquet"
TARGET = "LN_IC50"
SAMPLE_SIZE = 2000


def main() -> None:
    logger.info(f"Loading schema from {INPUT_PATH} ...")
    schema = pq.read_schema(INPUT_PATH)
    all_columns = schema.names

    if TARGET not in all_columns:
        raise ValueError(f"Target column '{TARGET}' not found in schema.")

    feature_columns = [col for col in all_columns if col != TARGET]
    use_columns = [TARGET] + feature_columns

    logger.info(f"Sampling {SAMPLE_SIZE} rows with selected columns ...")
    df = pd.read_parquet(INPUT_PATH, columns=use_columns).sample(
        n=SAMPLE_SIZE, random_state=42
    )

    logger.info("Filtering out non-informative genes (constant features) ...")
    non_target = df.drop(columns=[TARGET])
    informative_genes = [
        col for col in non_target.columns if df[col].nunique() > 1
    ]

    logger.info(f"Found {len(informative_genes)} informative genes.")
    df_filtered = df[[TARGET] + informative_genes]

    logger.info(f"Saving filtered subset to {OUTPUT_PATH} ...")
    df_filtered.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    main()
