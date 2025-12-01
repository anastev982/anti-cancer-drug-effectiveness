import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = "final_features/subset_genes_only.parquet"
CHUNK_DIR = "final_chunks"
CHUNK_SIZE = 50


def chunk_dataset(
    input_path: str = INPUT_PATH,
    chunk_dir: str = CHUNK_DIR,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """
    Shuffle the dataset and split it into parquet chunks of size `chunk_size`.
    """
    logger.info(f"Loading dataset from {input_path} ...")
    df = pd.read_parquet(input_path)

    logger.info("Shuffling dataset ...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(chunk_dir, exist_ok=True)

    logger.info(f"Saving chunks of size {chunk_size} to {chunk_dir}/ ...")
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size]
        chunk_path = Path(chunk_dir) / f"chunk_{i // chunk_size}.parquet"
        chunk.to_parquet(chunk_path)
        logger.info(f"Saved: {chunk_path}")

    logger.info("All chunks saved.")


if __name__ == "__main__":
    chunk_dataset()
