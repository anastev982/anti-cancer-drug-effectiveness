import logging
from collections import Counter
from typing import List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = "outputs/top_combos_per_chunk.csv"
OUTPUT_PATH = "outputs/top_30_genes_across_chunks.csv"
TOP_N = 30


def compute_top_genes_across_chunks(
    input_path: str = INPUT_PATH, top_n: int = TOP_N
) -> pd.DataFrame:
    """
    Aggregate gene frequencies across all chunks based on the 'genes' column.
    """
    logger.info(f"Loading chunk combination results from {input_path} ...")
    df = pd.read_csv(input_path)

    df = df.dropna(subset=["genes"])
    df["gene_list"] = df["genes"].apply(lambda x: x.split(","))

    all_genes: List[str] = [
        gene.strip() for sublist in df["gene_list"] for gene in sublist
    ]

    gene_counts = Counter(all_genes)
    top_genes_df = (
        pd.DataFrame(gene_counts.items(), columns=["gene", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("Top genes across chunks:")
    logger.info(top_genes_df.head(top_n).to_string(index=False))

    return top_genes_df.head(top_n)


def main() -> None:
    top_genes_df = compute_top_genes_across_chunks(INPUT_PATH, TOP_N)
    top_genes_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved top {TOP_N} genes across chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
