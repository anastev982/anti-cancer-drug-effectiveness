import logging
from collections import Counter
from typing import List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = "outputs/gene_combination_mse.csv"
TOP_COMBO_ROWS = 50  # number of top rows to inspect


def get_top_genes_from_top_combinations(
    input_path: str = INPUT_PATH,
    top_rows: int = TOP_COMBO_ROWS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    From the best-performing gene combinations (by lowest MSE),
    compute how frequently each gene appears.

    Returns:
        - DataFrame with columns ['gene', 'count'] sorted by count desc.
        - list of unique gene names ordered by frequency (most common first).
    """
    logger.info(f"Loading gene combination results from {input_path} ...")
    df = pd.read_csv(input_path)

    df_sorted = df.sort_values("mse")
    top_n_rows = df_sorted.head(top_rows).copy()

    # Split 'genes' into lists
    top_n_rows["genes"] = top_n_rows["genes"].apply(lambda x: x.split(","))

    # Flatten all genes
    all_genes = [
        gene.strip() for gene_list in top_n_rows["genes"] for gene in gene_list
    ]

    gene_counts = Counter(all_genes)
    top_genes_df = (
        pd.DataFrame(gene_counts.items(), columns=["gene", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    ordered_genes = top_genes_df["gene"].tolist()

    logger.info(f"Top 10 genes from top {top_rows} combinations:")
    logger.info(ordered_genes[:10])

    return top_genes_df, ordered_genes


if __name__ == "__main__":
    top_genes_df, ordered_genes = get_top_genes_from_top_combinations()
