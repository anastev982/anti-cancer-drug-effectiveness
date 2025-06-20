#!/usr/bin/env python3
"""
Pipeline for preprocessing and saving GDSC datasets separately.
"""

from __future__ import annotations

import dask.dataframe as dd
import logging

from src.dataloader import load_and_prepare_data
from src.merge_utils import merged_drug_data, merged_gene_data
from pathlib import Path

_LOG = logging.getLogger("gdsc.processing")


def preprocess_and_save_separately() -> None:
    datasets = load_and_prepare_data()

    df_drugs = merged_drug_data(
        datasets["drug_list"],
        datasets["gdsc1"],
        datasets["gdsc2"],
    )

    df_genes = merged_gene_data(
        datasets["variance"],
        datasets["mutations"],
        datasets["all_mutations"],
        out_path="final_chunks/gene_df.parquet",
    )

    # Save drug_df as well
    df_drugs.to_parquet("final_chunks/drug_df.parquet", write_index=False)
    _LOG.info("Both drug and gene datasets saved to final_chunks/.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    preprocess_and_save_separately()
