#!/usr/bin/env python3
"""Merging logic for large-scale GDSC datasets.

This script:
- Cleans and normalises datasets.
- Merges drug and gene data.
- Logs progress and saves intermediate samples for debugging.
- Writes final merged output as partitioned Parquet for downstream ML / analytics.
"""

from __future__ import annotations

from ast import arg
import logging
from pathlib import Path
from typing import Optional, Set

import dask.dataframe as dd

from src.config import PROCESSED_DIR, RAW_PROCESSED_DIR, NEW_PROCESSED_DATA_DIR
from src.dataloader import load_final_df
from src.helpers import (
    clean_df,
    ensure_str_column,
    normalize_columns,
    save_final_df_dask,
)
from src.main import FINAL_PATH
from src.merge_utils import (
    _fill_missing_columns,
    merged_drug_data,
    merged_gene_data,
    safe_fillna,
)


_LOG = logging.getLogger("gdsc.merge")


# Utility Functions


def dropna_and_log(df, key: str, name: str) -> dd.DataFrame:
    before = df.shape[0].compute()
    df = df.dropna(subset=[key])
    after = df.shape[0].compute()
    dropped = before - after
    pct = (dropped / before * 100) if before > 0 else 0
    log_fn = _LOG.warning if pct > 10 else _LOG.info
    log_fn("%s: Dropped %d rows (%.1f%%) due to missing '%s'.", name, dropped, pct, key)
    return df


def save_debug_sample(df: dd.DataFrame, name: str) -> None:
    try:
        if "drug_id" in df.columns:
            valid_sample = df[~df["drug_id"].isna()].head(5)
            _LOG.info(
                "%s drug_id sample (non-null): %s",
                name,
                valid_sample["drug_id"].tolist(),
            )

        sample = df.head(5)
        out = Path(f"debug_{name}_sample.csv")
        sample.to_csv(out, index=False)
        _LOG.info("Saved debug sample → %s", out)
    except Exception as exc:
        _LOG.warning("Could not save debug sample for %s: %s", name, exc)


def ensure_required_columns_local(
    df: dd.DataFrame, required_cols: Set[str], name: str, default: str | int = -1
) -> dd.DataFrame:
    missing = required_cols - set(df.columns)
    if missing:
        _LOG.warning(
            "%s: Missing columns %s. Filling with default %s", name, missing, default
        )
        for col in missing:
            df[col] = default
    return df


def rename_gene_columns(
    variance: dd.DataFrame, mutations: dd.DataFrame, all_mutations: dd.DataFrame
) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
    if "symbol" in variance.columns:
        variance = variance.rename(columns={"symbol": "variant_symbol"})
        _LOG.info("Renamed 'symbol' → 'variant_symbol' in variance.")
    if "symbol" in mutations.columns:
        mutations = mutations.rename(columns={"symbol": "mutation_symbol"})
        _LOG.info("Renamed 'symbol' → 'mutation_symbol' in mutations.")
    if "gene_symbol" in all_mutations.columns:
        all_mutations = all_mutations.rename(
            columns={"gene_symbol": "allmut_gene_symbol"}
        )
        _LOG.info("Renamed 'gene_symbol' → 'allmut_gene_symbol' in all_mutations.")
    return variance, mutations, all_mutations


# Main Pipeline


def preprocess_and_merge(
    drug_list: dd.DataFrame,
    gdsc1: dd.DataFrame,
    gdsc2: dd.DataFrame,
    variance: dd.DataFrame,
    mutations: dd.DataFrame,
    all_mutations: dd.DataFrame,
) -> tuple[dd.DataFrame, dd.DataFrame]:
    _LOG.info("=== Starting final data cleaning & preprocessing pipeline ===")

    for df in [drug_list, gdsc1, gdsc2, variance, mutations, all_mutations]:
        df.columns = [col.lower().strip() for col in df.columns]

    # Rename gene symbols for clarity
    if "symbol" in variance.columns:
        variance = variance.rename(columns={"symbol": "symbol_id"})
    if "symbol" in mutations.columns:
        mutations = mutations.rename(columns={"symbol": "symbol_id"})
    if "gene_symbol" in all_mutations.columns:
        all_mutations = all_mutations.rename(columns={"gene_symbol": "symbol_id"})

    for df in [gdsc1, gdsc2]:
        df = _fill_missing_columns(
            df, ["drug_id", "conc", "intensity", "cell_line_name", "date_created"]
        )

    if FINAL_PATH.exists() and not arg.force_remerge:
        print("Found cached merged file. Loading…")
        merged = dd.read_parquet(FINAL_PATH)
    else:
        print("Merging again…")
        merged = merged_drug_data(...)
        merged.to_parquet(FINAL_PATH)

    drug_df = merged_drug_data(drug_list, gdsc1, gdsc2)
    gene_df = merged_gene_data(variance, mutations, all_mutations)

    drug_df = dropna_and_log(drug_df, "drug_id", "drug_df")
    gene_df = dropna_and_log(gene_df, "symbol_id", "gene_df")

    drug_df = drug_df.drop_duplicates(subset=["drug_id"])
    gene_df = gene_df.drop_duplicates(subset=["symbol_id"])

    drug_df["drug_id"] = drug_df["drug_id"].astype(str)
    gene_df["symbol_id"] = gene_df["symbol_id"].astype(str)

    out_dir = Path("final_chunks")
    out_dir.mkdir(exist_ok=True)

    drug_df.to_parquet(out_dir / "drug_df.parquet", write_index=False)
    gene_df.to_parquet(out_dir / "gene_df.parquet", write_index=False)

    _LOG.info("Saved drug_df and gene_df to final_chunks/")

    return drug_df, gene_df


# Entry point for standalone runs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    drug_list = dd.read_csv(RAW_PROCESSED_DIR / "drug_list.csv").sample(
        frac=0.01, random_state=42
    )
    gdsc1 = dd.read_csv(RAW_PROCESSED_DIR / "gdsc1.csv").sample(
        frac=0.005, random_state=42
    )
    gdsc2 = dd.read_csv(RAW_PROCESSED_DIR / "gdsc2.csv").sample(
        frac=0.005, random_state=42
    )
    drug_list = dd.read_csv(RAW_PROCESSED_DIR / "drug_list.csv").sample(
        frac=0.01, random_state=42
    )
    variance = dd.read_csv(RAW_PROCESSED_DIR / "variance.csv").sample(
        frac=0.005, random_state=42
    )
    mutations = dd.read_csv(RAW_PROCESSED_DIR / "mutations.csv").sample(
        frac=0.005, random_state=42
    )
    all_mutations = dd.read_csv(RAW_PROCESSED_DIR / "all_mutations.csv").sample(
        frac=0.005, random_state=42
    )

    drug_df, gene_df = preprocess_and_merge(
        drug_list, gdsc1, gdsc2, variance, mutations, all_mutations
    )

    print(" Pipeline complete. drug_df shape:", drug_df.shape)
    print(" gene_df shape:", gene_df.shape)
    print(drug_df.head())
    print(gene_df.head())
