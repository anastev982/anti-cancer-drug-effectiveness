#!/usr/bin/env python3
"""
Data loading utilities for the GDSC project.

Handles:
- Fast Parquet caching of CSV files.
- Normalising column names.
- Merging chunked Feather or Parquet datasets.
- Final safe load for downstream steps.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from sys import path
from typing import Dict, Optional

import dask.dataframe as dd
import pandas as pd

from src.config import RAW_PROCESSED_DIR
from src.helpers import normalize_columns

_LOG = logging.getLogger("gdsc.dataloader")


# Core loading functions
def load_csv(path: Path, dtypes: Optional[Dict[str, str]] = None) -> dd.DataFrame:
    print("Using load_csv from:", inspect.getfile(load_csv))
    if dtypes is None:
        dtypes = {}
    # Normalize known problematic columns
    dtypes.setdefault(" pubchem", "object")
    dtypes.setdefault("barcode", "object")
    dtypes.setdefault("drug_id", "string")
    dtypes.setdefault("cell_line_name", "object")
    dtypes.setdefault("chr_name", "object")

    try:
        _LOG.info("Loading CSV: %s", path)
        dtypes = {k.strip().lower(): v for k, v in dtypes.items()}
        df = dd.read_csv(
            str(path),
            dtype={
                "drug_id": "object",
                "cell_line_name": "object",
                "conc": "float64",
                "intensity": "float64",
                "date_created": "object",
                " PubCHEM": "object",
                "BARCODE": "object",
                "chr_name": "object",
            },
            assume_missing=True,
            blocksize="64MB",
            low_memory=False,
        )

        df.columns = normalize_columns(df.columns)
        return df
    except Exception as exc:
        _LOG.warning("Could not load CSV %s: %s", path, exc)
        return dd.from_pandas(pd.DataFrame(), npartitions=1)


def load_chunks_from_dir(dir_path: Path) -> pd.DataFrame:
    """
    Merge chunked Parquet datasets in `dir_path` into one pandas dataframe.
    """
    chunk_files = sorted(dir_path.glob("chunk_*.parquet"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {dir_path}")

    _LOG.info("Loading %d chunks from %s", len(chunk_files), dir_path)
    dfs = [pd.read_parquet(f) for f in chunk_files]
    final_df = pd.concat(dfs, ignore_index=True)
    _LOG.info("Chunks successfully loaded and combined.")
    return final_df


def load_feather_chunks(dir_path: Path) -> pd.DataFrame:
    """
    Merge chunked Feather datasets in `dir_path` into one pandas dataframe.
    """
    chunk_files = sorted(dir_path.glob("chunk_*.feather"))
    if not chunk_files:
        raise FileNotFoundError(f"No Feather chunk files found in {dir_path}")

    _LOG.info("Loading %d Feather chunks from %s", len(chunk_files), dir_path)
    dfs = [pd.read_feather(f) for f in chunk_files]
    return pd.concat(dfs, ignore_index=True)


def load_final_df(out_dir: Path) -> dd.DataFrame:
    """
    Load the final Parquet dataset as a Dask dataframe.
    """
    _LOG.info("Loading Parquet dataset from %s", out_dir)
    return dd.read_parquet(out_dir)


# Bulk dataset loader


def load_and_prepare_data() -> Dict[str, dd.DataFrame]:
    # Remove `dtype="object"` from here!
    df_drug_list = load_csv(RAW_PROCESSED_DIR / "drug_list.csv")
    df_gdsc1 = load_csv(RAW_PROCESSED_DIR / "gdsc1.csv")
    df_gdsc2 = load_csv(RAW_PROCESSED_DIR / "gdsc2.csv")
    print("Columns in gdsc2:", df_gdsc2.columns)

    dtype_gene = {"gene_id": "object", "chr_name": "object", "pep_coord": "float64"}
    df_variance = load_csv(RAW_PROCESSED_DIR / "variance.csv", dtypes=dtype_gene)
    df_mutations = load_csv(RAW_PROCESSED_DIR / "mutations.csv", dtypes=dtype_gene)
    df_all_mutations = load_csv(
        RAW_PROCESSED_DIR / "mutations_all.csv", dtypes=dtype_gene
    )

    # Before normalization logs
    _LOG.info("Before normalization: %s", df_drug_list.columns.tolist())

    return {
        "drug_list": df_drug_list,
        "gdsc1": df_gdsc1,
        "gdsc2": df_gdsc2,
        "variance": df_variance,
        "mutations": df_mutations,
        "all_mutations": df_all_mutations,
    }


def load_sampled_data() -> dict[str, dd.DataFrame]:
    """
    Load only relevant columns & sample for lightweight testing.
    """

    # Relevant columns for each dataset
    drug_list_cols = ["drug_id", "name", "pubchem"]
    gdsc1_cols = ["drug_id", "seeding_density", "drugset_id", "assay", "duration"]
    gdsc2_cols = ["tag", "drug_id", "conc", "intensity"]
    variance_cols = ["gene_id", "symbol", "effect", "ensembl_transcript"]
    mutations_cols = ["gene_id", "symbol", "effect"]
    all_mutations_cols = ["gene_id", "rna_mutation", "chromosome"]

    def load_and_sample(file: str, cols: list[str]) -> dd.DataFrame:
        df = dd.read_csv(
            str(path),
            dtype={"drug_id": "object", "cell_line_name": "object"},
            assume_missing=True,
            blocksize="64MB",
            low_memory=False,
        )

        df.columns = normalize_columns(df.columns)
        print(df.columns.tolist())
        print(f"Columns in {file}: {df.columns.tolist()}")
        df = df[cols]
        df = df.sample(frac=0.05)  # Adjust sample size!
        return df

    return {
        "drug_list": load_and_sample("drug_list.csv", drug_list_cols),
        "gdsc1": load_and_sample("gdsc1.csv", gdsc1_cols),
        "gdsc2": load_and_sample("gdsc2.csv", gdsc2_cols),
        "variance": load_and_sample("variance.csv", variance_cols),
        "mutations": load_and_sample("mutations.csv", mutations_cols),
        "all_mutations": load_and_sample("mutations_all.csv", all_mutations_cols),
    }


if __name__ == "__main__":
    print("Running dataloader test...")
    df = load_csv("gdsc1.csv")
    print(df.head())
