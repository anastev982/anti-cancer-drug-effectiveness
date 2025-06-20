#!/usr/bin/env python3
"""Utility helpers for merging GDSC-related tables with **very** large row-counts.

All helpers return **lazy** :class:`dask.dataframe.DataFrame` objects so nothing is
materialised in memory until absolutely required (e.g. when saving or modelling).
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Iterable, Sequence

import dask
import dask.dataframe as dd
import pandas as pd

from src.constants import DRUG_COLUMNS, GDSC_COLUMNS, GENE_ID

__all__ = [
    "merged_drug_data",
    "merged_gene_data",
    "merged_all_data",
]

# Adding root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

_LOG = logging.getLogger("gdsc.merge_utils")


# Helper functions


def preprocess_df(
    df: dd.DataFrame | pd.DataFrame,
    expected_cols: Sequence[str],
    key_col: str = "drug_id",
    df_name: str = "unknown",
) -> dd.DataFrame:
    """Clean, validate, and normalize dataframe before merge."""
    df = _ensure_dask(df)
    _LOG.info("Preprocessing: %s", df_name)

    # Fill in any missing expected columns
    df = _fill_missing_columns(df, expected_cols)

    # Drop rows with missing key column (e.g., drug_id)
    df = df.dropna(subset=[key_col])

    # Drop duplicates
    df = df.drop_duplicates(subset=[key_col])

    # Normalize key column as string
    df[key_col] = df[key_col].astype(str)

    return df


def _ensure_dask(
    df: dd.DataFrame | pd.DataFrame, npartitions: int = 32
) -> dd.DataFrame:
    """Convert *pandas* input to :pymod:`dask` so the whole pipeline stays lazy."""
    if isinstance(df, dd.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return dd.from_pandas(df, npartitions=npartitions)
    raise TypeError(f"Unsupported dataframe type: {type(df)}")


def _validate_columns(df: dd.DataFrame, expected: Sequence[str], df_name: str) -> None:
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {df_name}")


def _fill_missing_columns(df: dd.DataFrame, cols: Iterable[str]) -> dd.DataFrame:
    """Add any columns from *cols* that don’t exist in *df* (filled with NA/-1)."""
    for c in cols:
        if c not in df.columns:
            default = -1 if c.endswith("_id") else ""
            _LOG.warning("%s missing — filling with %s", c, default)
            df[c] = default
    return df


def safe_fillna(df: dd.DataFrame, col: str, default: str | int) -> dd.DataFrame:
    if col not in df.columns:
        _LOG.warning("%s missing — adding with default %s", col, default)
        df[col] = default
    else:
        # If the dtype is string or object, convert default to string
        if "string" in str(df[col].dtype) or df[col].dtype == "object":
            default = str(default)
        df[col] = df[col].fillna(default)
    return df


def save_separately(
    drug_df: dd.DataFrame,
    gene_df: dd.DataFrame,
    out_dir: str | Path = "outputs/",
) -> None:
    """Save drug & gene datasets separately as Parquet for later modeling."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    drug_df.to_parquet(
        Path(out_dir) / "drug_features.parquet", engine="pyarrow", overwrite=True
    )
    gene_df.to_parquet(
        Path(out_dir) / "gene_features.parquet", engine="pyarrow", overwrite=True
    )

    _LOG.info("Saved drug & gene datasets separately.")


# Public API


def merged_drug_data(drug_list, gdsc1, gdsc2):
    expected_cols = [
        "drug_id",
        "conc",
        "intensity",
        "cell_line_name",
        "date_created",
        "drugset_id",
        "duration",
    ]

    gdsc1 = preprocess_df(gdsc1, expected_cols, df_name="GDSC1")
    gdsc2 = preprocess_df(gdsc2, expected_cols, df_name="GDSC2")
    drug_list = preprocess_df(
        drug_list, ["drug_id", "name", "pubchem"], df_name="Drug List"
    )

    # Convert drug_list to pandas for merging
    drug_list_pd = (
        drug_list.compute() if isinstance(drug_list, dd.DataFrame) else drug_list
    )

    # Merge into each GDSC dataset
    g1 = gdsc1.map_partitions(
        lambda part: part.merge(drug_list_pd, on="drug_id", how="left")
    )
    g2 = gdsc2.map_partitions(
        lambda part: part.merge(drug_list_pd, on="drug_id", how="left")
    )

    # Final outer merge
    with dask.config.set({"dataframe.shuffle.algorithm": "tasks"}):
        merged = g1.merge(g2, on="drug_id", how="outer", suffixes=("_g1", "_g2"))

    return merged


def merged_gene_data(
    variance: dd.DataFrame | pd.DataFrame,
    mutations: dd.DataFrame | pd.DataFrame,
    all_mutations: dd.DataFrame | pd.DataFrame,
    out_path: str | Path = "outputs/gene_features.parquet",
) -> dd.DataFrame:
    """Outer-join the three *gene* tables on the constant ``GENE_ID`` (typically "gene_id")."""

    v = _ensure_dask(variance)
    m = _ensure_dask(mutations)
    am = _ensure_dask(all_mutations)

    for name, df in zip(["variance", "mutations", "all_mutations"], [v, m, am]):
        _validate_columns(df, [GENE_ID], name)

    # Sample before merging to keep memory use low
    v_sample = v.sample(frac=0.1, random_state=42)
    m_sample = m.sample(frac=0.1, random_state=42)
    am_sample = am.sample(frac=0.1, random_state=42)

    _LOG.info("Merging sampled gene dataframes…")

    merged = v_sample.merge(
        m_sample, on=GENE_ID, how="outer", suffixes=("_var", "_mut")
    )
    merged = merged.merge(am_sample, on=GENE_ID, how="outer")

    # Save directly to Parquet instead of persisting
    _LOG.info("Saving gene features to: %s", out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, engine="pyarrow", overwrite=True)

    _LOG.info(
        "Gene merge completed with %d columns and %d partitions",
        merged.shape[1],
        merged.npartitions,
    )

    return merged
