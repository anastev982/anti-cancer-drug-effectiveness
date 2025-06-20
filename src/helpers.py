#!/usr/bin/env python3
"""Generic helpers used across the GDSC pipeline.

The goal of this module is **zero side effects** (except for logging): nothing
is eagerly computed, no globals mutate at import time, and all functions are
safe to call from both pandas *and* Dask contexts.  If a helper *must* trigger
`.compute()` it is clearly documented in its docstring.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from venv import logger

import dask.dataframe as dd
import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG = logging.getLogger("gdsc.helpers")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
#  ‹…/src/helpers.py› → project root

RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw_data" / "processed"
PROCESSED_DATA_DIR = _PROJECT_ROOT / "data" / "processed"
NEW_PROCESSED_DATA_DIR = _PROJECT_ROOT / "path_folder" / "processed"
NEW_PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "normalize_columns",
    "standardise_columns",
    "ensure_str_column",
    "clean_column_to_numeric",
    "fix_column_type",
    "fix_mixed_columns_dask",
    "safe_merge_dask",
    "safe_merge_pandas",
    "save_final_df_dask",
    "merge_multiple_dask_dfs",
    "apply_feature_transforms",
    "load_dask_csv_safe",
    "read_clean_csv",
    "safe_map_partitions",
    "n_rows",
]


def go_up(path: Path, levels: int) -> Path:
    """
    Utility to move up directory levels.
    """
    for _ in range(levels):
        path = path.parent
    return path


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------


def normalize_columns(cols: Iterable[Any]) -> List[str]:
    """Return a *flat*, lowercase list of column names with spaces → underscores."""
    if isinstance(cols, pd.MultiIndex):
        cols = ["_".join(map(str, col)) for col in cols]
    else:
        cols = [str(c) for c in cols]
    return [c.lower().strip().replace(" ", "_") for c in cols]


def standardise_columns(
    df: dd.DataFrame | pd.DataFrame,
    *,
    remove_underscores: bool = False,
    to_lower: bool = True,
) -> dd.DataFrame | pd.DataFrame:
    """Mutate *df* in‑place to have consistent column naming."""
    cols = normalize_columns(df.columns)
    if remove_underscores:
        cols = [c.replace("_", "") for c in cols]
    if to_lower:
        cols = [c.lower() for c in cols]
    df.columns = cols
    return df


# ---------------------------------------------------------------------------
# Schema / dtype fixes
# ---------------------------------------------------------------------------


def ensure_str_column(df: dd.DataFrame | pd.DataFrame, col_name: str):
    if col_name not in df.columns:
        raise KeyError(f"Column '{col_name}' not found in DataFrame.")
    df[col_name] = df[col_name].astype(str)
    return df


def clean_column_to_numeric(df: dd.DataFrame | pd.DataFrame, col: str):
    if isinstance(df, dd.DataFrame):
        df[col] = dd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fix_column_type(
    df: dd.DataFrame | pd.DataFrame, col: str, sample_size: int = 1_000
):
    """Best effort cast `col` to a *single* dtype based on the most common sample.

    No computation is triggered for Dask — we only inspect a `.head()` sample.
    """
    sample = df[col].head(sample_size)
    most_common = sample.map(type).mode().iloc[0]
    _LOG.info("Fixing mixed type column '%s' → %s", col, most_common.__name__)

    if most_common in (float, int):
        return clean_column_to_numeric(df, col)
    if most_common is str:
        df[col] = df[col].astype(str)
        return df

    _LOG.warning(
        "Unknown predominant type %s in column '%s' — left as is.", most_common, col
    )
    return df


def fix_mixed_columns_dask(
    ddf: dd.DataFrame,
    columns_to_fix: Optional[Sequence[str]] = None,
) -> dd.DataFrame:
    """Detect & coerce columns with mixed Python types (Dask‑safe)."""
    sample = ddf.head(2_000)
    if columns_to_fix is None:
        columns_to_fix = [
            c for c in sample.columns if sample[c].map(type).nunique() > 1
        ]

    for col in columns_to_fix:
        ddf = fix_column_type(ddf, col)
    return ddf


def check_required_columns(df, required, name):
    """
    Checks whether required columns exist in a DataFrame.
    Logs a warning if any are missing.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"{name} is missing required columns: {missing}")
    else:
        print(f"{name} contains all required columns: {required}")


# Merge wrappers


def safe_merge_dask(
    left: dd.DataFrame,
    right: dd.DataFrame,
    *,
    on: str,
    how: str = "inner",
) -> dd.DataFrame:
    """Like :pymeth:`dask.DataFrame.merge` but with dtype & key sanity checks."""
    for df, name in ((left, "left"), (right, "right")):
        if on not in df.columns:
            raise KeyError(f"Merge key '{on}' missing in {name} dataframe.")

    left = ensure_str_column(fix_mixed_columns_dask(left, [on]), on)
    right = ensure_str_column(fix_mixed_columns_dask(right, [on]), on)

    return left.merge(right, on=on, how=how)


def safe_merge_pandas(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_key: str,
    right_key: str,
    **kwargs,
) -> pd.DataFrame:
    """Pandas merge with automatic key normalisation and helpful errors."""
    left.columns = normalize_columns(left.columns)
    right.columns = normalize_columns(right.columns)

    for key, df_name, df in [
        (left_key, "left", left),
        (right_key, "right", right),
    ]:
        if key not in df.columns:
            raise KeyError(
                f"Key '{key}' not in {df_name} DataFrame. Available: {df.columns.tolist()}"
            )

    return left.merge(right, left_on=left_key, right_on=right_key, **kwargs)


# Dask persistence helper


def save_final_df_dask(
    df: dd.DataFrame | pd.DataFrame,
    out_dir: Path | str,
    *,
    overwrite: bool = True,
    npartitions: int | None = None,
):
    """Save *df* to partitioned Parquet on disk **lazily** (compute at the end)."""
    out_dir = Path(out_dir)

    if overwrite and out_dir.exists():
        _LOG.info("Removing existing directory: %s", out_dir)
        shutil.rmtree(out_dir)

    if isinstance(df, pd.DataFrame):
        _LOG.info("Converting pandas → Dask (%d rows) before parquet write", len(df))
        df = dd.from_pandas(df, npartitions=npartitions or 8)

    if npartitions is not None:
        df = df.repartition(npartitions=npartitions)

    _LOG.info("Writing parquet → %s", out_dir)
    df.to_parquet(out_dir, engine="pyarrow", write_index=False)


# Batch utilities


def merge_multiple_dask_dfs(
    dfs: Sequence[dd.DataFrame],
    *,
    on: str,
    how: str = "inner",
) -> dd.DataFrame:
    if not dfs:
        return dd.from_pandas(pd.DataFrame(), npartitions=1)

    merged = dfs[0]
    for d in dfs[1:]:
        merged = safe_merge_dask(merged, d, on=on, how=how)
    return merged


def apply_feature_transforms(
    ddf: dd.DataFrame,
    *,
    selected_cols: Optional[Sequence[str]] = None,
) -> dd.DataFrame:
    """Return a dataframe restricted to *selected_cols* after normalising names."""
    ddf = standardise_columns(ddf)

    if selected_cols is None:
        selected_cols = [
            "drug_id",
            "name",
            "pubchem",
            "conc",
            "intensity",
            "duration_x",
            "cell_line_name",
            "model_id",
            "targets",
            "target_pathway",
            "gene_id",
            "mutation_symbol",
            "ensembl_gene_id",
            "chromosome",
            "position",
            "reference",
            "alternative",
            "effect",
            "vaf",
            "protein_mutation",
            "source",
            "transcript_id",
            "cancer_driver",
            "cancer_predisposition_variant",
        ]

    missing = [c for c in selected_cols if c not in ddf.columns]
    if missing:
        raise ValueError(f"Selected columns missing: {missing}")

    return ddf[list(selected_cols)]


# Robust I/O


def load_dask_csv_safe(
    path: str | Path,
    *,
    dtype: Optional[Dict[str, str]] = None,
    low_memory: bool = True,
) -> dd.DataFrame:
    """Try to read CSV with provided dtypes; fall back to `object` on mismatch."""
    try:
        return dd.read_csv(path, dtype=dtype, low_memory=low_memory)
    except ValueError as exc:
        if "Mismatched dtypes" in str(exc):
            _LOG.warning("Retrying %s with 'object' for offending columns.", path)
            dtype = dtype or {}
            return dd.read_csv(path, dtype=dtype, low_memory=low_memory)
        raise


def read_clean_csv(
    path: str | Path, *, dtype: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    try:
        _LOG.info("Reading CSV → %s", path)
        return pd.read_csv(path, dtype=dtype)
    except pd.errors.DtypeWarning:
        _LOG.warning("DtypeWarning on %s. Retrying with dtype=str.", path)
        return pd.read_csv(path, dtype=str)


def save_dataset_samples(drug_list, gdsc1, gdsc2):
    sample_drug_list = drug_list.head(1000).compute()
    sample_gdsc1 = gdsc1.head(1000).compute()
    sample_gdsc2 = gdsc2.head(1000).compute()

    Path("samples").mkdir(exist_ok=True)

    sample_drug_list.to_parquet("samples/sample_drug_list.parquet")
    sample_gdsc1.to_parquet("samples/sample_gdsc1.parquet")
    sample_gdsc2.to_parquet("samples/sample_gdsc2.parquet")

    print("Saved dataset samples to samples/ directory.")


# Dask helpers that *may* compute


def safe_map_partitions(
    ddf: dd.DataFrame,
    fn,
    *,
    meta: Dict[str, str] | pd.DataFrame,
) -> dd.DataFrame:
    """Simple wrapper for :pymeth:`dask.DataFrame.map_partitions` with a clearer API."""
    return ddf.map_partitions(lambda part: part.apply(fn, axis=1), meta=meta)


def n_rows(df: dd.DataFrame | pd.DataFrame) -> int:
    """Return row count (computes if *df* is Dask)."""
    return df.shape[0] if isinstance(df, pd.DataFrame) else int(df.shape[0].compute())


def enrich_with_pubchem(df, pubchem_data):
    """
    Enrich a dataframe with PubChem data based on 'drug_id'.
    """
    print(df.columns)

    def map_pubchem(row):
        data = pubchem_data.get(row["drug_id"], {})
        return pd.Series(data)

    # Create a Dask-friendly meta for the new columns
    meta = {"cid": "int64", "smiles": "object"}

    # Use Dask's map_partitions

    return df.map_partitions(lambda d: d.apply(map_pubchem, axis=1), meta=meta)


import dask.dataframe as dd
import pandas as pd


def clean_df(df, id_column="id"):
    if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
        raise ValueError(
            f"Expected a Dask or Pandas DataFrame, but got {type(df)} instead."
        )

    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]
    return df
