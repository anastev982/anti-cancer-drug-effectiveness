#!/usr/bin/env python3
"""
Load and merge core datasets for the GDSC project.

Uses Dask for large CSVs and pandas for small metadata files.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import dask.dataframe as dd


def load_data(
    gdsc1_path: str | Path,
    gdsc2_path: str | Path,
    drug_list_path: str | Path,
) -> tuple[dd.DataFrame, dd.DataFrame, pd.DataFrame]:
    """
    Load the GDSC1, GDSC2, and drug list datasets.

    - GDSC1 & GDSC2 as Dask DataFrames (lazy, memory-efficient).
    - Drug list as pandas DataFrame (small enough for in-memory use).

    Args:
        gdsc1_path: Path to GDSC1 CSV.
        gdsc2_path: Path to GDSC2 CSV.
        drug_list_path: Path to drug list CSV.

    Returns:
        Tuple of (gdsc1, gdsc2, drug_list).
    """
    gdsc1 = dd.read_csv(
        gdsc1_path,
        blocksize="64MB",
        dtype={"BARCODE": "object", "seeding_density": "float64"},
    )
    gdsc2 = dd.read_csv(
        gdsc2_path,
        blocksize="64MB",
        dtype={"BARCODE": "object", "seeding_density": "float64"},
    )
    drug_list = pd.read_csv(drug_list_path)

    return gdsc1, gdsc2, drug_list


def merge_data(
    gdsc1: dd.DataFrame, gdsc2: dd.DataFrame, drug_list: pd.DataFrame
) -> dd.DataFrame:
    """
    Merge GDSC1, GDSC2, and the drug list.

    - Concatenates GDSC1 and GDSC2.
    - Joins with drug list (converted to Dask).

    Args:
        gdsc1: Dask DataFrame.
        gdsc2: Dask DataFrame.
        drug_list: pandas DataFrame.

    Returns:
        Merged Dask DataFrame (call .compute() downstream).
    """
    merged = dd.concat([gdsc1, gdsc2], axis=0, interleave_partitions=True)

    # Convert drug_list to Dask for merging
    drug_list_dd = dd.from_pandas(drug_list.set_index("DRUG_ID"), npartitions=1)

    # Join on index
    merged = merged.join(drug_list_dd, how="left")

    return merged
