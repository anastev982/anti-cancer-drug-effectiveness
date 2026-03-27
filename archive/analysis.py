#!/usr/bin/env python3
"""
Quick analysis of raw dataset column structures.

Loads CSVs via Dask (lazy!) and prints column names for inspection.
"""

from __future__ import annotations

import dask.dataframe as dd
from pathlib import Path

from src.config import RAW_PROCESSED_DIR


def main() -> None:
    file_paths = [
        RAW_PROCESSED_DIR / "gdsc1.csv",
        RAW_PROCESSED_DIR / "gdsc2.csv",
        RAW_PROCESSED_DIR / "drug_list.csv",
        RAW_PROCESSED_DIR / "variance.csv",
        RAW_PROCESSED_DIR / "mutations.csv",
        RAW_PROCESSED_DIR / "mutations_all.csv",
    ]

    for path in file_paths:
        try:
            df = dd.read_csv(path, dtype=str, assume_missing=True)
            print(f"Columns in {path.name}:")
            print(df.columns.tolist())
            print("-" * 40)
        except Exception as exc:
            print(f"Error reading {path}: {exc}")


if __name__ == "__main__":
    main()
