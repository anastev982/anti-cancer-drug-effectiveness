#!/usr/bin/env python3
"""
Convert a CSV file to Parquet format for faster downstream processing.

Example:
    python convert_to_parquet.py data/raw_data/processed/merged_clean_data.csv
"""

from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd


def convert_csv_to_parquet(
    csv_path: str | Path, parquet_path: str | Path | None = None
) -> None:
    """
    Convert a CSV file to Parquet format.

    Args:
        csv_path: Path to the CSV file.
        parquet_path: Optional path for output Parquet file (default: same basename with .parquet extension).
    """
    csv_path = Path(csv_path)
    parquet_path = (
        Path(parquet_path) if parquet_path else csv_path.with_suffix(".parquet")
    )

    print(f"Reading CSV: {csv_path}")
    df = dd.read_csv(csv_path, assume_missing=True, low_memory=False)

    print(f"Saving to Parquet: {parquet_path}")
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

    print("Conversion complete!")


if __name__ == "__main__":
    convert_csv_to_parquet("data/raw_data/processed/merged_clean_data.csv")
