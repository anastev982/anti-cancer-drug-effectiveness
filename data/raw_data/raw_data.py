#!/usr/bin/env python3
"""
Process raw CSV and ZIP files of FPKM data and merge them into a single CSV.

Key steps:
- Load CSVs & ZIPs with CSVs inside.
- Drop unneeded columns.
- Merge them into one large Dask dataframe.
- Drop rows with missing crucial data.
- Save as a single merged CSV.
"""

from __future__ import annotations

import glob
import os
import zipfile
from pathlib import Path

import dask.dataframe as dd


def load_and_merge_files(folder: str | Path) -> dd.DataFrame:
    """
    Load and merge all CSV and ZIP-contained CSV files in the given folder.

    Args:
        folder: Path to the folder.

    Returns:
        Merged Dask DataFrame.
    """
    folder = Path(folder)
    csv_files = list(folder.glob("*.csv"))
    zip_files = list(folder.glob("*.zip"))

    all_dfs = []

    # Load CSV files directly
    for file in csv_files:
        df = dd.read_csv(file, low_memory=False, dtype={"SCAN_DATE": "object"})
        df = df.drop(columns=["BARCODE"], errors="ignore")
        all_dfs.append(df)

    # Load CSV files from ZIP archives
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file) as z:
            for filename in z.namelist():
                if filename.endswith(".csv"):
                    with z.open(filename) as f:
                        df = dd.read_csv(f, low_memory=False, dtype={"SCAN_DATE": "object"})
                        df = df.drop(columns=["BARCODE"], errors="ignore")
                        all_dfs.append(df)

    # Merge all into one Dask DataFrame
    merged_df = dd.concat(all_dfs, axis=0, ignore_unknown_divisions=True)
    return merged_df


def main() -> None:
    folder_path = Path(
        r"C:\Users\steva\OneDrive\Programiranje\Neural Networks\Projects\Predicting Anti-Cancer Drug Effectiveness from Molecular Data\data\fpkm_files\row_data"
    )
    merged_df = load_and_merge_files(folder_path)

    # Drop rows with missing crucial data
    merged_df = merged_df.dropna(subset=["DRUG_ID", "CELL_LINE_NAME", "INTENSITY"])

    # Save to CSV
    out_file = "merged_clean_data.csv"
    print(f"Saving merged dataset to: {out_file}")
    merged_df.compute().to_csv(out_file, index=False)

    # Basic inspection
    df = merged_df.compute()
    print("First few rows:")
    print(df.head())

    print("\nShape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nBasic stats:")
    print(df.describe())

    # Quick intensity histogram
    df["INTENSITY"].hist(bins=50).figure.show()


if __name__ == "__main__":
    main()
