#!/usr/bin/env python3
"""
Script to inspect column names and shapes for various raw data files.

Handles .csv, .tsv, and .xlsx formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def inspect_data_files(data_dir: str | Path = "data") -> Dict[str, pd.DataFrame]:
    """
    Load and print summary of data files in a directory.

    Args:
        data_dir: Directory containing data files.

    Returns:
        dict of {filename: pd.DataFrame}
    """
    data_path = Path(data_dir)
    file_extensions = {".csv", ".tsv", ".xlsx"}
    dataframes: Dict[str, pd.DataFrame] = {}

    for file in data_path.iterdir():
        if file.suffix not in file_extensions:
            continue

        print(f"\n Inspecting: {file.name}")

        try:
            if file.suffix == ".xlsx":
                df = pd.read_excel(file)
            else:
                sep = "\t" if file.suffix == ".tsv" else ","
                df = pd.read_csv(file, sep=sep)

            print(f"Columns: {list(df.columns)}")
            print(f"Shape: {df.shape}")
            dataframes[file.name] = df

        except Exception as exc:
            print(f"Could not load {file.name}: {exc}")

    return dataframes


if __name__ == "__main__":
    dfs = inspect_data_files()
