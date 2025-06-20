#!/usr/bin/env python3
"""
Data cleaning and loading pipeline for the GDSC project.

Responsibilities:
- Load CSV and Parquet datasets (with type fallback for Dask).
- Check for missing files and handle them gracefully.
- Enrich the drug list with dummy PubChem data.
- Apply feature engineering if run as script.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import dask.dataframe as dd
import pandas as pd

from src.config import RAW_PROCESSED_DIR
from src.dataloader import load_csv
from src.helpers import enrich_with_pubchem, normalize_columns, save_dataset_samples
from src.visualization.feature_engineering import feature_engineering
import inspect


_LOG = logging.getLogger("gdsc.clean_data")


# Utility functions


def load_parquet(path: Path) -> Optional[dd.DataFrame]:
    """
    Load a Parquet file as a Dask DataFrame.
    """
    try:
        _LOG.info("Loading Parquet: %s", path)
        return dd.read_parquet(path)
    except FileNotFoundError:
        _LOG.error("Missing Parquet file: %s", path)
        return None


# Main cleaning pipeline


def load_and_clean_data() -> Dict[str, Optional[dd.DataFrame]]:
    """
    Load and clean all required datasets from RAW_PROCESSED_DIR.

    Returns:
        dict of {dataset_name: dd.DataFrame or None}
    Raises:
        FileNotFoundError if any critical datasets are missing.
    """
    datasets = {
        "merged_data": ("merged_clean_data.parquet", "parquet"),
        "gdsc1": ("gdsc1.csv", "csv"),
        "gdsc2": ("gdsc2.csv", "csv"),
        "drug_list": ("drug_list.csv", "csv"),
        "variance": ("variance.csv", "csv"),
        "mutations": ("mutations.csv", "csv"),
        "all_mutations": ("mutations_all.csv", "csv"),
    }

    loaded: Dict[str, Optional[dd.DataFrame]] = {}
    missing_files: list[str] = []

    for key, (fname, ftype) in datasets.items():
        path = RAW_PROCESSED_DIR / fname
        if not path.exists():
            _LOG.error("Missing file: %s", path)
            missing_files.append(fname)
            loaded[key] = None
            continue

        if ftype == "csv":
            loaded[key] = load_csv(path)
            print(f"Loaded {key}: type={type(loaded[key])}")  # 💡 Add this!
        elif ftype == "parquet":
            loaded[key] = load_parquet(path)
        else:
            raise ValueError(f"Unsupported file type: {ftype}")

    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}")

    print(type(loaded["drug_list"]))

    if not isinstance(loaded["drug_list"], dd.DataFrame):
        raise TypeError("drug_list is not a Dask DataFrame! Check load_csv logic.")

    # Dummy PubChem enrichment for demonstration
    pubchem_stub = {
        "DrugA": {"cid": 123, "smiles": "C1CCCCC1"},
        "DrugB": {"cid": 456, "smiles": "C1=CC=CC=C1"},
    }
    for key in ("gdsc1", "gdsc2", "drug_list"):
        df = loaded.get(key)
        if isinstance(df, dd.DataFrame):
            if "DRUG_ID" in df.columns:
                enrichment_data = enrich_with_pubchem(df, pubchem_stub)
                df = dd.concat([df, enrichment_data], axis=1)
                loaded[key] = df
        else:
            print(f"Skipping enrichment: df is not a DataFrame (type={type(df)})")

    return loaded


# CLI entry point

if __name__ == "__main__":
    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        data = load_and_clean_data()
    except FileNotFoundError as e:
        _LOG.error("Data load failed: %s", e)
        print(e)
        exit(1)

    df = data.get("merged_data")
    if df is None or df.shape[0].compute() == 0:
        _LOG.warning("Merged data missing or empty — skipping feature engineering.")
        print("Data loading failed or is empty. Skipping feature engineering.")
        exit(1)
    save_dataset_samples(data.get("drug_list"), data.get("gdsc1"), data.get("gdsc2"))

    try:
        df = feature_engineering(df)
        _LOG.info("Feature engineering complete.")
        print(df.head())
    except Exception as e:
        _LOG.error("Error during feature engineering: %s", e)
        print(f"Feature engineering error: {e}")
        exit(1)
