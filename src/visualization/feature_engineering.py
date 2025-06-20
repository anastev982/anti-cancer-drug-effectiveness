#!/usr/bin/env python3
"""
Feature engineering for the GDSC project.

- Calculates average intensity by drug_id.
- Scales gene columns (GeneA, GeneB, GeneC).
- Bins concentration values into quantiles.
"""

from __future__ import annotations

import logging

import dask.dataframe as dd
import pandas as pd
from sklearn.preprocessing import StandardScaler


_LOG = logging.getLogger("gdsc.feature_engineering")


def feature_engineering(df: dd.DataFrame | pd.DataFrame) -> dd.DataFrame | pd.DataFrame:
    """
    Apply feature engineering transformations on a DataFrame:
    - Compute average intensity per drug (if present).
    - Scale known gene features (GeneA, GeneB, GeneC).
    - Bin concentrations into quantiles.

    Args:
        df: Dask or pandas DataFrame.

    Returns:
        Updated DataFrame with new features.
    """
    if isinstance(df, dd.DataFrame):
        _LOG.info("Processing Dask DataFrame lazily — no early compute triggered.")
        return df  # Dask is typically for downstream ML; skip in-memory feature engineering

    if isinstance(df, pd.DataFrame):
        # Groupby average response
        if {"drug_id", "intensity"} <= set(df.columns):
            avg_response = df.groupby("drug_id")["intensity"].mean().reset_index()
            df = df.merge(avg_response, on="drug_id", suffixes=("", "_avg"))
        else:
            _LOG.warning("Missing 'drug_id' or 'intensity' for average computation.")

        # Scaling gene features
        scaler = StandardScaler()
        for gene in ["GeneA", "GeneB", "GeneC"]:
            if gene in df.columns:
                if df[gene].notna().sum() > 0:
                    gene_values = df[gene].fillna(0).values.reshape(-1, 1)
                    df[gene] = scaler.fit_transform(gene_values)
                else:
                    _LOG.warning("Gene %s only has NaNs; skipping scaling.", gene)

        # Binning concentration into quantiles
        if "conc" in df.columns and df["conc"].notna().sum() >= 3:
            df["conc_bin"] = pd.qcut(df["conc"], q=3, labels=["Low", "Medium", "High"])
        else:
            _LOG.warning("Missing or too few valid 'conc' values for binning.")

    return df
