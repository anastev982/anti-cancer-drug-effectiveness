#!/usr/bin/env python3
"""
Preprocessing pipeline for ML models.

Handles:
- Column normalization
- Feature selection
- Train-test split
- Logging of dataset shape and column status
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.helpers import normalize_columns

_LOG = logging.getLogger("gdsc.preprocessing")


# Preprocessing utilities


def get_preprocessor() -> ColumnTransformer:
    """
    Return a ColumnTransformer for numeric and categorical features.
    """
    numeric_features = ["conc"]
    categorical_features = ["cell_line_name", "assay"]

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return transformer


def preprocess_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocess a dataframe for ML:
    - Normalise column names.
    - Select features & target.
    - Train/test split.

    Returns:
        x_train, x_test, y_train, y_test
    """
    _LOG.info("Before cleaning: %s", df.columns.tolist())
    df.columns = normalize_columns(df.columns)
    _LOG.info("After cleaning: %s", df.columns.tolist())

    # Validate presence of 'drug_id'
    if "drug_id" not in df.columns:
        _LOG.warning("'drug_id' column not found after normalization.")
    else:
        _LOG.info("Unique drug_id entries: %d", df["drug_id"].nunique())

    # Expected columns
    features = ["date_created", "drug_id", "drugset_id", "duration"]
    target = ["intensity"]

    # Check for missing columns
    missing_cols = [c for c in features + target if c not in df.columns]
    if missing_cols:
        _LOG.warning("Missing expected columns: %s", missing_cols)

    # Only select available columns
    x = df[[c for c in features if c in df.columns]]
    y = df[[c for c in target if c in df.columns]]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    return x_train, x_test, y_train, y_test
