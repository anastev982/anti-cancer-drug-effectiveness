#!/usr/bin/env python3
"""
Builds a complete scikit-learn pipeline for the GDSC project.

The pipeline:
- Preprocesses numeric & categorical features.
- Handles missing values.
- Trains a GradientBoostingRegressor model.
"""

from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline() -> Pipeline:
    """
    Build a full ML pipeline (preprocessing + regressor).

    Returns:
        A scikit-learn Pipeline instance.
    """
    numeric_features: List[str] = ["conc"]
    categorical_features: List[str] = ["cell_line_name", "assay"]

    # Preprocessing for numeric and categorical features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    # Final ML pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=80,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]
    )

    return pipeline
