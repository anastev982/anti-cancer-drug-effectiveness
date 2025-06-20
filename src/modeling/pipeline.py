#!/usr/bin/env python3
"""
Builds a complete scikit-learn pipeline for the GDSC project.

Handles:
- Preprocessing numeric & categorical features.
- Missing value imputation.
- GradientBoostingRegressor model training.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline() -> Pipeline:
    """
    Build a full ML pipeline (preprocessing + regressor).

    Returns:
        scikit-learn Pipeline
    """
    numeric_features = ["conc"]
    categorical_features = ["cell_line_name", "assay"]

    # Preprocessing transformers
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    # Final ML pipeline
    pipeline = Pipeline(
        [
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
