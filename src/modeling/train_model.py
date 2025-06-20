import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from scipy import sparse
from sklearn.model_selection import GridSearchCV
import dask.dataframe as dd
import pyarrow.parquet as pq

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Constants ===
INPUT_PATH = Path("final_features/full_model_input.parquet")
TARGET_COL = "LN_IC50"
NON_FEATURE_COLS = [
    "DATASET",
    "NLME_RESULT_ID",
    "NLME_CURVE_ID",
    "COSMIC_ID",
    "CELL_LINE_NAME",
    "SANGER_MODEL_ID",
    "TCGA_DESC",
    "DRUG_ID",
    "DRUG_NAME",
    "PUTATIVE_TARGET",
    "PATHWAY_NAME",
    "COMPANY_ID",
    "WEBRELEASE",
    "MIN_CONC",
    "MAX_CONC",
    "LN_IC50",
    "AUC",
    "RMSE",
    "Z_SCORE",
]


def load_data(path: Path, sample_n: int = 2000) -> pd.DataFrame:
    logger.info("Loading dataset from: %s", path)

    # Load schema only (no memory load)
    all_columns = pd.read_parquet(path, engine="pyarrow", columns=None).columns.tolist()

    # Target must be there
    if "LN_IC50" not in all_columns:
        raise ValueError("Target column 'LN_IC50' is missing from the dataset.")

    # Define columns to keep: target + features (everything else)
    metadata_cols = set(
        [
            "DATASET",
            "NLME_RESULT_ID",
            "NLME_CURVE_ID",
            "COSMIC_ID",
            "CELL_LINE_NAME",
            "SANGER_MODEL_ID",
            "TCGA_DESC",
            "DRUG_ID",
            "DRUG_NAME",
            "PUTATIVE_TARGET",
            "PATHWAY_NAME",
            "COMPANY_ID",
            "WEBRELEASE",
            "MIN_CONC",
            "MAX_CONC",
            "AUC",
            "RMSE",
            "Z_SCORE",
        ]
    )

    feature_cols = [
        col for col in all_columns if col not in metadata_cols and col != "LN_IC50"
    ]
    selected_cols = ["LN_IC50"] + feature_cols

    logger.info(f"Found {len(feature_cols)} feature columns.")

    # Load only needed columns
    ddf = dd.read_parquet(path, columns=selected_cols)

    logger.info("Sampling first %d rows...", sample_n)
    df = ddf.head(sample_n)

    return df


def preprocess_data(df: pd.DataFrame):
    logger.info("Preprocessing dataset...")

    y = df["LN_IC50"]
    X = df.drop(columns=["LN_IC50"])
    feature_names = X.columns.tolist()

    logger.info("Feature matrix shape: %s", X.shape)
    return X, y, feature_names


def train_model(X, y, feature_names):
    logger.info("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info("MSE on test set: %.4f", mse)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "max_features": ["sqrt", "log2"],
    }

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    print("Starting hyperparameter tuning…")
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best CV score (R²):", grid_search.best_score_)

    # === Feature importances ===
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_k = 20
    top_features = [feature_names[i] for i in indices[:top_k]]
    top_importances = importances[indices[:top_k]]

    plt.figure(figsize=(10, 6))
    plt.title("Top Feature Importances")
    plt.barh(range(top_k), top_importances[::-1], align="center")
    plt.yticks(range(top_k), top_features[::-1])
    plt.xlabel("Importance Score")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/feature_importance.png")
    plt.close()
    print("Feature importance plot saved to: outputs/feature_importance.png")

    return model


def main():
    df = load_data(INPUT_PATH)
    X, y, feature_names = preprocess_data(df)
    model = train_model(X, y, feature_names)
    logger.info("Model trained successfully.")


if __name__ == "__main__":
    main()
