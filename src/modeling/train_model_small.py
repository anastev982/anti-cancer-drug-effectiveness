import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Constants ===
DATA_PATH = "final_features/subset_preview.parquet"
TARGET_COL = "LN_IC50"


def load_data(path):
    logger.info("Loading dataset from: %s", path)
    df = pd.read_parquet(path)
    return df


def preprocess(df):
    logger.info("Preprocessing data...")
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y, X.columns.tolist()


def train_and_evaluate(X, y, feature_names):
    logger.info("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info("MSE on test set: %.4f", mse)

    # Plot feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.title("Top Feature Importances")
    plt.barh(range(len(feature_names)), importances[indices[::-1]])
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices[::-1]])
    plt.xlabel("Importance Score")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/feature_importance_small.png")
    logger.info("Feature importance plot saved to outputs/feature_importance_small.png")
    return model


def main():
    df = load_data(DATA_PATH)
    X, y, feature_names = preprocess(df)
    model = train_and_evaluate(X, y, feature_names)


if __name__ == "__main__":
    main()
