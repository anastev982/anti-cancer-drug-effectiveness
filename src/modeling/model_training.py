#!/usr/bin/env python3
"""
Model evaluation and training pipeline for GDSC project.

Includes:
- Hyperparameter tuning
- Cross-validation
- Model evaluation metrics
"""

from __future__ import annotations

import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.modeling.pipeline import build_pipeline


# Evaluation utilities


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model on the test set and save the trained model.
    """
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    score = model.score(x_test, y_test)

    joblib.dump(model, "best_model.pkl")
    return mse, r2, score


def display_results(df, mse, score, r2, cv_scores):
    """
    Display key evaluation metrics and top features.
    """
    print(f"Score on Test Set: {score:.3f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.4f}")

    print("\nTop 10 Cell Lines:")
    print(df["cell_line_name"].value_counts().head(10))

    print("\nTop 10 Assays:")
    print(df["assay"].value_counts().head(10))

    print("\nCV Mean Score: {:.3f}".format(cv_scores.mean()))
    print("CV Std Dev: {:.3f}".format(cv_scores.std()))


def train_and_evaluate_model(x_train, y_train, x_test, y_test):
    """
    Train the ML pipeline, perform grid search, cross-validation,
    and evaluate on the test set.
    """
    model = build_pipeline()

    param_grid = {
        "regressor__n_estimators": [50, 100, 200],
        "regressor__learning_rate": [0.01, 0.1, 0.2],
        "regressor__max_depth": [3, 5, 7],
        "regressor__min_samples_split": [2, 5, 10],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
        verbose=2,
    )
    grid_search.fit(x_train, y_train.values.ravel())

    best_model = grid_search.best_estimator_

    cv_scores = cross_val_score(
        best_model,
        x_train,
        y_train.values.ravel(),
        cv=5,
        scoring="neg_mean_squared_error",
    )

    mse, r2, score = evaluate_model(best_model, x_test, y_test)

    return best_model, cv_scores, mse, score, r2
