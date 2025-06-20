# src/interpretability/shap_analysis.py

import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === Load a specific chunk (e.g. chunk_14) ===
df = pd.read_parquet("final_chunks/chunk_14.parquet")

# === Use top combo genes ===
genes = ["lrrc37a2", "mstn"]
X = df[genes]
y = df["LN_IC50"]

# === Train the model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === SHAP ===
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# === Plot summary ===
shap.summary_plot(shap_values, X_test)
