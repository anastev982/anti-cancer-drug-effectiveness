import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load the dataset ===
logger.info("Loading subset...")
df = pd.read_parquet("final_features/subset_preview.parquet")

# === Use only brca1 as feature ===
X = df[["brca1"]]
y = df["LN_IC50"]

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Train ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
logger.info(f"MSE using only brca1: {mse:.4f}")

# === Visualize simple prediction line (scatter) ===
plt.figure(figsize=(6, 6))
plt.scatter(X_test["brca1"], y_test, color="blue", label="True")
plt.scatter(X_test["brca1"], y_pred, color="red", alpha=0.6, label="Predicted")
plt.xlabel("brca1")
plt.ylabel("LN_IC50")
plt.title("Prediction vs. True (using only brca1)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/brca1_vs_ln_ic50.png")
plt.close()
