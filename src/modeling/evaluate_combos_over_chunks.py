import os
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CHUNK_DIR = "final_chunks"
GENES = [
    "fcrl6",
    "osbp2",
    "bambi",
    "lrrc37a2",
    "fyn",
    "tmt1a",
    "aifm2",
    "mstn",
    "sfmbt1",
    "pfn2",
]
OUT_PATH = "outputs/top_combos_per_chunk.csv"

results = []

# Loop over each chunk
for fname in sorted(os.listdir(CHUNK_DIR)):
    if fname.endswith(".parquet"):
        chunk_path = os.path.join(CHUNK_DIR, fname)
        df = pd.read_parquet(chunk_path)

        logger.info(f"Evaluating: {fname}")

        y = df["LN_IC50"]

        for k in range(2, 11):
            for combo in combinations(GENES, k):
                X = df[list(combo)]
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    model = RandomForestRegressor(random_state=42, n_estimators=100)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    mse = mean_squared_error(y_test, preds)

                    results.append(
                        {
                            "chunk": fname,
                            "genes": ",".join(combo),
                            "num_genes": k,
                            "mse": mse,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Skipping combo {combo} in {fname}: {e}")

# Save results
pd.DataFrame(results).to_csv(OUT_PATH, index=False)
logger.info(f"\nSaved all results to: {OUT_PATH}")
