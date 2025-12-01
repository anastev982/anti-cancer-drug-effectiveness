import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === Load full dataset and combos ===
df = pd.read_parquet("final_features/full_model_input.parquet")
combos_df = pd.read_csv("outputs/top_50_gene_combos.csv")
combos_df["genes"] = combos_df["genes"].apply(lambda x: x.split(","))

y = df["LN_IC50"]
results = []

for genes in combos_df["genes"]:
    cols = [f"mutation_{g}" for g in genes if f"mutation_{g}" in df.columns]
    if not cols:
        continue
    X = df[cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    results.append({"genes": ",".join(genes), "num_genes": len(genes), "mse": mse})

# === Save and preview results ===
results_df = pd.DataFrame(results).sort_values("mse")
results_df.to_csv("outputs/eval_top_50_on_full.csv", index=False)
print("Evaluation complete. Best combos:")
print(results_df.head(10))
