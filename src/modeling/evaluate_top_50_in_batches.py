import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import ast

# === Load gene combos ===
combos_df = pd.read_csv("outputs/top_50_gene_combos.csv")

# Handle format of gene lists
if isinstance(combos_df["genes"].iloc[0], str):
    try:
        combos_df["genes"] = combos_df["genes"].apply(ast.literal_eval)
    except:
        combos_df["genes"] = combos_df["genes"].apply(lambda x: x.split(","))

# === Extract all unique genes ===
all_genes = set(g.strip() for genes in combos_df["genes"] for g in genes)
use_columns = list(all_genes) + ["LN_IC50"]

print(f"Total unique genes: {len(all_genes)}")

# === Load only necessary columns ===
df = pd.read_parquet("final_features/full_model_input.parquet", columns=use_columns)
print(f"df shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:10]}...")

y = df["LN_IC50"]
all_results = []

# === Batch size ===
batch_size = 10

for start in range(0, len(combos_df), batch_size):
    batch = combos_df.iloc[start : start + batch_size]
    batch_id = start // batch_size + 1
    print(f"\n=== Processing batch {batch_id} ===")

    for genes in batch["genes"]:
        cols = [g for g in genes if g in df.columns]
        print(f"  Genes: {genes} → found: {cols}")

        # Skip if no usable features
        if len(cols) < 2:
            print(f"  Skipping {genes} — not enough usable columns")
            all_results.append(
                {
                    "genes": ",".join(genes),
                    "num_genes": len(genes),
                    "used_genes": ",".join(cols),
                    "mse": None,
                    "r2": None,
                }
            )
            continue

        # Train model
        X = df[cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        all_results.append(
            {
                "genes": ",".join(genes),
                "num_genes": len(genes),
                "used_genes": ",".join(cols),
                "mse": mse,
                "r2": r2,
            }
        )

    # === Save after each batch ===
    pd.DataFrame(all_results).to_csv("outputs/eval_top_50_in_batches.csv", index=False)
    print(f"Batch {batch_id} saved")

print("\nAll batches done!")

# === Save final results ===
results_df = pd.read_csv("outputs/eval_top_50_in_batches.csv")
sorted_df = results_df.sort_values(by=["mse", "r2"], ascending=[True, False])
sorted_df.to_csv("outputs/sorted_gene_combos.csv", index=False)
print("Saved sorted gene combos to outputs/sorted_gene_combos.csv")
