Predicting Anti-Cancer Drug Effectiveness from Molecular Data

This project builds a machine-learning pipeline for predicting anti-cancer drug response (LN_IC50) using genomic mutation data.
The goal is to understand which genes — individually and in combination — provide the strongest predictive signal for drug sensitivity.

The project includes:

Large-scale preprocessing of molecular and mutation datasets

Construction of gene-level feature matrices (20k+ genomic features)

Evaluation of thousands of gene combinations using Random Forest models

Identification of top predictive genes and gene groups

Chunk-level analysis for validation stability

Fully reproducible ML pipeline

Project Structure

── data/ # Raw & intermediate data (ignored by Git)
── final_features/ # Final Parquet feature sets used for modeling
── final_chunks/ # Small Parquet chunks for rapid evaluation
── outputs/ # Plots, metrics, ranked genes, MSE scores

── src/
── preprocessing/ # Building informative subsets & chunking
── modeling/ # ML models, training scripts, evaluation
── analysis/ # Gene ranking, combo evaluation, stats
── utils/ # Helper functions (optional)

── requirements.txt
── README.md

Project Motivation

Drug response prediction is a core challenge in precision oncology.
By modeling molecular features (gene mutations), we aim to:

predict IC50/LN_IC50 drug sensitivity

identify high-impact genes

discover robust multi-gene combinations

support future feature selection and biomarker discovery

Pipeline Overview

1. Preprocessing & Feature Engineering

Load multiple genomic datasets (.parquet, .csv)

Merge mutation matrices, gene metadata, drug metadata

Build:

full_model_input.parquet (20k+ genes)

subset_genes_only.parquet (gene + target)

subset_informative.parquet (filtered genes with variance > 1)

Chunk large datasets into final_chunks/ for faster experiments

2. Gene-Level Evaluation

Evaluate each gene individually:

Train Random Forest on each gene separately

Compute MSE per gene

Identify top N most predictive genes

Save results & plots

3. Multi-Gene Combination Search

For selected top genes:

Generate combinations (2–10 genes)

Train RF model for each combination

Rank combinations by MSE

Save:

best combos per chunk

global ranked combos

plots showing MSE vs number of genes

This mimics feature-selection strategies used in real computational oncology research.

4. Chunk-Level Robustness

To avoid bias:

Split filtered dataset into 50-row chunks

Evaluate combinations on each chunk

Count how often each gene appears in top combinations

Extract robust genes across data splits

5. Final Model

Train on the full dataset using optimal gene combinations.

Supported metrics:

MSE

R²

Feature importance

Key Results (Short Summary)

(You can update these once you upload your plots and outputs.)

Top single genes by predictive power:
mstn, bambi, tmt1a, fcrl6, osbp2

Best gene combinations reached MSE ≈ X.XX

Multi-gene models outperform single-gene models in all experiments

Chunk-level consistency confirms biological stability of selected genes

Technologies Used

Languages & Libraries

Python 3.12

Pandas, NumPy

Dask (large-scale processing)

Scikit-learn

Matplotlib / Seaborn

PyArrow / Parquet

How to Run
Install requirements:
pip install -r requirements.txt

Run key scripts:
Build informative subset:
python -m src.preprocessing.build_subset_informative

Chunk dataset:
python -m src.preprocessing.chunk_subset_genes

Evaluate gene combinations:
python -m src.modeling.evaluate_combos_over_chunks

Train selected genes:
python -m src.modeling.train_selected_genes --genes mstn bambi tmt1a

Analyze gene frequency across combinations:
python -m src.analysis.top_genes_across_chunks

Data Disclaimer

Large .parquet files are not included due to GitHub size limits.
Outputs, plots, and intermediate results are saved inside the repo structure but heavy datasets are excluded via .gitignore.

Future Improvements

Add Gradient Boosting / XGBoost comparison

Implement SHAP value interpretability

Add experiment tracking (MLflow)

Add GitHub Actions for automated testing

Contact

If you have questions or ideas, feel free to open an Issue or reach out.
This project was created as part of a broader effort in molecular data analysis and precision oncology research.
