# Predicting Anti-Cancer Drug Effectiveness from Molecular Data

This project focuses on predicting the effectiveness of anti-cancer drugs using genomic and molecular-level data. It combines powerful data science tools, feature engineering, and machine learning to support better drug response prediction.

## Project Structure

├── data/ # Raw and processed data (excluded from Git)
├── final_features/ # Final features used for modeling
├── outputs/ # Model results, logs, plots
├── src/ # Source code
│ ├── dataloader.py
│ ├── features/
│ ├── modeling/
│ └── visualization/
├── .gitignore
├── requirements.txt
└── README.md

## Technologies Used

- Python 3.12+
- Dask
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter (optional for experiments)

## Workflow Summary

1. Load & clean data – Integrate and filter molecular, gene, and drug metadata.
2. Feature engineering – Build sparse mutation matrices and gene-target profiles.
3. Modeling – Use ML models like Random Forest to predict drug response.
4. Evaluation – Analyze results via MSE and feature importance metrics.

## Key Goals

- Predict drug effectiveness (e.g., IC50) for cell lines.
- Identify top-performing gene combinations.
- Optimize model performance across various data chunks.

## Disclaimer

Some datasets are excluded from this repository due to size (e.g., `.parquet`, `.csv`) or privacy concerns. See `.gitignore` for details.

## To Do

- [ ] Add model comparison
- [ ] Integrate GitHub Actions (CI)
- [ ] Improve data documentation

---

Want to contribute? Ideas or questions? Feel free to open an issue or get in touch.
