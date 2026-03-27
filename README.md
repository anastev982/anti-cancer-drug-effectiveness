# Predicting Anti-Cancer Drug Effectiveness from Molecular Data

## Overview

This project explores how molecular mutation data can be used to predict anti-cancer drug response (`LN_IC50`) through a full machine learning pipeline. Rather than starting from a clean, modeling-ready table, I worked with raw biomedical datasets that required extensive cleaning, validation, merging, and restructuring before modeling could even begin.

The project combines **data preprocessing, feature engineering, large-scale chunked processing, and machine learning evaluation** to identify which genes — individually and in combination — provide the strongest predictive signal for drug sensitivity.

This repository reflects a realistic data science workflow: not just training a model, but solving the practical challenges of turning messy, high-dimensional biological data into a usable analytical pipeline.

---

## Why this project matters

Drug response prediction is an important problem in precision oncology. The long-term goal of this type of work is to better understand which molecular signals may be associated with sensitivity or resistance to treatment.

In this project, I focused on:

- predicting drug response from mutation-based molecular features
- identifying informative genes with strong predictive value
- comparing single-gene vs multi-gene models
- evaluating whether top signals remain stable across data chunks
- building a reproducible pipeline for further feature selection and modeling

---

## What makes this project interesting

This was not a “plug in a clean dataset and train a model” type of project.

A large part of the work involved dealing with real-world problems such as:

- **raw, non-curated datasets**
- **many columns with high missingness**
- **large file sizes and memory constraints**
- **iterative chunk-size tuning to make processing feasible**
- **careful filtering of invalid or low-value columns before modeling**

One of the biggest lessons from this project was that real machine learning work is often less about choosing a model and more about making the data usable, reliable, and computationally manageable.

---

## Technical challenges solved

### 1. Data cleaning and validation

The original datasets were not analysis-ready. I spent significant time inspecting missing values, identifying unreliable columns, and deciding which features were too incomplete to keep in the pipeline.

### 2. Large-scale processing

Because the dataset was large, I could not treat preprocessing as a small in-memory task. I had to iteratively adjust chunk sizes and processing strategy to avoid memory bottlenecks and freezing during execution.

### 3. High-dimensional feature space

The project works with mutation-derived molecular features at genomic scale, which required transforming the data into structured feature matrices suitable for downstream model evaluation.

### 4. Reproducible pipeline design

Instead of keeping everything inside one notebook, I organized the workflow into reusable scripts and pipeline steps so the project could be rerun and extended more easily.

---

## Pipeline summary

### Step 1 — Data preprocessing

- Load raw molecular, mutation, and metadata files
- Clean inconsistent or incomplete columns
- Remove unusable high-missingness features
- Merge mutation, gene, and drug-related data
- Save structured intermediate outputs

### Step 2 — Feature engineering

- Build model-ready datasets from cleaned molecular information
- Create subsets for more focused experiments
- Filter informative features
- Save feature matrices as Parquet files for efficient reuse

### Step 3 — Chunk-based processing

- Split large datasets into smaller chunks
- Run experiments more efficiently on local hardware
- Reduce memory pressure during exploratory modeling
- Enable more robust comparison across subsets of the data

### Step 4 — Gene-level modeling

- Evaluate genes individually
- Train Random Forest models on selected inputs
- Rank genes by predictive performance
- Save ranked outputs and plots

### Step 5 — Multi-gene combination experiments

- Generate combinations of top-performing genes
- Compare predictive performance across combinations
- Analyze whether combining genes improves over single-gene models
- Identify stronger candidate feature groups

### Step 6 — Robustness analysis

- Compare top genes and combinations across chunks
- Measure how consistently strong predictors reappear
- Use chunk-level stability as an additional signal of robustness

---

## Tools and technologies

- **Python 3**
- **Pandas**
- **NumPy**
- **Dask**
- **Scikit-learn**
- **PyArrow / Parquet**
- **Matplotlib**

---

## Repository structure

```text
anti-cancer-drug-effectiveness/
├── archive/                  # Older or archived files
├── cleaned/                  # Cleaned outputs
├── configs/                  # Configuration files
├── data/                     # Raw and intermediate datasets
├── final_chunks/             # Chunked datasets for scalable experiments
├── final_features/           # Final feature tables used for modeling
├── notebooks/                # Exploration and notebook-based analysis
├── outputs/                  # Results, rankings, plots, and exported metrics
├── src/                      # Source code for preprocessing and modeling
├── README.md
├── requirements.txt
└── run.py
```

### Selected results

This project focuses on both predictive performance and feature discovery.

## Current findings from the workflow include:

identification of top-performing single genes for prediction
evidence that multi-gene models outperform single-gene models
chunk-level comparisons to assess which genes remain informative across splits
a reproducible foundation for future feature selection and model comparison

Note: This section can be expanded with final metrics, plots, and best-performing gene combinations once the analysis is finalized.

## What I learned

This project strengthened my practical understanding of:

real-world data cleaning
handling missing and imperfect biological data
memory-aware preprocessing on large datasets
building reproducible ML workflows
feature selection thinking in a biomedical context
balancing modeling goals with computational limitations

Most importantly, it showed me that meaningful machine learning projects are built long before the final model is trained.

## How to run

Install dependencies:

pip install -r requirements.txt

Run the main pipeline:

python src/main.py

Depending on the workflow stage, additional preprocessing or modeling scripts can be run from the src/ directory.

### Future improvements

Planned next steps include:

comparing additional model families beyond Random Forest
improving feature selection strategies
adding clearer experiment tracking and evaluation summaries
visualizing top genes and combinations more systematically
exploring more advanced biological interpretation of selected features
building a lightweight dashboard for presenting results

### About this project

I built this project to practice end-to-end data science on a realistic biomedical problem: from messy raw data to structured features and model evaluation.

What makes it meaningful to me is not only the final model, but the full process of solving data quality, scale, and reproducibility challenges along the way.

```

```
