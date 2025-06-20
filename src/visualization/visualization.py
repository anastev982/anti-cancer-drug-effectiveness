#!/usr/bin/env python3
"""
Visualization module for the GDSC project.

- Loads final dataset
- Plots correlation heatmaps
- Plots concentration vs. intensity scatterplots
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Configuration

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset loading


def load_final_dataset(
    path: Path | str = "data/processed/final_merged.parquet",
) -> pd.DataFrame:
    """
    Load the final merged dataset as a pandas dataframe.
    """
    return pd.read_parquet(path)


# Plotting functions


def plot_correlation_heatmap(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (12, 10),
    title: str = "Correlation Matrix",
) -> None:
    """
    Plots a heatmap of the correlation matrix for numeric columns.
    """
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_conc_vs_intensity(
    df: pd.DataFrame,
    save_path: Path | str = PLOTS_DIR / "conc_vs_intensity.png",
    show_plot: bool = True,
) -> None:
    """
    Plots concentration (CONC) vs. intensity (INTENSITY).
    Optionally saves or shows the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="conc",
        y="intensity",
        hue="cell_line_name",
        alpha=0.6,
    )
    plt.xlabel("Concentration")
    plt.ylabel("Intensity")
    plt.title("Concentration vs Intensity")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    df = load_final_dataset()
    plot_correlation_heatmap(df)
    plot_conc_vs_intensity(df)
