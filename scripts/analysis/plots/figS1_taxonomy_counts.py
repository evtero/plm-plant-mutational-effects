#!/usr/bin/env python3
"""
Taxonomic distribution of the dataset proteins (Figure S1): number of proteins (unique Id)
per class, order and family. Orders and families with fewer than 5 proteins are grouped
as "Other" and also plotted separately.

Usage: edit the paths below, then run: python plot_taxonomy.py
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

total_file = Path("../../../data/total_benchmark_data_enriched.tsv")  # change this to your path
OUTDIR = Path("../../../plots_test")  # change this to your path

SOFT_RED = "#d95f5f"
FIGSIZE = (15, 3.5)  # width, height (inches): lower height = flatter plots
OUTDIR.mkdir(parents=True, exist_ok=True)

total = pd.read_csv(total_file, sep="\t")

# One row per protein (Id) with its taxonomy
merged = (
    total[["Id", "tax_class", "tax_order", "tax_family"]]
    .drop_duplicates("Id")
    .rename(columns={"tax_class": "class", "tax_order": "order", "tax_family": "family"})
)


def add_bar_labels(bars, fontsize=9):
    for bar in bars:
        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.2,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=fontsize
        )


def plot_counts(
    counts,
    xlabel,
    ylabel,
    output_file,
    colors=None,
    figsize=FIGSIZE
):

    plt.figure(figsize=figsize)

    bars = plt.bar(
        counts.index,
        counts.values,
        color=colors,
        width=0.7
    )

    add_bar_labels(bars, fontsize=9)

    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=9
    )

    plt.yticks(fontsize=9)

    plt.tight_layout()

    plt.savefig(
        OUTDIR / output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    print(f"Saved: {OUTDIR / output_file}")


# Groups below the threshold are summed into "Other"
def group_low_frequency(counts, threshold=5):
    main = counts[counts >= threshold]
    other = counts[counts < threshold]

    grouped = main.copy()

    if len(other) > 0:
        grouped["Other"] = other.sum()

    return grouped, other


# Order (>= 5 proteins + Other)
order_counts = (
    merged["order"]
    .dropna()
    .value_counts()
    .sort_values(ascending=False)
)

order_grouped, order_other = group_low_frequency(
    order_counts,
    threshold=5
)

order_colors = [
    SOFT_RED if name == "Other" else "C0"
    for name in order_grouped.index
]

plot_counts(
    order_grouped,
    xlabel="Order",
    ylabel="Counts",
    output_file="order_frequency_grouped.png",
    colors=order_colors,
)


# Order (< 5 proteins only)
if len(order_other) > 0:

    order_other = order_other.sort_values(
        ascending=False
    )

    order_other_colors = [SOFT_RED] * len(order_other)

    plot_counts(
        order_other,
        xlabel="Order",
        ylabel="Counts",
        output_file="order_frequency_other_only.png",
        colors=order_other_colors,
    )

else:
    print("No orders with frequency below 5.")


# Class
class_counts = (
    merged["class"]
    .dropna()
    .value_counts()
    .sort_values(ascending=False)
)

plot_counts(
    class_counts,
    xlabel="Class",
    ylabel="Counts",
    output_file="class_frequency.png",
)


# Family (>= 5 proteins + Other)
family_counts = (
    merged["family"]
    .dropna()
    .value_counts()
    .sort_values(ascending=False)
)

family_grouped, family_other = group_low_frequency(
    family_counts,
    threshold=5
)

family_colors = [
    SOFT_RED if name == "Other" else "C0"
    for name in family_grouped.index
]

plot_counts(
    family_grouped,
    xlabel="Family",
    ylabel="Counts",
    output_file="family_frequency_grouped.png",
    colors=family_colors,
)


# Family (< 5 proteins only)
if len(family_other) > 0:

    family_other = family_other.sort_values(
        ascending=False
    )

    family_other_colors = [SOFT_RED] * len(family_other)

    plot_counts(
        family_other,
        xlabel="Family",
        ylabel="Counts",
        output_file="family_frequency_other_only.png",
        colors=family_other_colors,
    )

else:
    print("No families with frequency below 5.")