#!/usr/bin/env python3
# figure2B
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

INFILE = Path("../../../predictions_report.xlsx")
SHEET_NAME = "metrics_mean_std"
OUTDIR = Path("../../../plots_test")

FIGSIZE = (6, 6)
DPI = 300
FSIZE = 18

BAR_WIDTH = 0.22
ERROR_CAPSIZE = 4
VALUE_DECIMALS = 3
VALUE_OFFSET = 0.008


# ============================================================
# METHODS AND COLORS
# ============================================================

METHODS = [
    ("esm1v", "PLM-ZS"),
    ("sift", "SIFT"),
    ("provean", "PROVEAN"),
]

COLORS = {
    "esm1v": "turquoise",
    "sift": "orange",
    "provean": "green",
}


# ============================================================
# DATA FUNCTIONS
# ============================================================

def load_data(infile):
    """
    Read the mean and SD metrics from the Excel report.
    """
    df = pd.read_excel(
        infile,
        sheet_name=SHEET_NAME,
    )

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    if "method" not in df.columns:
        raise ValueError(
            f"Column 'method' was not found in sheet '{SHEET_NAME}'."
        )

    # Normalize method names
    df["method"] = (
        df["method"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


def method_exists(df, method):
    """
    Check whether a method is present in the input table.
    """
    return method in df["method"].values


def get_value(df, method, column):
    """
    Return one value for a method and column.

    NaN is returned when the method or value is missing.
    """
    row = df[df["method"] == method]

    if row.empty:
        return np.nan

    value = row.iloc[0][column]

    if pd.isna(value):
        return np.nan

    return float(value)


# ============================================================
# PLOTTING FUNCTION
# ============================================================

def plot_recall_precision(
    infile=INFILE,
    outdir=OUTDIR,
    ylabeltitle=None,
    xlabeltitle=None,
    xlabel_fontsize=FSIZE,
    ylabel_fontsize=FSIZE,
    xtick_fontsize=FSIZE - 1,
    ytick_fontsize=FSIZE - 1,
    legend_fontsize=FSIZE - 1.5,
    value_fontsize=FSIZE - 2,
    ax=None,
    save=True,
):
    """
    Create a grouped bar plot comparing precision and recall
    for ESM-1v, SIFT and optionally PROVEAN.
    """

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data(infile)

    required_columns = [
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in sheet '{SHEET_NAME}': "
            f"{missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Keep only methods that exist in the Excel file
    available_methods = [
        (method, label)
        for method, label in METHODS
        if method_exists(df, method)
    ]

    if not available_methods:
        raise ValueError(
            "None of the requested methods were found: "
            f"{[method for method, _ in METHODS]}"
        )

    labels = ["Precision", "Recall"]
    x = np.arange(len(labels))

    number_of_methods = len(available_methods)

    # Center all method bars around each metric position
    offsets = (
        np.arange(number_of_methods)
        - (number_of_methods - 1) / 2
    ) * BAR_WIDTH

    if ax is None:
        fig, ax = plt.subplots(
            figsize=FIGSIZE,
        )
    else:
        fig = ax.figure

    all_means = []
    all_stds = []

    for offset, (method, method_label) in zip(
        offsets,
        available_methods,
    ):

        means = [
            get_value(df, method, "precision_mean"),
            get_value(df, method, "recall_mean"),
        ]

        standard_deviations = [
            get_value(df, method, "precision_std"),
            get_value(df, method, "recall_std"),
        ]

        bars = ax.bar(
            x + offset,
            means,
            BAR_WIDTH,
            yerr=standard_deviations,
            capsize=ERROR_CAPSIZE,
            label=method_label,
            color=COLORS.get(method),
        )

        all_means.extend(means)
        all_stds.extend(standard_deviations)

        # Add mean values above each error bar
        for bar, mean, std in zip(
            bars,
            means,
            standard_deviations,
        ):

            if np.isnan(mean):
                continue

            label_height = mean

            if not np.isnan(std):
                label_height += std

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_height + VALUE_OFFSET,
                f"{mean:.{VALUE_DECIMALS}f}",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
            )

    # --------------------------------------------------------
    # Axis configuration
    # --------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        fontsize=xtick_fontsize,
    )

    ax.tick_params(
        axis="y",
        labelsize=ytick_fontsize,
    )

    lower_values = []
    upper_values = []

    for mean, std in zip(
        all_means,
        all_stds,
    ):

        if np.isnan(mean):
            continue

        valid_std = 0.0 if np.isnan(std) else std

        lower_values.append(mean - valid_std)
        upper_values.append(mean + valid_std)

    if not upper_values:
        raise ValueError(
            "No valid precision or recall values were found."
        )

    ymin = max(
        0.0,
        min(lower_values) - 0.03,
    )

    ymax = min(
        1.0,
        max(upper_values) + 0.08,
    )

    ax.set_ylim(
        ymin,
        ymax,
    )

    if ylabeltitle is None:
        ylabeltitle = "Score"

    ax.set_ylabel(
        ylabeltitle,
        fontsize=ylabel_fontsize,
        labelpad=5,
    )

    if xlabeltitle is not None:
        ax.set_xlabel(
            xlabeltitle,
            fontsize=xlabel_fontsize,
            labelpad=5,
        )

    ax.legend(
        fontsize=legend_fontsize,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    if save:

        outfile = outdir / "barplot_recall_precision.png"

        fig.savefig(
            outfile,
            dpi=DPI,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {outfile}")

    return fig, ax


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare precision and recall using mean ± SD "
            "from predictions_report.xlsx."
        )
    )

    parser.add_argument(
        "--infile",
        default=str(INFILE),
        help="Input predictions_report.xlsx file.",
    )

    parser.add_argument(
        "--outdir",
        default=str(OUTDIR),
        help="Directory where the PNG will be saved.",
    )

    parser.add_argument(
        "--ylabel",
        default=None,
        help="Optional y-axis label.",
    )

    parser.add_argument(
        "--xlabel",
        default=None,
        help="Optional x-axis label.",
    )

    args = parser.parse_args()

    plot_recall_precision(
        infile=Path(args.infile),
        outdir=Path(args.outdir),
        ylabeltitle=args.ylabel,
        xlabeltitle=args.xlabel,
    )


if __name__ == "__main__":
    main()