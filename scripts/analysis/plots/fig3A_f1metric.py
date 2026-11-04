#!/usr/bin/env python3
# figure2A
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INFILE = Path("../../../predictions_report.xlsx")
SHEET_NAME = "metrics_mean_std"
OUTDIR = Path("../../github/plots_test")

FIGSIZE = (26, 5)
DPI = 300
FSIZE = 22

BAR_WIDTH = 0.36
ERROR_CAPSIZE = 4
VALUE_DECIMALS = 3

# Metric argument: mean column, SD column, y-axis label
METRICS = {
    "accuracy": ("accuracy_mean", "accuracy_std", "Accuracy"),
    "f1": ("f1_mean", "f1_std", "F1 score"),
    "precision": ("precision_mean", "precision_std", "Precision"),
    "recall": ("recall_mean", "recall_std", "Recall"),
    "auroc": ("auroc_mean", "auroc_std", "AUROC"),
}

EMB_PAIRS = [
    ("fullmean", "fullmean_sift", "Full-Mean"),
    ("localmean", "localmean_sift", "Local-Mean"),
    ("mutsite", "mutsite_sift", "Mut-Site"),

    # Uncomment these lines to include delta embeddings
    ("fullmean_delta", "fullmean_delta_sift", "Delta \nFull-Mean"),
    ("localmean_delta", "localmean_delta_sift", "Delta \nLocal-Mean"),
    ("mutsite_delta", "mutsite_delta_sift", "Delta \nMut-Site"),
]

COLORS = {
    "emb": "deepskyblue",
    "emb_sift": "tomato",
    "esm1v": "turquoise",
    "sift": "orange",
    "provean": "green",
}

def load_data(infile):
    """
    Read the metrics_mean_std sheet from the Excel report.
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


def get_value(df, method, column):
    """
    Return one value for a method and column.

    NaN is returned when the method is not present.
    """
    row = df[df["method"] == method]

    if row.empty:
        return np.nan

    value = row.iloc[0][column]

    if pd.isna(value):
        return np.nan

    return float(value)


def method_exists(df, method):
    """
    Check whether a method exists in the input data.
    """
    return method in df["method"].values


def clean_filename(text):
    """
    Create a safe filename from a text string.
    """
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

def plot_metric(
    metric,
    infile=INFILE,
    outdir=OUTDIR,
    ylabel=None,
    title=None,
    xlabel_fontsize=FSIZE,
    ylabel_fontsize=FSIZE,
    xtick_fontsize=FSIZE - 3,
    ytick_fontsize=FSIZE - 1,
    title_fontsize=FSIZE - 1,
    value_fontsize=FSIZE - 1,
    section_fontsize=FSIZE + 1,
    classic_section_fontsize=FSIZE - 1,
    legend_fontsize=FSIZE - 3,
    ax=None,
    save=True,
):
    """
    Create a grouped bar plot comparing embedding-based methods,
    embedding + SIFT methods, ESM-1v and classic predictors.
    """

    if metric not in METRICS:
        raise ValueError(
            f"Unknown metric: {metric}\n"
            f"Available metrics: {list(METRICS)}"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mean_column, std_column, default_ylabel = METRICS[metric]

    df = load_data(infile)

    # Check that the selected metric columns exist
    missing_columns = [
        column
        for column in [mean_column, std_column]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in sheet '{SHEET_NAME}': "
            f"{missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Read embedding method values
    # --------------------------------------------------------

    labels = []

    without_sift_mean = []
    without_sift_std = []

    with_sift_mean = []
    with_sift_std = []

    for no_sift, with_sift, label in EMB_PAIRS:

        labels.append(label)

        without_sift_mean.append(
            get_value(df, no_sift, mean_column)
        )

        without_sift_std.append(
            get_value(df, no_sift, std_column)
        )

        with_sift_mean.append(
            get_value(df, with_sift, mean_column)
        )

        with_sift_std.append(
            get_value(df, with_sift, std_column)
        )

    x = np.arange(len(labels))

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    else:
        fig = ax.figure

    # --------------------------------------------------------
    # Grouped embedding bars
    # --------------------------------------------------------

    bars_without_sift = ax.bar(
        x - BAR_WIDTH / 2,
        without_sift_mean,
        BAR_WIDTH,
        yerr=without_sift_std,
        capsize=ERROR_CAPSIZE,
        label="ESM-1v embeddings",
        color=COLORS["emb"],
    )

    bars_with_sift = ax.bar(
        x + BAR_WIDTH / 2,
        with_sift_mean,
        BAR_WIDTH,
        yerr=with_sift_std,
        capsize=ERROR_CAPSIZE,
        label="ESM-1v embeddings + SIFT",
        color=COLORS["emb_sift"],
    )

    # --------------------------------------------------------
    # Additional individual predictors
    # --------------------------------------------------------

    extra_methods = [
        ("esm1v", "ESM-1v\nzero-shot", COLORS["esm1v"]),
        ("sift", "SIFT", COLORS["sift"]),
        ("provean", "PROVEAN", COLORS["provean"]),
    ]

    # PROVEAN or another predictor is skipped when absent
    extra_methods = [
        item
        for item in extra_methods
        if method_exists(df, item[0])
    ]

    # Positions preserve a visual gap between sections
    first_extra_x = len(labels) + 0.01

    extra_x = [
        first_extra_x + index * 0.70
        for index in range(len(extra_methods))
    ]

    extra_means = [
        get_value(df, method, mean_column)
        for method, _, _ in extra_methods
    ]

    extra_stds = [
        get_value(df, method, std_column)
        for method, _, _ in extra_methods
    ]

    extra_labels = [
        label
        for _, label, _ in extra_methods
    ]

    extra_colors = [
        color
        for _, _, color in extra_methods
    ]

    extra_bars = ax.bar(
        extra_x,
        extra_means,
        width=0.40,
        yerr=extra_stds,
        capsize=ERROR_CAPSIZE,
        color=extra_colors,
    )

    # --------------------------------------------------------
    # Add mean labels above error bars
    # --------------------------------------------------------

    all_bars = (
        list(bars_without_sift)
        + list(bars_with_sift)
        + list(extra_bars)
    )

    all_means = (
        without_sift_mean
        + with_sift_mean
        + extra_means
    )

    all_stds = (
        without_sift_std
        + with_sift_std
        + extra_stds
    )

    for bar, mean, std in zip(
        all_bars,
        all_means,
        all_stds,
    ):
        if np.isnan(mean):
            continue

        label_height = mean

        if not np.isnan(std):
            label_height += std

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_height + 0.003,
            f"{mean:.{VALUE_DECIMALS}f}",
            ha="center",
            va="bottom",
            fontsize=value_fontsize,
        )

    # --------------------------------------------------------
    # X-axis labels
    # --------------------------------------------------------

    xticks = list(x) + extra_x
    xticklabels = labels + extra_labels

    ax.set_xticks(xticks)

    ax.set_xticklabels(
        xticklabels,
        rotation=0,
        fontsize=xtick_fontsize,
    )

    ax.tick_params(
        axis="y",
        labelsize=ytick_fontsize,
    )

    # --------------------------------------------------------
    # Y-axis limits
    # --------------------------------------------------------

    lower_values = []
    upper_values = []

    for mean, std in zip(all_means, all_stds):

        if np.isnan(mean):
            continue

        valid_std = 0.0 if np.isnan(std) else std

        lower_values.append(mean - valid_std)
        upper_values.append(mean + valid_std)

    if not upper_values:
        raise ValueError(
            "No valid values were found for the selected metric."
        )

    ymin = max(
        0.0,
        min(lower_values) - 0.03,
    )

    ymax = min(
        1.0,
        max(upper_values) + 0.08,
    )

    ax.set_ylim(ymin, ymax)

    # --------------------------------------------------------
    # Section titles
    # --------------------------------------------------------

    section_y = ymax - 0.01

    # Method keys are used instead of display labels so that
    # renaming a tick label does not break the section layout
    extra_keys = [
        method
        for method, _, _ in extra_methods
    ]

    ax.text(
        np.mean(x)+0.9,
        section_y,
        "PLM-Emb and PLM-EmbSIFT",
        ha="center",
        va="top",
        fontsize=section_fontsize,
    )

    esm_position = None

    if "esm1v" in extra_keys:

        esm_position = extra_x[extra_keys.index("esm1v")]

        ax.text(
            esm_position,
            section_y,
            "PLM-ZS",
            ha="center",
            va="top",
            fontsize=section_fontsize,
        )

    classic_positions = [
        position
        for position, key in zip(extra_x, extra_keys)
        if key in {"sift", "provean"}
    ]

    if classic_positions:

        ax.text(
            np.mean(classic_positions),
            section_y,
            "Classic\nconservation\n-based",
            ha="center",
            va="top",
            fontsize=classic_section_fontsize,
        )

    # --------------------------------------------------------
    # Vertical section separators
    # --------------------------------------------------------

    if extra_x:

        ax.axvline(
            x=(x[-1] + extra_x[0]) / 2,
            color="black",
            linestyle=":",
            linewidth=1,
        )

    if (
        esm_position is not None
        and len(classic_positions) > 0
    ):
        ax.axvline(
            x=(esm_position + classic_positions[0]) / 2,
            color="black",
            linestyle=":",
            linewidth=1,
        )

    # --------------------------------------------------------
    # Axis labels, title and legend
    # --------------------------------------------------------

    ax.set_ylabel(
        ylabel if ylabel else default_ylabel,
        fontsize=ylabel_fontsize,
        labelpad=5,
    )

    if title:
        ax.set_title(
            title,
            fontsize=title_fontsize,
        )

    legend = ax.legend(
        fontsize=legend_fontsize,
        loc="upper left",
    )

    legend._legend_box.align = "left"

    fig.tight_layout()

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    if save:

        outfile = outdir / f"barplot_emb_{clean_filename(metric)}.png"

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
            "Plot mean ± SD metrics from predictions_report.xlsx."
        )
    )

    parser.add_argument(
        "metric",
        choices=METRICS.keys(),
        help=(
            "Metric to plot: accuracy, f1, precision, "
            "recall or auroc."
        ),
    )

    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )

    parser.add_argument(
        "--infile",
        default=str(INFILE),
        help="Input predictions_report.xlsx file.",
    )

    parser.add_argument(
        "--outdir",
        default=str(OUTDIR),
        help="Output directory.",
    )

    parser.add_argument(
        "--ylabel",
        default=None,
        help="Optional y-axis label.",
    )

    args = parser.parse_args()

    plot_metric(
        metric=args.metric,
        infile=Path(args.infile),
        outdir=Path(args.outdir),
        ylabel=args.ylabel,
        title=args.title,
    )


if __name__ == "__main__":
    main()