#!/usr/bin/env python3
# figure 2C
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


# USER SETTINGS
INFILE = Path("../../../predictions_report.xlsx") # can be overridden with --infile
OUTDIR = Path("../../../plots_test") # can be overridden with --outdir

SPLIT_SHEETS = [
    "split_1",
    "split_2",
    "split_3",
    "split_4",
    "split_5",
]

FIGSIZE = (7, 6)
DPI = 300
FSIZE = 17
LINE_WIDTH = 3

# Common FPR grid on which each split's ROC curve is interpolated
# before averaging
MEAN_FPR = np.linspace(0, 1, 1000)


# METHODS TO PLOT
# Tuple structure:
# (score column, display label, color, invert score)
#
# SIFT and ESM-1v are inverted because lower raw scores
# correspond to a stronger predicted deleterious effect.
METHODS = [
    ("sift_score", "SIFT", "orange", True),
    ("provean_score", "PROVEAN", "green", True),
    ("esm1v_score", "PLM-ZS", "turquoise", True),
    ("mutsite_y_proba", "PLM-Emb", "deepskyblue", False),
    (
        "mutsite_sift_y_proba",
        "PLM-EmbSIFT",
        "tomato",
        False,
    ),
]

OUTPUT_NAME = "auroc_curve.png"

def load_data(infile):
    """
    Read and combine the individual prediction sheets.

    Each split contains the test predictions for one random
    train/test split. The splits overlap, so they are NOT pooled
    for the ROC analysis: source_split is used to evaluate each
    split separately.
    """
    if not infile.exists():
        raise FileNotFoundError(
            f"Input file not found: {infile.resolve()}"
        )

    excel_file = pd.ExcelFile(infile)

    available_sheets = [
        sheet
        for sheet in SPLIT_SHEETS
        if sheet in excel_file.sheet_names
    ]

    if not available_sheets:
        raise ValueError(
            "None of the expected split sheets were found.\n"
            f"Expected: {SPLIT_SHEETS}\n"
            f"Available: {excel_file.sheet_names}"
        )

    dataframes = []

    for sheet in available_sheets:

        split_df = pd.read_excel(
            infile,
            sheet_name=sheet,
        )

        split_df.columns = split_df.columns.str.strip()

        # Keep the source split to evaluate each split separately
        split_df["source_split"] = sheet

        dataframes.append(split_df)

    df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    return df


def get_binary_labels(df):
    """
    Obtain binary true labels.

    The prediction report already contains y_true:
        0 = neutral
        1 = functional/deleterious

    Effect is used only as a fallback.
    """
    if "y_true" in df.columns:

        y_true = pd.to_numeric(
            df["y_true"],
            errors="coerce",
        )

    elif "Effect" in df.columns:

        effects = (
            df["Effect"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        y_true = effects.map(
            {
                "neutral": 0,
                "functional": 1,
                "deleterious": 1,
            }
        )

    else:
        raise ValueError(
            "Neither 'y_true' nor 'Effect' was found."
        )

    return y_true

def mean_roc(tmp, invert_score):
    """
    Compute one ROC curve per split and average them.

    Each split's TPR is interpolated on MEAN_FPR and the
    interpolated curves are averaged (vertical averaging).
    The reported AUC is the mean of the per-split AUCs.
    """
    tprs = []
    aucs = []

    for split, split_df in tmp.groupby("split"):

        # ROC requires both classes
        if split_df["y_true"].nunique() < 2:
            print(f"  {split}: only one class present, skipped.")
            continue

        scores = split_df["score"].astype(float)

        if invert_score:
            scores = -scores

        fpr, tpr, _ = roc_curve(
            split_df["y_true"].astype(int),
            scores,
        )

        interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
        interp_tpr[0] = 0.0

        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    if not aucs:
        return None

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    return {
        "mean_tpr": mean_tpr,
        # "std_tpr": np.std(tprs, axis=0),
        "mean_auc": np.mean(aucs),
        "std_auc": np.std(aucs),
        "n_splits": len(aucs),
    }

def plot_roc_curves(
    infile=INFILE,
    outdir=OUTDIR,
    xlabel_fontsize=FSIZE,
    ylabel_fontsize=FSIZE,
    xtick_fontsize=FSIZE - 3,
    ytick_fontsize=FSIZE - 1,
    legend_fontsize=FSIZE - 2,
    legend_title_fontsize=FSIZE - 1,
    ax=None,
    save=True,
):
    """
    Plot the mean ROC curve across test splits for each method.

    Legend shows the mean AUC across splits. The SD is printed
    to the console; the SD band and "± SD" in the legend are
    left commented out below.
    """
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data(infile)
    y_true = get_binary_labels(df)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=FIGSIZE,
        )
    else:
        fig = ax.figure

    plotted_methods = 0

    for score_column, label, color, invert_score in METHODS:

        if score_column not in df.columns:
            print(
                f"Skipped '{label}': "
                f"column '{score_column}' was not found."
            )
            continue

        # Keep only rows with label, score and split available
        tmp = pd.DataFrame(
            {
                "y_true": y_true,
                "score": pd.to_numeric(
                    df[score_column],
                    errors="coerce",
                ),
                "split": df["source_split"],
            }
        ).dropna()

        if tmp.empty:
            print(
                f"Skipped '{label}': no valid values found."
            )
            continue

        roc = mean_roc(tmp, invert_score)

        if roc is None:
            print(
                f"Skipped '{label}': no split with both classes."
            )
            continue

        print(
            f"{label}: mean AUC = {roc['mean_auc']:.3f} "
            f"± {roc['std_auc']:.3f} (n = {roc['n_splits']} splits)"
        )

        ax.plot(
            MEAN_FPR,
            roc["mean_tpr"],
            color=color,
            linewidth=LINE_WIDTH,
            alpha=0.9,
            label=f"{label} ({roc['mean_auc']:.3f})",
            # label=f"{label} ({roc['mean_auc']:.3f} ± {roc['std_auc']:.3f})",
        )

        # SD band around the mean ROC curve
        # (uncomment together with "std_tpr" in mean_roc)
        # ax.fill_between(
        #     MEAN_FPR,
        #     np.clip(roc["mean_tpr"] - roc["std_tpr"], 0, 1),
        #     np.clip(roc["mean_tpr"] + roc["std_tpr"], 0, 1),
        #     color=color,
        #     alpha=0.2,
        #     linewidth=0,
        # )

        plotted_methods += 1

    if plotted_methods == 0:
        raise ValueError(
            "No valid method could be plotted."
        )

    # Random-classifier reference line
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color="gray",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel(
        "False Positive Rate",
        fontsize=xlabel_fontsize,
        labelpad=5,
    )

    ax.set_ylabel(
        "True Positive Rate",
        fontsize=ylabel_fontsize,
        labelpad=5,
    )

    ax.tick_params(
        axis="x",
        labelsize=xtick_fontsize,
    )

    ax.tick_params(
        axis="y",
        labelsize=ytick_fontsize,
    )

    legend = ax.legend(
        title="Method (mean AUC)",
        # title="Method (mean AUC ± SD)",
        title_fontsize=legend_title_fontsize,
        fontsize=legend_fontsize,
        loc="lower right",
    )

    legend._legend_box.align = "left"

    fig.tight_layout()

    if save:

        outfile = outdir / OUTPUT_NAME

        fig.savefig(
            outfile,
            dpi=DPI,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {outfile}")

    return fig, ax

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot mean ROC curves across test splits from "
            "predictions_report.xlsx."
        )
    )

    parser.add_argument(
        "--infile",
        default=str(INFILE),
        help="Path to predictions_report.xlsx.",
    )

    parser.add_argument(
        "--outdir",
        default=str(OUTDIR),
        help="Directory where the PNG will be saved.",
    )

    args = parser.parse_args()

    plot_roc_curves(
        infile=Path(args.infile),
        outdir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()
