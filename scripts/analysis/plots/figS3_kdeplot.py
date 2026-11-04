#!/usr/bin/env python3
"""
Score distributions of functional and neutral variants with the decision threshold
of each method (Figure S3), from the split sheets of predictions_report.xlsx.

Usage: python kdeplots_thresholds.py [--infile ...] [--outdir ...]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

INFILE = Path("../../results/predictions/predictions_report.xlsx")  # change this to your path
OUTDIR = Path("../../results/figures")  # change this to your path

SPLIT_SHEETS = [
    "split_1",
    # "split_2",
    # "split_3",
    # "split_4",
    # "split_5",
]

OUTPUT_NAME = "kdeplots_split_1.png"

FIGSIZE = (24, 5.5)
DPI = 300
FSIZE = 18

COLOR_FUNCTIONAL = "#1f77b4"
COLOR_NEUTRAL = "#ff7f0e"
COLOR_THRESHOLD = "black"

PANEL_LABELS = ["A", "B", "C", "D"]

# True: each class density has area equal to its proportion (same scale as the
# weighted GMM components). False: each class density has area 1.
COMMON_NORM = True

# Same GMM parameters as in collect_predictions.py
GMM_PARAMS = dict(
    n_components=2,
    random_state=0,
)

# (score column, x-axis label, clip range for KDE, threshold)
# threshold: float = fixed threshold; "gmm" = two-component GMM fitted on the scores
METHODS = [
    (
        "mutsite_y_proba",
        "Mut-Site model predicted probability\n(PLM-Emb)",
        (0.0, 1.0),
        0.5,
    ),
    (
        "esm1v_score",
        "ESM-1v log-likelihood ratio\n(PLM-ZS)",
        None,
        "gmm",
    ),
    (
        "sift_score",
        "SIFT score\n(classic conservation-based)",
        (0.0, 1.0),
        0.05,
    ),
    (
        "provean_score",
        "PROVEAN score\n(classic conservation-based)",
        None,
        -2.5,
    ),
]


def load_data(infile):
    """Read and combine the selected split sheets (test predictions)."""
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
        split_df = pd.read_excel(infile, sheet_name=sheet)
        split_df.columns = split_df.columns.str.strip()
        split_df["source_split"] = sheet
        dataframes.append(split_df)

    return pd.concat(dataframes, ignore_index=True)


def get_effect_labels(df):
    """Text labels (functional / neutral) from y_true or Effect."""
    if "y_true" in df.columns:
        y_true = pd.to_numeric(df["y_true"], errors="coerce")
        return y_true.map({0: "neutral", 1: "functional"})

    if "Effect" in df.columns:
        effects = df["Effect"].astype(str).str.strip().str.lower()
        return effects.replace({"deleterious": "functional"})

    raise ValueError("Neither 'y_true' nor 'Effect' was found.")


def fit_gmm(scores):
    """
    Fit a two-component GMM. Returns the model, the functional component
    (lowest mean) and the scores where its posterior probability crosses 0.5
    (more than one boundary is possible with unequal variances).
    """
    X = np.asarray(scores, dtype=float).reshape(-1, 1)
    gmm = GaussianMixture(**GMM_PARAMS).fit(X)

    means = gmm.means_.ravel()
    func_k = int(means.argmin())

    grid = np.linspace(X.min(), X.max(), 20000).reshape(-1, 1)
    post = gmm.predict_proba(grid)[:, func_k]
    crossings = np.where(np.diff(np.sign(post - 0.5)))[0]
    boundaries = grid[crossings].ravel()

    return gmm, func_k, boundaries


def plot_gmm_components(ax, gmm, func_k, xmin, xmax):
    """Weighted GMM components as dashed curves."""
    xs = np.linspace(xmin, xmax, 1000)
    means = gmm.means_.ravel()
    sds = np.sqrt(gmm.covariances_.ravel())
    weights = gmm.weights_

    for k in range(len(means)):
        color = COLOR_FUNCTIONAL if k == func_k else COLOR_NEUTRAL
        name = "GMM functional" if k == func_k else "GMM neutral"

        ax.plot(
            xs,
            weights[k] * norm.pdf(xs, means[k], sds[k]),
            linestyle="--",
            linewidth=2.5,
            color=color,
            label=name,
        )


def plot_density_panel(ax, data, x_label, clip, threshold, show_ylabel):
    """Class densities and decision threshold(s) of one method."""
    kde_kwargs = dict(
        data=data,
        x="score",
        bw_adjust=1.2,
        hue="effect",
        hue_order=["functional", "neutral"],
        palette={
            "functional": COLOR_FUNCTIONAL,
            "neutral": COLOR_NEUTRAL,
        },
        fill=True,
        alpha=0.4,
        linewidth=0,
        common_norm=COMMON_NORM,
        ax=ax,
    )

    # Bounded scores: keep the density inside the valid range
    if clip is not None:
        kde_kwargs["clip"] = clip
        kde_kwargs["cut"] = 0

    sns.kdeplot(**kde_kwargs)

    # Legend handles for the classes (replaces the seaborn legend)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_FUNCTIONAL, alpha=0.4),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_NEUTRAL, alpha=0.4),
    ]
    labels = ["functional", "neutral"]

    if threshold == "gmm":
        gmm, func_k, boundaries = fit_gmm(data["score"])

        plot_gmm_components(
            ax,
            gmm,
            func_k,
            data["score"].min(),
            data["score"].max(),
        )

        print(f"GMM means: {gmm.means_.ravel()}")
        print(f"GMM SDs: {np.sqrt(gmm.covariances_.ravel())}")
        print(f"GMM weights: {gmm.weights_}")
        print(f"GMM decision boundaries: {boundaries}")

        if len(boundaries) > 1:
            print(
                "WARNING: more than one GMM decision boundary. "
                "Some high scores may be assigned to the functional component."
            )

        # for b in boundaries:
        #     ax.axvline(
        #         b,
        #         color=COLOR_THRESHOLD,
        #         linestyle=":",
        #         linewidth=2,
        #     )

        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles += ax_handles
        labels += ax_labels

        # handles.append(
        #     plt.Line2D([0], [0], color=COLOR_THRESHOLD, linestyle=":", linewidth=2)
        # )
        # labels.append(
        #     "GMM boundary = " + ", ".join(f"{b:.2f}" for b in boundaries)
        # )

    else:
        ax.axvline(
            threshold,
            color=COLOR_THRESHOLD,
            linestyle=":",
            linewidth=2,
        )

        handles.append(
            plt.Line2D([0], [0], color=COLOR_THRESHOLD, linestyle=":", linewidth=2)
        )
        labels.append(f"Threshold = {threshold}")

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.legend(
        handles,
        labels,
        fontsize=FSIZE - 5,
        loc="upper left",
        frameon=False,
    )

    ax.set_xlabel(x_label, fontsize=FSIZE - 2)
    ax.set_ylabel("Density" if show_ylabel else "", fontsize=FSIZE - 2)
    ax.tick_params(axis="both", labelsize=FSIZE - 4)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.15)


def plot_density_thresholds(
    infile=INFILE,
    outdir=OUTDIR,
    save=True,
):
    """One panel per method with the class score densities and its threshold."""
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(infile)
    df["effect"] = get_effect_labels(df)

    # Same variants in all panels
    score_columns = [m[0] for m in METHODS if m[0] in df.columns]
    missing = [m[0] for m in METHODS if m[0] not in df.columns]

    if missing:
        print(f"Missing columns (panels skipped): {missing}")

    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["effect", *score_columns])
    print(f"Variants plotted (all panels): {len(df)}")
    print(df["effect"].value_counts().to_string())

    methods = [m for m in METHODS if m[0] in df.columns]

    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=FIGSIZE,
    )
    axes = np.atleast_1d(axes)

    for i, (ax, (col, x_label, clip, threshold)) in enumerate(zip(axes, methods)):
        data = pd.DataFrame(
            {
                "score": df[col].astype(float),
                "effect": df["effect"],
            }
        )

        plot_density_panel(
            ax,
            data,
            x_label,
            clip,
            threshold,
            show_ylabel=(i == 0),
        )

        ax.text(
            -0.12,
            1.05,
            PANEL_LABELS[i],
            transform=ax.transAxes,
            fontsize=FSIZE + 4,
            fontweight="bold",
            va="top",
            ha="right",
        )

    fig.tight_layout()

    if save:
        outfile = outdir / OUTPUT_NAME
        fig.savefig(outfile, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outfile}")

    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot class-specific score densities and decision thresholds "
            "from predictions_report.xlsx."
        )
    )
    parser.add_argument("--infile", default=str(INFILE), help="Path to predictions_report.xlsx.")
    parser.add_argument("--outdir", default=str(OUTDIR), help="Directory where the PNG will be saved.")
    args = parser.parse_args()

    plot_density_thresholds(
        infile=Path(args.infile),
        outdir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()