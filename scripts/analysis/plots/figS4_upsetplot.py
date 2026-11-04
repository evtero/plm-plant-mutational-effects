#!/usr/bin/env python3
"""
UpSet plot of correct classifications from the split_1 sheet of predictions_report.xlsx.

Each set contains the variants correctly classified by one method. Intersections show
how many variants were correctly classified by a given combination of methods.

Example:
python upset_correct_predictions_split1.py \
    --infile predictions_report.xlsx \
    --output upset_correct_predictions_split1.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import UpSet, from_indicators

# Settings
SHEET_NAME = "split_1"
TRUE_COL = "y_true"

# Methods and their order in the UpSet plot (comment out entries to exclude methods)
PREDICTION_COLS = {
    "sift_label": "SIFT",
    "provean_label": "PROVEAN",
    "esm1v_label": "PLM-ZS",
    # "fullmean_y_pred": "Full mean",
    # "fullmean_delta_y_pred": "Full mean + delta",
    # "fullmean_delta_sift_y_pred": "Full mean + delta + SIFT",
    # "fullmean_sift_y_pred": "Full mean + SIFT",
    # "localmean_y_pred": "Local mean",
    # "localmean_delta_y_pred": "Local mean + delta",
    # "localmean_delta_sift_y_pred": "Local mean + delta + SIFT",
    # "localmean_sift_y_pred": "Local mean + SIFT",
    "mutsite_y_pred": "PLM-Emb",
    # "mutsite_delta_y_pred": "Mut-site emb + delta",
    # "mutsite_delta_sift_y_pred": "Mut-site emb + delta + SIFT",
    "mutsite_sift_y_pred": "PLM-EmbSIFT",
}

# Text labels to binary classes (1 = functional/deleterious, 0 = neutral/tolerated)
TEXT_TO_BINARY = {
    "functional": 1,
    "deleterious": 1,
    "damaging": 1,
    "pathogenic": 1,
    "neutral": 0,
    "tolerated": 0,
    "benign": 0,
}

FIGSIZE = (32, 9)
DPI = 600
BACKGROUND_COLOR = "white"
INTERSECTION_COLOR = "#303030"
TOTALS_COLOR = "#202020"
GRID_COLOR = "#D9D9D9"
FONT_SIZE = 10
TITLE = ""
OUTPUT_FILENAME = "upset_correct_predictions_split1.png"

# UpSet options
SORT_CATEGORIES_BY = None       # keep the order of PREDICTION_COLS
SORT_BY = "cardinality"         # sort intersections by size
SHOW_COUNTS = True
SHOW_PERCENTAGES = False
ELEMENT_SIZE = 29
MIN_SUBSET_SIZE = 1             # increase to hide small intersections
MAX_INTERSECTIONS = None        # e.g. 40 to show only the 40 largest


def normalize_binary(series: pd.Series, column_name: str) -> pd.Series:
    """Convert numeric or text predictions to nullable 0/1."""
    numeric = pd.to_numeric(series, errors="coerce")

    text = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(TEXT_TO_BINARY)
    )

    normalized = numeric.fillna(text).astype("Int64")

    invalid = normalized.notna() & ~normalized.isin([0, 1])
    if invalid.any():
        values = sorted(series.loc[invalid].astype(str).unique())
        raise ValueError(
            f"Column '{column_name}' contains non-binary values: {values}"
        )

    return normalized


def load_predictions(
    infile: Path,
    sheet_name: str = SHEET_NAME,
) -> pd.DataFrame:
    """Read only the needed columns of the selected sheet."""
    required_cols = [TRUE_COL, *PREDICTION_COLS.keys()]

    try:
        df = pd.read_excel(
            infile,
            sheet_name=sheet_name,
            usecols=lambda col: col in required_cols,
            engine="openpyxl",
        )
    except ValueError as exc:
        raise ValueError(
            f"Could not read sheet '{sheet_name}' from {infile}."
        ) from exc

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            "Missing columns in sheet "
            f"'{sheet_name}': {missing}\n"
            f"Columns read: {list(df.columns)}"
        )

    return df


def build_correctness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per method: True when the prediction is correct."""
    y_true = normalize_binary(df[TRUE_COL], TRUE_COL)
    indicators = {}

    for pred_col, display_name in PREDICTION_COLS.items():
        y_pred = normalize_binary(df[pred_col], pred_col)

        valid = y_true.notna() & y_pred.notna()
        correct = pd.Series(False, index=df.index, dtype=bool)
        correct.loc[valid] = (
            y_pred.loc[valid].astype(int).to_numpy()
            == y_true.loc[valid].astype(int).to_numpy()
        )

        indicators[display_name] = correct

        n_valid = int(valid.sum())
        n_correct = int(correct.sum())
        accuracy = n_correct / n_valid if n_valid else float("nan")
        print(
            f"{display_name:30s} | valid: {n_valid:7d} | "
            f"correct: {n_correct:7d} | accuracy: {accuracy:.4f}"
        )

    indicators_df = pd.DataFrame(indicators, index=df.index)

    if not indicators_df.any(axis=None):
        raise ValueError(
            "No method has correct classifications. "
            "Check the encoding of y_true and the predictions."
        )

    return indicators_df


def create_upset(
    indicators_df: pd.DataFrame,
    title: str = TITLE,
):
    """Build and style the UpSet plot."""
    upset_data = from_indicators(indicators_df)

    upset_kwargs = dict(
        subset_size="count",
        show_counts=SHOW_COUNTS,
        show_percentages=SHOW_PERCENTAGES,
        sort_categories_by=SORT_CATEGORIES_BY,
        sort_by=SORT_BY,
        element_size=ELEMENT_SIZE,
        facecolor=INTERSECTION_COLOR,
        min_subset_size=MIN_SUBSET_SIZE,
    )

    if MAX_INTERSECTIONS is not None:
        upset_kwargs["max_subset_rank"] = MAX_INTERSECTIONS

    upset = UpSet(upset_data, **upset_kwargs)

    fig = plt.figure(figsize=FIGSIZE, facecolor=BACKGROUND_COLOR)
    axes = upset.plot(fig=fig)

    # Total bars per method
    if "totals" in axes:
        for patch in axes["totals"].patches:
            patch.set_facecolor(TOTALS_COLOR)

    # Intersection bars
    if "intersections" in axes:
        for patch in axes["intersections"].patches:
            patch.set_facecolor(INTERSECTION_COLOR)

    # Common style
    for axis in axes.values():
        axis.set_facecolor(BACKGROUND_COLOR)
        axis.tick_params(labelsize=FONT_SIZE)
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)

    fig.suptitle(title, fontsize=FONT_SIZE + 4, fontweight="bold", y=1.02)
    return fig, axes


def plot_upset(
    infile: Path,
    output: Path,
    sheet_name: str = SHEET_NAME,
) -> None:
    df = load_predictions(infile, sheet_name=sheet_name)
    indicators_df = build_correctness_indicators(df)
    fig, _ = create_upset(indicators_df)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight",
        facecolor=BACKGROUND_COLOR,
    )
    plt.close(fig)
    print(f"\nSaved: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "UpSet plot of correct predictions for the split_1 sheet "
            "of predictions_report.xlsx."
        )
    )
    parser.add_argument(
        "--infile",
        type=Path,
        default=Path("predictions_report.xlsx"),
        help="Path to predictions_report.xlsx.",
    )
    parser.add_argument(
        "--sheet",
        default=SHEET_NAME,
        help=f"Excel sheet. Default: {SHEET_NAME}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(OUTPUT_FILENAME),
        help=f"Output PNG. Default: {OUTPUT_FILENAME}",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    plot_upset(
        infile=args.infile,
        output=args.output,
        sheet_name=args.sheet,
    )

if __name__ == "__main__":
    main()