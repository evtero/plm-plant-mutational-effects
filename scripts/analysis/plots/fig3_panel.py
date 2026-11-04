#!/usr/bin/env python3
"""
Figure 1: F1 barplot (A), precision/recall (B) and ROC curves (C)
from predictions_report.xlsx.

Usage: python figure1_panel.py [--infile ...] [--outdir ...]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from fig3A_f1metric import plot_metric
from fig3B_precision_recall import plot_recall_precision
from fig3C_auroc import plot_roc_curves

INFILE = Path("../../../predictions_report.xlsx")  # change this to your path
OUTDIR = Path("../../../plots_test")  # change this to your path
OUTPUT_NAME = "figure1_panel.png"

FIGSIZE = (22, 16)
DPI = 300

# Font sizes (FSIZE controls the whole panel)
FSIZE = 26
AXIS_LABEL_SIZE = FSIZE
XTICK_SIZE = FSIZE - 1
YTICK_SIZE = FSIZE - 1
LEGEND_SIZE = FSIZE - 3
LEGEND_TITLE_SIZE = FSIZE - 4
VALUE_SIZE = FSIZE - 1
TITLE_SIZE = FSIZE
PANEL_LABEL_SIZE = FSIZE + 5

# Panel A (F1 barplot)
BAR_XTICK_SIZE = FSIZE - 3
SECTION_SIZE = FSIZE + 1
CLASSIC_SECTION_SIZE = FSIZE - 1

# Layout: A takes the first row, B and C share the second row
ROW_HEIGHT_RATIOS = [1.0, 1.0]
ROW2_WIDTH_RATIOS = [1.0, 1.0]
HSPACE = 0.25
ROW2_WSPACE = 0.20

LEFT_MARGIN = 0.07
RIGHT_MARGIN = 0.98
TOP_MARGIN = 0.97
BOTTOM_MARGIN = 0.06

PANEL_LABEL_X = -0.08
PANEL_LABEL_Y = 1.06


# Bold panel label outside the upper-left corner
def add_panel_label(ax, label, x=PANEL_LABEL_X, y=PANEL_LABEL_Y):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="right",
    )

def create_panel(infile=INFILE, outdir=OUTDIR):
    outdir.mkdir(parents=True, exist_ok=True)

    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile.resolve()}")

    fig = plt.figure(figsize=FIGSIZE)

    main_grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=ROW_HEIGHT_RATIOS,
        hspace=HSPACE,
    )

    # First row: A over the full width
    ax_a = fig.add_subplot(main_grid[0])

    # Second row: B and C
    row2_grid = main_grid[1].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=ROW2_WIDTH_RATIOS,
        wspace=ROW2_WSPACE,
    )
    ax_b = fig.add_subplot(row2_grid[0, 0])
    ax_c = fig.add_subplot(row2_grid[0, 1])

    # A: F1 barplot
    plot_metric(
        metric="f1",
        infile=infile,
        outdir=outdir,
        ylabel="F1 score",
        title=None,
        xlabel_fontsize=AXIS_LABEL_SIZE,
        ylabel_fontsize=AXIS_LABEL_SIZE,
        xtick_fontsize=BAR_XTICK_SIZE,
        ytick_fontsize=YTICK_SIZE,
        title_fontsize=TITLE_SIZE,
        value_fontsize=VALUE_SIZE,
        section_fontsize=SECTION_SIZE,
        classic_section_fontsize=CLASSIC_SECTION_SIZE,
        legend_fontsize=LEGEND_SIZE,
        ax=ax_a,
        save=False,
    )

    # B: precision and recall
    plot_recall_precision(
        infile=infile,
        outdir=outdir,
        ylabeltitle="Score",
        xlabeltitle=None,
        xlabel_fontsize=AXIS_LABEL_SIZE,
        ylabel_fontsize=AXIS_LABEL_SIZE,
        xtick_fontsize=XTICK_SIZE,
        ytick_fontsize=YTICK_SIZE,
        legend_fontsize=LEGEND_SIZE,
        value_fontsize=VALUE_SIZE,
        ax=ax_b,
        save=False,
    )

    # C: ROC curves
    plot_roc_curves(
        infile=infile,
        outdir=outdir,
        xlabel_fontsize=AXIS_LABEL_SIZE,
        ylabel_fontsize=AXIS_LABEL_SIZE,
        xtick_fontsize=XTICK_SIZE,
        ytick_fontsize=YTICK_SIZE,
        legend_fontsize=LEGEND_SIZE,
        legend_title_fontsize=LEGEND_TITLE_SIZE,
        ax=ax_c,
        save=False,
    )

    add_panel_label(ax_a, "A", x=-0.04)
    add_panel_label(ax_b, "B", x=-0.12)
    add_panel_label(ax_c, "C", x=-0.12)

    fig.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        top=TOP_MARGIN,
        bottom=BOTTOM_MARGIN,
    )

    outfile = outdir / OUTPUT_NAME
    fig.savefig(outfile, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {outfile}")

def main():
    parser = argparse.ArgumentParser(
        description="Create Figure 1 (A: F1, B: precision/recall, C: ROC) from predictions_report.xlsx."
    )
    parser.add_argument("--infile", default=str(INFILE), help="Path to predictions_report.xlsx.")
    parser.add_argument("--outdir", default=str(OUTDIR), help="Output directory.")
    args = parser.parse_args()

    create_panel(infile=Path(args.infile), outdir=Path(args.outdir))

if __name__ == "__main__":
    main()