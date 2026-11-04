#!/usr/bin/env python3
"""
2x3 UMAP panel coloured by Effect.

Panel order:
    Mut-Site | Local-Mean | Full-Mean
    Delta Mut-Site | Delta Local-Mean | Delta Full-Mean

Usage:
    python umap_panel_effect.py \
        --parquet-dir results_training/split_1/parquet \
        --split train \
        --output results_training/split_1/umap_effect_panel.png

Requires: pandas, numpy, matplotlib, pyarrow, umap-learn, scikit-learn
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Editable settings

# Output
FIG_WIDTH = 15
FIG_HEIGHT = 10
DPI = 300
OUTPUT_FORMAT = "png"

# Points
POINT_SIZE = 2.5
POINT_ALPHA = 0.85
POINT_LINEWIDTH = 0
SHUFFLE_BEFORE_PLOT = True
RANDOM_STATE = 42

# Text
FONT_SIZE = 11
TITLE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
AXIS_TICK_FONT_SIZE = 11

# Axes
AXIS_X_LABEL = "UMAP-1"
AXIS_Y_LABEL = "UMAP-2"
AXIS_SPINE_LINEWIDTH = 1.0
EQUAL_ASPECT = True
SQUARE_AXES = True            # each subplot has width = height
SHARED_AXIS_LIMITS = False     # all panels use the same limits
AXIS_PADDING_FRACTION = 0.00  # margin around the points

# UMAP
MIN_DIST = 0.8
N_NEIG = 150
N_COMPONENTS = 2
METRIC_DISTANCE = "cosine"
STANDARDIZE_BEFORE_UMAP = True

# Effect colours
EFFECT_COLUMN = "Effect"
FUNCTIONAL_LABEL = "functional"
NEUTRAL_LABEL = "neutral"
FUNCTIONAL_COLOR = "#1f77b4"  # blue
NEUTRAL_COLOR = "#ff7f0e"     # orange
EFFECT_COLORS = {
    FUNCTIONAL_LABEL: FUNCTIONAL_COLOR,
    NEUTRAL_LABEL: NEUTRAL_COLOR,
}

# Layout
SUBPLOT_WSPACE = 0.22
SUBPLOT_HSPACE = 0.28
LEGEND_LOC = "lower center"
LEGEND_NCOL = 2
LEGEND_BBOX = (0.5, 0.015)
BOTTOM_MARGIN = 0.10

# Robust outlier clipping
CLIP_OUTLIERS = True
CLIP_PERCENTILES = (1.0, 99.0)
CLIP_POINTS_TO_BORDER = True

# SIFT variants are excluded
EXCLUDE_MODEL_SUBSTRINGS = ("sift",)

# Panel keys and titles; the order sets the position in the panel
PANEL_SPECS = [
    ("mutsite", "Mut-Site"),
    ("localmean", "Local-Mean"),
    ("fullmean", "Full-Mean"),
    ("mutsite_delta", r"Delta Mut-Site"),
    ("localmean_delta", r"Delta Local-Mean"),
    ("fullmean_delta", r"Delta Full-Mean"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("umap_panel_2x3_effect")

def normalize_name(value: str) -> str:
    """Normalise names so variants with hyphens or underscores match."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())

def identify_model(path: Path, df: pd.DataFrame | None = None) -> str | None:
    """Return one of the six PANEL_SPECS keys, or None."""
    candidates = [path.stem]
    if df is not None and "model" in df.columns and not df.empty:
        candidates.insert(0, str(df["model"].iloc[0]))

    for candidate in candidates:
        name = normalize_name(candidate)
        if any(excluded in name for excluded in EXCLUDE_MODEL_SUBSTRINGS):
            return None

        is_delta = "delta" in name
        if "mutsite" in name:
            return "mutsite_delta" if is_delta else "mutsite"
        if "localmean" in name:
            return "localmean_delta" if is_delta else "localmean"
        if "fullmean" in name:
            return "fullmean_delta" if is_delta else "fullmean"

    return None

def discover_parquets(parquet_dir: Path, split: str) -> list[Path]:
    """Find the parquet files from manifest.tsv or by file name."""
    manifest = parquet_dir / "manifest.tsv"
    paths: list[Path] = []

    if manifest.exists():
        man = pd.read_csv(manifest, sep="\t")
        required = {"split", "path"}
        if required.issubset(man.columns):
            man = man[man["split"].astype(str) == split]
            for raw_path in man["path"]:
                path = Path(raw_path)
                if not path.is_absolute():
                    candidate = parquet_dir / path
                    if candidate.exists():
                        path = candidate
                if path.exists():
                    paths.append(path)

    if not paths:
        paths = sorted(parquet_dir.glob(f"*_{split}.parquet"))
    if not paths:
        paths = sorted(parquet_dir.glob("*.parquet"))

    return sorted(
        p for p in paths
        if not any(excluded in normalize_name(p.stem) for excluded in EXCLUDE_MODEL_SUBSTRINGS)
    )

def clip_embedding_for_plot(embedding: np.ndarray) -> np.ndarray:
    """
    Clip UMAP coordinates by percentiles so a few outliers do not
    stretch the plot limits. The original embedding is not modified.
    """
    emb = np.asarray(embedding, dtype=float).copy()

    if not CLIP_OUTLIERS:
        return emb

    low_pct, high_pct = CLIP_PERCENTILES

    if not 0 <= low_pct < high_pct <= 100:
        raise ValueError(
            "CLIP_PERCENTILES must satisfy "
            "0 <= lower percentile < upper percentile <= 100."
        )

    x_low, x_high = np.nanpercentile(
        emb[:, 0],
        [low_pct, high_pct],
    )
    y_low, y_high = np.nanpercentile(
        emb[:, 1],
        [low_pct, high_pct],
    )

    if CLIP_POINTS_TO_BORDER:
        emb[:, 0] = np.clip(emb[:, 0], x_low, x_high)
        emb[:, 1] = np.clip(emb[:, 1], y_low, y_high)

    return emb

def compute_umap(features: np.ndarray) -> np.ndarray:
    import umap

    X = np.asarray(features, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError(f"Feature matrix must be 2D with at least 3 rows; shape={X.shape}")

    if STANDARDIZE_BEFORE_UMAP:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

    n_neighbors = min(N_NEIG, X.shape[0] - 1)
    if n_neighbors != N_NEIG:
        logger.warning("N_NEIG=%d reduced to %d for n=%d.", N_NEIG, n_neighbors, X.shape[0])

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=MIN_DIST,
        n_components=N_COMPONENTS,
        metric=METRIC_DISTANCE,
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(X)

def normalize_effect(series: pd.Series) -> pd.Series:
    labels = series.astype("string").str.strip().str.lower()
    valid = labels.isin(EFFECT_COLORS)
    if not valid.all():
        unknown = sorted(labels[~valid].dropna().unique().tolist())
        raise ValueError(
            f"Effect has categories other than functional/neutral: {unknown}"
        )
    return labels

def plot_effect(ax: plt.Axes, embedding: np.ndarray, effects: pd.Series, title: str) -> None:
    labels = normalize_effect(effects).to_numpy()
    order = np.arange(len(labels))
    if SHUFFLE_BEFORE_PLOT:
        rng = np.random.default_rng(RANDOM_STATE)
        rng.shuffle(order)

    emb = embedding[order]
    labels = labels[order]

    for category in (FUNCTIONAL_LABEL, NEUTRAL_LABEL):
        mask = labels == category
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            linewidths=POINT_LINEWIDTH,
            c=EFFECT_COLORS[category],
            rasterized=True,
        )

    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel(AXIS_X_LABEL, fontsize=FONT_SIZE)
    ax.set_ylabel(AXIS_Y_LABEL, fontsize=FONT_SIZE)
    ax.tick_params(labelsize=AXIS_TICK_FONT_SIZE)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_SPINE_LINEWIDTH)

    if EQUAL_ASPECT:
        ax.set_aspect("equal", adjustable="box")

    # Force a square box for each subplot so the layout does not stretch it
    if SQUARE_AXES:
        ax.set_box_aspect(1)

def load_six_models(paths: list[Path]) -> dict[str, tuple[np.ndarray, pd.Series]]:
    models: dict[str, tuple[np.ndarray, pd.Series]] = {}

    for path in paths:
        logger.info("Reading %s", path)
        df = pd.read_parquet(path)
        key = identify_model(path, df)
        if key is None:
            logger.info("Skipping %s: not one of the six panels.", path.name)
            continue
        if key in models:
            raise ValueError(f"Several parquet files found for panel '{key}'.")
        if "feature" not in df.columns:
            raise ValueError(f"{path.name} has no 'feature' column.")
        if EFFECT_COLUMN not in df.columns:
            raise ValueError(f"{path.name} has no '{EFFECT_COLUMN}' column.")

        features = np.asarray(df["feature"].tolist(), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedding = compute_umap(features)

        models[key] = (embedding, df[EFFECT_COLUMN])
        logger.info("Panel %s computed: %d points.", key, len(df))

    expected = {key for key, _ in PANEL_SPECS}
    missing = expected.difference(models)
    if missing:
        raise FileNotFoundError(
            "Missing parquet files for: " + ", ".join(sorted(missing))
        )

    return models

def get_shared_square_limits(
    models: dict[str, tuple[np.ndarray, pd.Series]]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Common limits for the six UMAPs, with the same range on X and Y so
    the scale is identical and panels are square without distortion.
    """
    all_embeddings = [models[key][0] for key, _ in PANEL_SPECS]

    xmin = min(float(emb[:, 0].min()) for emb in all_embeddings)
    xmax = max(float(emb[:, 0].max()) for emb in all_embeddings)
    ymin = min(float(emb[:, 1].min()) for emb in all_embeddings)
    ymax = max(float(emb[:, 1].max()) for emb in all_embeddings)

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    # One range for both axes
    span = max(xmax - xmin, ymax - ymin)
    if span == 0:
        span = 1.0

    half_span = span * (0.5 + AXIS_PADDING_FRACTION)

    x_limits = (x_center - half_span, x_center + half_span)
    y_limits = (y_center - half_span, y_center + half_span)

    return x_limits, y_limits

def create_panel(models: dict[str, tuple[np.ndarray, pd.Series]], output: Path) -> None:
    plt.rcParams.update({"font.size": FONT_SIZE})

    # 3:2 ratio for a 3-column x 2-row grid (square cells with 15 x 10)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        squeeze=False,
    )

    if SHARED_AXIS_LIMITS:
        x_limits, y_limits = get_shared_square_limits(models)
    else:
        x_limits = y_limits = None

    for ax, (key, title) in zip(axes.flat, PANEL_SPECS):
        embedding, effects = models[key]

        plot_effect(ax, embedding, effects, title)

        # Same limits for all UMAPs
        if SHARED_AXIS_LIMITS:
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)

        # Keep the box square after setting limits
        if SQUARE_AXES:
            ax.set_box_aspect(1)

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=FUNCTIONAL_COLOR,
            markeredgewidth=0,
            markersize=7,
            label=FUNCTIONAL_LABEL,
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=NEUTRAL_COLOR,
            markeredgewidth=0,
            markersize=7,
            label=NEUTRAL_LABEL,
        ),
    ]
    fig.legend(
        handles=handles,
        loc=LEGEND_LOC,
        ncol=LEGEND_NCOL,
        bbox_to_anchor=LEGEND_BBOX,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.94,
        bottom=BOTTOM_MARGIN,
        wspace=SUBPLOT_WSPACE,
        hspace=SUBPLOT_HSPACE,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=DPI,
        format=output.suffix.lstrip(".") or OUTPUT_FORMAT,
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info("Panel saved to %s", output)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output panel path, e.g. umap_effect_panel.png",
    )
    args = parser.parse_args()

    paths = discover_parquets(args.parquet_dir, args.split)
    if not paths:
        logger.error("No parquet files found in %s", args.parquet_dir)
        return 1

    try:
        models = load_six_models(paths)
        create_panel(models, args.output)
    except Exception as exc:
        logger.exception("Could not create the panel: %s", exc)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
