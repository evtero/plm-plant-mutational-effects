#!/usr/bin/env python3
"""
2x3 UMAP panel coloured by the first annotation term of each
mutation/protein.

Panel order:
    Mut-Site | Local-Mean | Full-Mean
    Delta Mut-Site | Delta Local-Mean | Delta Full-Mean

Usage:
    python umap_panel_ann_component.py \
        --parquet-dir results_training/split_1/parquet \
        --split train \
        --output results_training/split_1/umap_ann_panel.png

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
FIG_WIDTH = 19
FIG_HEIGHT = 10
DPI = 300
OUTPUT_FORMAT = "png"

# Point size per panel
DEFAULT_POINT_SIZE = 3.2
POINT_SIZE_BY_PANEL = {
    "mutsite": 3.2,
    "localmean": 3.2,
    "fullmean": 2.5,
    "mutsite_delta": 3.2,
    "localmean_delta": 3.2,
    "fullmean_delta": 3.2,
}

POINT_ALPHA = 0.70
POINT_LINEWIDTH = 0
OTHER_POINT_SIZE_FACTOR = 0.60
OTHER_ALPHA_FACTOR = 0.35
SHUFFLE_BEFORE_PLOT = True
RANDOM_STATE = 42

# Robust outlier clipping per panel
CLIP_OUTLIERS_BY_PANEL = {
    "mutsite": False,
    "localmean": False,
    "fullmean": True,
    "mutsite_delta": False,
    "localmean_delta": False,
    "fullmean_delta": False,
}

DEFAULT_CLIP_PERCENTILES = (1.0, 99.0)
CLIP_PERCENTILES_BY_PANEL = {
    "mutsite": (1.0, 99.0),
    "localmean": (1.0, 99.0),
    "fullmean": (1.0, 98.5),
    "mutsite_delta": (1.0, 99.0),
    "localmean_delta": (1.0, 99.0),
    "fullmean_delta": (1.0, 99.0),
}

# True: keep outliers by moving them to the border.
# False: leave them outside the visible range.
CLIP_POINTS_TO_BORDER = False

# Text
FONT_SIZE = 13
TITLE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 13
AXIS_TICK_FONT_SIZE = 11

# Axes
AXIS_X_LABEL = "UMAP-1"
AXIS_Y_LABEL = "UMAP-2"
AXIS_SPINE_LINEWIDTH = 1.0
EQUAL_ASPECT = True
SQUARE_AXES = True
SHARED_AXIS_LIMITS = False
AXIS_PADDING_FRACTION = 0.03

# UMAP
MIN_DIST = 0.8
N_NEIG = 150
N_COMPONENTS = 2
METRIC_DISTANCE = "cosine"
STANDARDIZE_BEFORE_UMAP = True

# Annotation
ANN_COLUMN = "interpro_family_names"
ANN_SEPARATOR = "|"
TOP_N = 10
OTHER_LABEL = "Other"
OTHER_COLOR = "#EEEEEE"
CATEGORICAL_CMAP = "tab10"

# Layout
SUBPLOT_WSPACE = 0.22
SUBPLOT_HSPACE = 0.28
LEGEND_LOC = "lower center"
LEGEND_NCOL = 4
LEGEND_BBOX = (0.5, 0.01)
BOTTOM_MARGIN = 0.20

# SIFT variants are excluded
EXCLUDE_MODEL_SUBSTRINGS = ("sift",)

# Panel keys and titles
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
logger = logging.getLogger("umap_panel_2x3_ann_component")

def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())

def identify_model(path: Path, df: pd.DataFrame | None = None) -> str | None:
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
    manifest = parquet_dir / "manifest.tsv"
    paths: list[Path] = []

    if manifest.exists():
        man = pd.read_csv(manifest, sep="\t")
        if {"split", "path"}.issubset(man.columns):
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


def first_ann_term(value) -> str:
    """First term of a '|'-separated list."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return OTHER_LABEL

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return OTHER_LABEL

    first = text.split(ANN_SEPARATOR)[0].strip()
    return first if first else OTHER_LABEL


def build_ann_colors(labels_by_model: dict[str, pd.Series]) -> tuple[list[str], dict[str, object]]:
    """Global TOP_N most frequent first terms."""
    all_labels = pd.concat(labels_by_model.values(), ignore_index=True)
    counts = all_labels[all_labels != OTHER_LABEL].value_counts()
    top_terms = counts.head(TOP_N).index.tolist()

    cmap = plt.get_cmap(CATEGORICAL_CMAP)
    colors = {term: cmap(i % cmap.N) for i, term in enumerate(top_terms)}
    colors[OTHER_LABEL] = OTHER_COLOR
    return top_terms, colors

def clip_embedding_for_plot(
    embedding: np.ndarray,
    panel_key: str,
) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
    """Clip only the panels enabled in CLIP_OUTLIERS_BY_PANEL."""
    emb = np.asarray(embedding, dtype=float).copy()

    if not CLIP_OUTLIERS_BY_PANEL.get(panel_key, False):
        return emb, None

    low_pct, high_pct = CLIP_PERCENTILES_BY_PANEL.get(
        panel_key, DEFAULT_CLIP_PERCENTILES
    )
    if not 0 <= low_pct < high_pct <= 100:
        raise ValueError(
            f"Invalid percentiles for {panel_key}: {(low_pct, high_pct)}"
        )

    x_low, x_high = np.nanpercentile(emb[:, 0], [low_pct, high_pct])
    y_low, y_high = np.nanpercentile(emb[:, 1], [low_pct, high_pct])
    robust_limits = (float(x_low), float(x_high), float(y_low), float(y_high))

    if CLIP_POINTS_TO_BORDER:
        emb[:, 0] = np.clip(emb[:, 0], x_low, x_high)
        emb[:, 1] = np.clip(emb[:, 1], y_low, y_high)

    return emb, robust_limits

def set_tight_square_limits(ax: plt.Axes, embedding: np.ndarray) -> None:
    xmin = float(embedding[:, 0].min())
    xmax = float(embedding[:, 0].max())
    ymin = float(embedding[:, 1].min())
    ymax = float(embedding[:, 1].max())

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    if span == 0:
        span = 1.0

    half_span = span * (0.5 + AXIS_PADDING_FRACTION)
    ax.set_xlim(x_center - half_span, x_center + half_span)
    ax.set_ylim(y_center - half_span, y_center + half_span)


def set_square_limits_from_bounds(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
) -> None:
    xmin, xmax, ymin, ymax = bounds
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    if span == 0:
        span = 1.0
    half_span = span * (0.5 + AXIS_PADDING_FRACTION)
    ax.set_xlim(x_center - half_span, x_center + half_span)
    ax.set_ylim(y_center - half_span, y_center + half_span)


def get_shared_square_limits(
    models: dict[str, tuple[np.ndarray, pd.Series]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_embeddings = [models[key][0] for key, _ in PANEL_SPECS]

    xmin = min(float(emb[:, 0].min()) for emb in all_embeddings)
    xmax = max(float(emb[:, 0].max()) for emb in all_embeddings)
    ymin = min(float(emb[:, 1].min()) for emb in all_embeddings)
    ymax = max(float(emb[:, 1].max()) for emb in all_embeddings)

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    if span == 0:
        span = 1.0

    half_span = span * (0.5 + AXIS_PADDING_FRACTION)
    return (
        (x_center - half_span, x_center + half_span),
        (y_center - half_span, y_center + half_span),
    )


def plot_ann_component(
    ax: plt.Axes,
    embedding: np.ndarray,
    labels: pd.Series,
    title: str,
    top_terms: list[str],
    colors: dict[str, object],
    point_size: float,
) -> None:
    labels = labels.where(labels.isin(top_terms), OTHER_LABEL).to_numpy()

    order = np.arange(len(labels))
    if SHUFFLE_BEFORE_PLOT:
        rng = np.random.default_rng(RANDOM_STATE)
        rng.shuffle(order)

    emb = embedding[order]
    labels = labels[order]

    # Other first, so it stays in the background
    for category in [OTHER_LABEL] + top_terms:
        mask = labels == category
        if not mask.any():
            continue

        is_other = category == OTHER_LABEL
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=point_size * OTHER_POINT_SIZE_FACTOR if is_other else point_size,
            alpha=POINT_ALPHA * OTHER_ALPHA_FACTOR if is_other else POINT_ALPHA,
            linewidths=POINT_LINEWIDTH,
            c=[colors[category]],
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
        if ANN_COLUMN not in df.columns:
            raise ValueError(f"{path.name} has no '{ANN_COLUMN}' column.")

        features = np.asarray(df["feature"].tolist(), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedding = compute_umap(features)

        labels = df[ANN_COLUMN].map(first_ann_term)
        models[key] = (embedding, labels)
        logger.info("Panel %s computed: %d points.", key, len(df))

    expected = {key for key, _ in PANEL_SPECS}
    missing = expected.difference(models)
    if missing:
        raise FileNotFoundError("Missing parquet files for: " + ", ".join(sorted(missing)))

    return models


def create_panel(models: dict[str, tuple[np.ndarray, pd.Series]], output: Path) -> None:
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), squeeze=False)

    labels_by_model = {key: labels for key, (_, labels) in models.items()}
    top_terms, colors = build_ann_colors(labels_by_model)

    if SHARED_AXIS_LIMITS:
        x_limits, y_limits = get_shared_square_limits(models)
    else:
        x_limits = y_limits = None

    for ax, (key, title) in zip(axes.flat, PANEL_SPECS):
        embedding, labels = models[key]

        point_size = POINT_SIZE_BY_PANEL.get(key, DEFAULT_POINT_SIZE)
        embedding_plot, robust_limits = clip_embedding_for_plot(
            embedding, panel_key=key
        )

        plot_ann_component(
            ax, embedding_plot, labels, title, top_terms, colors,
            point_size=point_size,
        )

        if SHARED_AXIS_LIMITS:
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
        elif robust_limits is not None:
            set_square_limits_from_bounds(ax, robust_limits)
        else:
            set_tight_square_limits(ax, embedding_plot)

        if SQUARE_AXES:
            ax.set_box_aspect(1)

    legend_order = top_terms + [OTHER_LABEL]
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor=colors[label],
            markeredgewidth=0,
            markersize=7,
            label=label,
        )
        for label in legend_order
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
        help="Output panel path, e.g. umap_ann_component_panel.png",
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
