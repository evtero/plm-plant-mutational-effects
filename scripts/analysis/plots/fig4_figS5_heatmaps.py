#!/usr/bin/env python3
"""
AUROC per category (taxonomy, GO, InterPro) as the mean across test splits.

For each split and category, the AUROC of every method is computed on the same
mutations, then averaged across splits. Each heatmap cell shows the mean
(optionally ± SD); the SD is always saved in the CSV.

Categories may overlap: a mutation can belong to several GO or InterPro annotations.

Filters within each split:
    - at least MIN_TOTAL mutations;
    - at least MIN_PER_CLASS mutations per class.

A category enters the summary only if it passes the filters in at least
MIN_SPLITS_PRESENT splits.

Taxonomic redundancy: a mutation appears in its class, order, family and genus,
so rows of different levels may show almost the same data. For each pair of
heatmap rows:
    redundancy(A, B) = shared mutations / mutations of the larger row
A mutation is identified by (split, Excel row) and only counts in the splits where
the row passes the filters, as in n_total_sum. 100% = both rows use exactly the
same mutations.

Outputs (OUTPUT_DIR):
    - <group>_all_splits.csv             : AUROC per split
    - <group>_consensus.csv              : mean and SD across splits
    - <group>_consensus.png              : individual heatmap (if SAVE_INDIVIDUAL)
    - <panel>.png                        : panel figures defined in PANELS
    - taxonomy_redundancy_pairs.csv      : overlap of each pair of rows
    - taxonomy_redundancy_by_row.csv     : most similar row and redundant group
    - taxonomy_redundancy_matrix.csv/png : redundancy matrix between rows

Usage:
    python auroc_category_heatmaps.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.metrics import roc_auc_score

# Data
INPUT_XLSX = Path("../../results/predictions/predictions_report.xlsx")  # change this to your path
OUTPUT_DIR = Path("../../results/heatmaps")  # change this to your path

SPLIT_SHEETS = [
    "split_1",
    "split_2",
    "split_3",
    "split_4",
    "split_5",
]

Y_TRUE_COL = "y_true"
PROTEIN_COL = "Id"
ORGANISM_COL = "Organism"

# Filters and statistics
MIN_TOTAL = 40
MIN_PER_CLASS = 20

# Minimum number of splits where the category must pass the filters
MIN_SPLITS_PRESENT = 2

# ddof=0: population SD; ddof=1: sample SD (as in the rest of the paper)
SD_DDOF = 1

# Methods
# Display name: (Excel column, invert sign)
# The sign is inverted when a low score means functional effect (y_true = 1).
# The order sets the heatmap column order.
METHODS = {
    "PLM-EmbSIFT": ("mutsite_sift_y_proba", False),
    "PLM-Emb": ("mutsite_y_proba", False),
    "PLM-ZS": ("esm1v_score", True),
    "SIFT": ("sift_score", True),
    "PROVEAN": ("provean_score", True),
}


# Categories
# Single column (taxonomy) or (id_col, name_col) for "|"-separated annotations.
# Comment out entries to exclude them.
CATEGORY_GROUPS = {
    "Taxonomic class": "tax_class",
    "Taxonomic order": "tax_order",
    "Taxonomic family": "tax_family",
    "Taxonomic genus": "tax_genus",
    # "Taxonomic species": "tax_species",

    "GO cellular component": ("go_C_ids", "go_C_terms"),
    "GO biological process": ("go_P_ids", "go_P_terms"),
    "GO molecular function": ("go_F_ids", "go_F_terms"),

    "InterPro families": ("interpro_family_ids", "interpro_family_names"),
    # "InterPro homologous superfamilies": (
    #     "interpro_homologous_superfamily_ids",
    #     "interpro_homologous_superfamily_names",
    # ),
    # "InterPro domains": ("interpro_domain_ids", "interpro_domain_names"),
}

# Lineage fix for genera with missing or wrong taxonomy.
# Applied to rows whose Organism starts with "<genus> ".
TAXONOMY_OVERRIDES = {
    "Arabidopsis": {
        "tax_genus": "Arabidopsis",
        "tax_family": "Brassicaceae",
        "tax_order": "Brassicales",
        "tax_class": "Magnoliopsida",
    },
    "Oryza": {
        "tax_genus": "Oryza",
        "tax_family": "Poaceae",
        "tax_order": "Poales",
        "tax_class": "Liliopsida",
    },
    "Triticum": {
        "tax_genus": "Triticum",
        "tax_family": "Poaceae",
        "tax_order": "Poales",
        "tax_class": "Liliopsida",
    },
    "Zea": {
        "tax_genus": "Zea",
        "tax_family": "Poaceae",
        "tax_order": "Poales",
        "tax_class": "Liliopsida",
    },
    "Hordeum": {
        "tax_genus": "Hordeum",
        "tax_family": "Poaceae",
        "tax_order": "Poales",
        "tax_class": "Liliopsida",
    },
}

# Taxonomic redundancy
# CATEGORY_GROUPS included in the analysis, from high to low level
REDUNDANCY_TITLES = [
    "Taxonomic class",
    "Taxonomic order",
    "Taxonomic family",
    "Taxonomic genus",
]

# Two rows with redundancy >= this value are considered the same row
REDUNDANCY_THRESHOLD = 0.90
REDUNDANCY_CMAP = "Blues"

# Figures
# One heatmap per CATEGORY_GROUPS entry
SAVE_INDIVIDUAL = True

# Panel figures: name -> titles (must exist in CATEGORY_GROUPS) and columns.
# ncols = 1 stacks the panels vertically (better for long names, e.g. GO).
PANELS = {
    "taxonomy_panel": {
        "titles": [
            "Taxonomic class",
            "Taxonomic order",
            "Taxonomic family",
            "Taxonomic genus",
        ],
        "ncols": 2,
    },
}

PANEL_LABELS = ["A", "B", "C", "D", "E", "F"]

# Groups whose category names are written in italics (binomial nomenclature)
ITALIC_GROUPS = {
    "Taxonomic genus",
    "Taxonomic species",
}

# Cell content
SHOW_SD = False
DECIMALS = 3

# Grey text under each category. Available fields:
# means across splits: {n_total}, {n_pos}, {n_neg}, {n_proteins}
# sums across splits:  {n_total_sum}, {n_pos_sum}, {n_neg_sum}
# other:               {n_splits}
# "\n" splits the subtitle into several lines.
SUBTITLE_TEMPLATE = "(n={n_total_sum:.0f}; F={n_pos_sum:.0f}, N={n_neg_sum:.0f})\n(splits={n_splits}/5)"

# Row order: "overall" (mean of all methods) or a METHODS name
SORT_ROWS_BY = "overall"

# Titles
INDIVIDUAL_TITLE_TEMPLATE = "AUROC across {title}"
PANEL_TITLE_TEMPLATE = "{title}"

# Colour scale: AUROC <= 0.50 gets the minimum colour, so the gradient covers
# 0.50-1.00. The real value is still written in each cell.
AUC_MIN = 0.50
AUC_MAX = 1.00
CMAP = "viridis"
CBAR_LABEL = "Mean AUROC"

# Sizes (inches)
# Width of the cells (+ colorbar) of each heatmap without the row names; the width
# of the longest name is added, so cells do not shrink with long GO terms.
HEATMAP_WIDTH = 5.5
CHAR_WIDTH = 0.6           # mean character width (fraction of the font size)
ROW_HEIGHT = 0.62          # height per category (name + 2-line subtitle)
EXTRA_HEIGHT = 1.4         # space for the title and method names
MIN_FIG_HEIGHT = 3.0
DPI = 300

# Vertical position of name and subtitle within each row
# (row units; 0 = centre, negative = up). The name is centred at CATEGORY_LABEL_DY;
# the subtitle starts (top edge) at SUBTITLE_DY and grows downwards.
CATEGORY_LABEL_DY = -0.25
SUBTITLE_DY = -0.07

# Shared colorbar in panels: fraction of width reserved on the right
PANEL_CBAR_SPACE = 0.08

# Fonts
ANNOT_FONTSIZE = 10
METHOD_FONTSIZE = 10
CATEGORY_FONTSIZE = 10
SUBTITLE_FONTSIZE = 8.5
TITLE_FONTSIZE = 13
PANEL_LABEL_FONTSIZE = 15

# Data loading
def apply_taxonomy_overrides(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Fix the lineage of the genera in TAXONOMY_OVERRIDES."""
    if not TAXONOMY_OVERRIDES:
        return df

    if ORGANISM_COL not in df.columns:
        print(
            f"[WARNING] {split_name}: column '{ORGANISM_COL}' not found, "
            "TAXONOMY_OVERRIDES not applied."
        )
        return df

    organism = df[ORGANISM_COL].astype("string").str.strip()

    for genus, taxonomy in TAXONOMY_OVERRIDES.items():
        mask = (organism == genus) | organism.str.startswith(f"{genus} ", na=False)
        mask = mask.fillna(False)

        if mask.any():
            for column, value in taxonomy.items():
                df.loc[mask, column] = value

    return df

def load_splits() -> dict[str, pd.DataFrame]:
    """Read each sheet once and apply the taxonomy fix."""
    split_dfs = {}

    for split_name in SPLIT_SHEETS:
        print(f"[INFO] Reading {split_name}")
        df = pd.read_excel(INPUT_XLSX, sheet_name=split_name, engine="openpyxl")
        df.columns = df.columns.str.strip()
        split_dfs[split_name] = apply_taxonomy_overrides(df, split_name)

    return split_dfs

def check_required_columns(df: pd.DataFrame, spec) -> None:
    """Check that all needed columns exist."""
    required = {
        Y_TRUE_COL,
        PROTEIN_COL,
        *[column for column, _ in METHODS.values()],
    }

    if isinstance(spec, tuple):
        required.update(spec)
    else:
        required.add(spec)

    missing = sorted(required.difference(df.columns))

    if missing:
        raise KeyError(
            "Missing columns in the Excel file: " + ", ".join(missing)
        )

def split_annotation_field(value) -> list[str]:
    """Split a "|"-separated annotation field."""
    if pd.isna(value):
        return []

    return [item.strip() for item in str(value).split("|") if item.strip()]

def to_long(df: pd.DataFrame, spec) -> pd.DataFrame:
    """
    Long format: one row per mutation-category.

    A mutation can appear in several categories, but only once per category.
    """
    check_required_columns(df, spec)

    subset = df.copy()
    subset["_mutation_row"] = np.arange(len(subset))

    if isinstance(spec, tuple):
        id_col, name_col = spec

        def combine(ids_value, names_value):
            ids = split_annotation_field(ids_value)
            names = split_annotation_field(names_value)
            labels = []

            for i in range(max(len(ids), len(names))):
                annotation_id = ids[i] if i < len(ids) else ""
                annotation_name = names[i] if i < len(names) else ""

                if annotation_id and annotation_name:
                    label = f"{annotation_id} | {annotation_name}"
                else:
                    label = annotation_id or annotation_name

                if label:
                    labels.append(label)

            # Keep the order and drop duplicates
            return list(dict.fromkeys(labels))

        subset["_category"] = [
            combine(ids_value, names_value)
            for ids_value, names_value in zip(subset[id_col], subset[name_col])
        ]
        subset = subset.explode("_category")

    else:
        subset["_category"] = subset[spec]

    subset["_category"] = subset["_category"].astype("string").str.strip()
    subset = subset.dropna(subset=["_category"])
    subset = subset.loc[subset["_category"] != ""]

    return subset.drop_duplicates(subset=["_mutation_row", "_category"])


# AUROC per split and summary
def compute_auroc_table(
    long_df: pd.DataFrame, split_name: str
) -> tuple[pd.DataFrame, dict[str, set[int]]]:
    """
    AUROC per category in one split.

    All methods are evaluated on exactly the same mutations. Also returns, for
    each category that passes the filters, the set of evaluated mutations.
    """
    rows = []
    members = {}
    methods = list(METHODS)

    for category, group in long_df.groupby("_category", sort=False):
        evaluation = pd.DataFrame(index=group.index)
        evaluation["mutation_row"] = group["_mutation_row"]
        evaluation["protein"] = group[PROTEIN_COL].astype("string").str.strip()
        evaluation["y"] = pd.to_numeric(group[Y_TRUE_COL], errors="coerce")

        for method, (column, invert) in METHODS.items():
            scores = pd.to_numeric(group[column], errors="coerce")
            evaluation[method] = -scores if invert else scores

        evaluation = evaluation.loc[evaluation["y"].isin([0, 1])]
        evaluation = evaluation.dropna(subset=["protein", "y", *methods])
        evaluation = evaluation.loc[evaluation["protein"] != ""]

        n_total = len(evaluation)
        n_pos = int((evaluation["y"] == 1).sum())
        n_neg = int((evaluation["y"] == 0).sum())
        n_proteins = int(evaluation["protein"].nunique())

        if (
            n_total < MIN_TOTAL
            or n_pos < MIN_PER_CLASS
            or n_neg < MIN_PER_CLASS
        ):
            continue

        row = {
            "split": split_name,
            "category": category,
            "n_total": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_proteins": n_proteins,
        }

        for method in methods:
            row[method] = roc_auc_score(evaluation["y"], evaluation[method])

        rows.append(row)
        members[category] = set(evaluation["mutation_row"])

    columns = ["split", "category", "n_total", "n_pos", "n_neg", "n_proteins", *methods]
    return pd.DataFrame(rows, columns=columns), members

def summarize_across_splits(split_results: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of the AUROC across splits for each category."""
    if split_results.empty:
        return pd.DataFrame()

    rows = []

    for category, group in split_results.groupby("category", sort=False):
        n_splits = int(group["split"].nunique())

        if n_splits < MIN_SPLITS_PRESENT:
            continue

        row = {
            "category": category,
            "n_splits": n_splits,
            "n_total_mean": group["n_total"].mean(),
            "n_total_sd": group["n_total"].std(ddof=SD_DDOF),
            "n_pos_mean": group["n_pos"].mean(),
            "n_neg_mean": group["n_neg"].mean(),
            "n_total_sum": group["n_total"].sum(),
            "n_pos_sum": group["n_pos"].sum(),
            "n_neg_sum": group["n_neg"].sum(),
            "n_proteins_mean": group["n_proteins"].mean(),
            "n_proteins_sd": group["n_proteins"].std(ddof=SD_DDOF),
        }

        for method in METHODS:
            row[f"{method}_mean"] = group[method].mean()
            row[f"{method}_sd"] = group[method].std(ddof=SD_DDOF)

        rows.append(row)

    return pd.DataFrame(rows)


# Taxonomic redundancy
def level_name(title: str) -> str:
    """'Taxonomic family' -> 'Family'."""
    return title.replace("Taxonomic", "").strip().capitalize()

def compute_redundancy(
    summaries: dict[str, pd.DataFrame],
    members: dict[str, dict[tuple[str, str], set[int]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Mutation overlap between the taxonomic rows shown in the heatmaps.

    Returns (pairs, summary per row, redundancy matrix).
    """
    heatmap_rows = []

    for title in REDUNDANCY_TITLES:
        summary = summaries.get(title, pd.DataFrame())

        if summary.empty:
            continue

        ordered = summary.sort_values("n_total_sum", ascending=False)

        for category in ordered["category"]:
            mutations = {
                (split, mutation)
                for (split, cat), rows in members[title].items()
                if cat == category
                for mutation in rows
            }
            heatmap_rows.append({
                "level": level_name(title),
                "clade": category,
                "label": f"{level_name(title)}: {category}",
                "mutations": mutations,
            })

    n = len(heatmap_rows)
    labels = [row["label"] for row in heatmap_rows]
    matrix = np.full((n, n), np.nan)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = heatmap_rows[i], heatmap_rows[j]
            n_a, n_b = len(a["mutations"]), len(b["mutations"])
            shared = len(a["mutations"] & b["mutations"])
            redundancy = shared / max(n_a, n_b) if max(n_a, n_b) else np.nan
            matrix[i, j] = matrix[j, i] = redundancy

            if shared and a["level"] != b["level"]:
                pairs.append({
                    "level_A": a["level"],
                    "clade_A": a["clade"],
                    "level_B": b["level"],
                    "clade_B": b["clade"],
                    "n_A": n_a,
                    "n_B": n_b,
                    "n_shared": shared,
                    "frac_of_A": shared / n_a,
                    "frac_of_B": shared / n_b,
                    "redundancy": redundancy,
                    "redundant": redundancy >= REDUNDANCY_THRESHOLD,
                })

    # Groups: rows joined by redundancy >= threshold (connected components)
    group_of = list(range(n))

    def find(i):
        while group_of[i] != i:
            group_of[i] = group_of[group_of[i]]
            i = group_of[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] >= REDUNDANCY_THRESHOLD:
                group_of[find(i)] = find(j)

    roots = [find(i) for i in range(n)]
    group_ids = {root: k + 1 for k, root in enumerate(dict.fromkeys(roots))}

    by_row = []

    for i, row in enumerate(heatmap_rows):
        others = matrix[i].copy()
        j = int(np.nanargmax(others)) if n > 1 and not np.all(np.isnan(others)) else None
        group = [heatmap_rows[k]["label"] for k in range(n) if roots[k] == roots[i]]

        by_row.append({
            "level": row["level"],
            "clade": row["clade"],
            "n_mutations": len(row["mutations"]),
            "most_similar_row": heatmap_rows[j]["label"] if j is not None else "",
            "max_redundancy": others[j] if j is not None else np.nan,
            "group_id": group_ids[roots[i]],
            "group_size": len(group),
            "group_members": " ; ".join(group),
        })

    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values("redundancy", ascending=False)

    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)

    return pairs_df, pd.DataFrame(by_row), matrix_df

def plot_redundancy_matrix(
    matrix_df: pd.DataFrame, by_row: pd.DataFrame, output_png: Path
) -> None:
    """Redundancy matrix (% shared mutations) between taxonomic rows."""
    if matrix_df.empty:
        return

    n = len(matrix_df)
    labels = [
        f"{label} (n={n_mut})"
        for label, n_mut in zip(matrix_df.index, by_row["n_mutations"])
    ]
    annotation = matrix_df.apply(
        lambda column: column.map(lambda v: "" if pd.isna(v) or v == 0 else f"{v:.0%}")
    )

    size = 0.5 * n + 3
    fig, ax = plt.subplots(figsize=(size + 1.5, size))

    sns.heatmap(
        matrix_df,
        ax=ax,
        cmap=REDUNDANCY_CMAP,
        vmin=0,
        vmax=1,
        annot=annotation,
        fmt="",
        annot_kws={"fontsize": 8},
        linewidths=0.35,
        linecolor="white",
        mask=matrix_df.isna(),
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Shared variants (% of the larger row)", "shrink": 0.6},
    )

    # Lines between taxonomic levels
    levels = by_row["level"].tolist()
    for k in range(1, n):
        if levels[k] != levels[k - 1]:
            ax.axhline(k, color="#5f5e57", lw=1.2)
            ax.axvline(k, color="#5f5e57", lw=1.2)

    ax.tick_params(axis="x", labelsize=8.5, rotation=90)
    ax.tick_params(axis="y", labelsize=8.5, rotation=0)

    n_groups = by_row["group_id"].nunique()
    ax.set_title(
        f"Redundancy between taxonomic heatmap rows: {n} rows, "
        f"{n_groups} distinct at >= {REDUNDANCY_THRESHOLD:.0%}",
        fontsize=TITLE_FONTSIZE, loc="left",
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved: {output_png}")

def print_redundancy(pairs: pd.DataFrame, by_row: pd.DataFrame) -> None:
    """Redundancy summary on screen."""
    print("\n=== Redundancy between taxonomic rows ===")

    if by_row.empty:
        print("No taxonomic rows in the heatmaps.")
        return

    n_rows = len(by_row)
    n_groups = by_row["group_id"].nunique()
    n_redundant_rows = n_rows - n_groups

    print(f"Rows in the taxonomic heatmaps: {n_rows}")
    print(f"Distinct groups (redundancy < {REDUNDANCY_THRESHOLD:.0%}): {n_groups}")
    print(
        f"Redundant rows: {n_redundant_rows} "
        f"({100 * n_redundant_rows / n_rows:.0f}% of the rows)"
    )

    for level, group in by_row.groupby("level", sort=False):
        n_level = len(group)
        n_dup = int((group["group_size"] > 1).sum())
        print(f"  {level:8s}: {n_level} rows, {n_dup} in a redundant group")

    multi = by_row.loc[by_row["group_size"] > 1, "group_members"].drop_duplicates()
    if not multi.empty:
        print("Redundant groups:")
        for members in multi:
            print(f"  - {members}")


# Plotting
def sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Sort the categories by SORT_ROWS_BY."""
    mean_columns = [f"{method}_mean" for method in METHODS]

    if SORT_ROWS_BY == "overall":
        key = summary[mean_columns].mean(axis=1)
    elif SORT_ROWS_BY in METHODS:
        key = summary[f"{SORT_ROWS_BY}_mean"]
    else:
        raise ValueError(
            f"SORT_ROWS_BY='{SORT_ROWS_BY}' is neither 'overall' nor a METHODS name."
        )

    return summary.assign(_sort_key=key).sort_values("_sort_key", ascending=False)

def format_cell(mean_value, sd_value) -> str:
    """Cell text: mean, or mean ± SD if SHOW_SD."""
    if pd.isna(mean_value):
        return ""

    text = f"{mean_value:.{DECIMALS}f}"

    if SHOW_SD:
        sd_text = "NA" if pd.isna(sd_value) else f"{sd_value:.{DECIMALS}f}"
        text += f" ± {sd_text}"

    return text

def draw_heatmap(
    ax: plt.Axes,
    summary: pd.DataFrame,
    title: str,
    panel_label: str | None = None,
    cbar: bool = True,
) -> None:
    """
    Heatmap of one category group in `ax`.

    With `panel_label`, the bold letter is placed left of the title.
    """
    if panel_label:
        ax.set_title(
            rf"$\bf{{{panel_label}}}$    {PANEL_TITLE_TEMPLATE.format(title=title)}",
            loc="left",
            fontsize=TITLE_FONTSIZE,
        )
    else:
        ax.set_title(
            INDIVIDUAL_TITLE_TEMPLATE.format(title=title),
            fontsize=TITLE_FONTSIZE,
        )

    if summary.empty:
        ax.text(
            0.5, 0.5, "No categories passed the filters",
            ha="center", va="center", transform=ax.transAxes, color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    methods = list(METHODS)
    summary = sort_summary(summary).set_index("category")

    heat = summary[[f"{method}_mean" for method in methods]]
    heat.columns = methods

    annotation = pd.DataFrame(
        [
            [
                format_cell(summary.loc[category, f"{method}_mean"],
                            summary.loc[category, f"{method}_sd"])
                for method in methods
            ]
            for category in heat.index
        ],
        index=heat.index,
        columns=methods,
    )

    # Without "color" in annot_kws, seaborn picks white or black text per cell
    sns.heatmap(
        heat,
        ax=ax,
        cmap=CMAP,
        vmin=AUC_MIN,
        vmax=AUC_MAX,
        annot=annotation,
        fmt="",
        annot_kws={"fontsize": ANNOT_FONTSIZE},
        linewidths=0.35,
        linecolor="white",
        cbar=cbar,
        cbar_kws={"label": CBAR_LABEL} if cbar else None,
        mask=heat.isna(),
        yticklabels=False,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=METHOD_FONTSIZE, rotation=0)

    category_style = "italic" if title in ITALIC_GROUPS else "normal"

    for i, category in enumerate(heat.index):
        row = summary.loc[category]

        subtitle = SUBTITLE_TEMPLATE.format(
            n_total=row["n_total_mean"],
            n_pos=row["n_pos_mean"],
            n_neg=row["n_neg_mean"],
            n_total_sum=row["n_total_sum"],
            n_pos_sum=row["n_pos_sum"],
            n_neg_sum=row["n_neg_sum"],
            n_proteins=row["n_proteins_mean"],
            n_splits=int(row["n_splits"]),
        )

        y = i + 0.5

        ax.text(
            -0.01, y + CATEGORY_LABEL_DY, category,
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=CATEGORY_FONTSIZE,
            fontstyle=category_style,
        )
        ax.text(
            -0.01, y + SUBTITLE_DY, subtitle,
            transform=ax.get_yaxis_transform(),
            ha="right", va="top", fontsize=SUBTITLE_FONTSIZE, color="gray",
            linespacing=1.1,
        )

def label_width(summary: pd.DataFrame) -> float:
    """Approximate width (inches) of the longest name or subtitle line."""
    if summary.empty:
        return 0.0

    longest_category = summary["category"].astype(str).str.len().max()
    longest_subtitle = max(
        len(line)
        for line in SUBTITLE_TEMPLATE.format(
            n_total=0, n_pos=0, n_neg=0, n_proteins=0,
            n_total_sum=summary["n_total_sum"].max(),
            n_pos_sum=summary["n_pos_sum"].max(),
            n_neg_sum=summary["n_neg_sum"].max(),
            n_splits=5,
        ).split("\n")
    )

    return CHAR_WIDTH / 72 * max(
        longest_category * CATEGORY_FONTSIZE,
        longest_subtitle * SUBTITLE_FONTSIZE,
    )

def plot_individual(summary: pd.DataFrame, title: str, output_png: Path) -> None:
    """Individual heatmap of one category group."""
    if summary.empty:
        print(f"[WARNING] Not enough categories for: {title}")
        return

    fig_height = max(MIN_FIG_HEIGHT, ROW_HEIGHT * len(summary) + EXTRA_HEIGHT)
    fig_width = HEATMAP_WIDTH + label_width(summary)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    draw_heatmap(ax, summary, title, panel_label=None, cbar=True)

    fig.tight_layout()
    fig.savefig(output_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved: {output_png}")

def plot_panel(
    titles: list[str],
    ncols: int,
    summaries: dict[str, pd.DataFrame],
    output_png: Path,
) -> None:
    """
    Panel figure with one heatmap per title (A, B, C...) and a shared colorbar.

    With ncols = 1, the height of each panel is proportional to its number of rows.
    """
    n = len(titles)
    nrows = -(-n // ncols)
    n_rows_each = [max(1, len(summaries.get(t, pd.DataFrame()))) for t in titles]
    subplot_width = HEATMAP_WIDTH + max(
        label_width(summaries.get(t, pd.DataFrame())) for t in titles
    )

    if ncols == 1:
        heights = [ROW_HEIGHT * r + EXTRA_HEIGHT for r in n_rows_each]
        fig, axes = plt.subplots(
            nrows, 1,
            figsize=(subplot_width, sum(heights)),
            gridspec_kw={"height_ratios": heights},
        )
    else:
        subplot_height = max(MIN_FIG_HEIGHT, ROW_HEIGHT * max(n_rows_each) + EXTRA_HEIGHT)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(subplot_width * ncols, subplot_height * nrows),
        )

    axes = np.atleast_1d(axes).flatten()

    for ax, title, label in zip(axes, titles, PANEL_LABELS):
        draw_heatmap(ax, summaries.get(title, pd.DataFrame()), title,
                     panel_label=label, cbar=False)

    for ax in axes[n:]:
        ax.axis("off")

    # Leave space on the right for a shared colorbar
    fig.tight_layout(rect=[0, 0, 1 - PANEL_CBAR_SPACE, 1])

    cax = fig.add_axes([1 - PANEL_CBAR_SPACE * 0.6, 0.3, 0.012, 0.4])
    mappable = ScalarMappable(norm=Normalize(AUC_MIN, AUC_MAX), cmap=CMAP)
    fig.colorbar(mappable, cax=cax, label=CBAR_LABEL)

    fig.savefig(output_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Panel saved: {output_png}")


# Summary on screen
def print_summary(summary: pd.DataFrame, title: str) -> None:
    """Unweighted mean of the mean AUROC of all categories."""
    print(f"\n--- {title} ---")

    if summary.empty:
        print("No categories passed the filters.")
        return

    print(
        f"Categories: {len(summary)} "
        f"(present in >= {MIN_SPLITS_PRESENT} splits)"
    )

    for method in METHODS:
        label = method.replace("\n", " ")
        print(
            f"{label:25s}: "
            f"mean AUROC = {summary[f'{method}_mean'].mean():.3f} | "
            f"mean split SD = {summary[f'{method}_sd'].mean():.3f}"
        )

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_dfs = load_splits()
    summaries: dict[str, pd.DataFrame] = {}
    members: dict[str, dict[tuple[str, str], set[int]]] = {}

    for title, spec in CATEGORY_GROUPS.items():
        tables = []
        members[title] = {}

        for split_name, df in split_dfs.items():
            table, split_members = compute_auroc_table(to_long(df, spec), split_name)
            tables.append(table)
            members[title].update(
                {(split_name, category): rows for category, rows in split_members.items()}
            )

        split_results = pd.concat(tables, ignore_index=True)
        summary = summarize_across_splits(split_results)
        summaries[title] = summary

        stem = title.lower().replace(" ", "_").replace("/", "_").replace("-", "_")

        split_results.to_csv(OUTPUT_DIR / f"{stem}_all_splits.csv", index=False)
        summary.to_csv(OUTPUT_DIR / f"{stem}_consensus.csv", index=False)

        if SAVE_INDIVIDUAL:
            plot_individual(summary, title, OUTPUT_DIR / f"{stem}_consensus.png")

        print_summary(summary, title)

    for panel_name, config in PANELS.items():
        missing = [t for t in config["titles"] if t not in CATEGORY_GROUPS]
        if missing:
            print(f"[WARNING] {panel_name}: titles not defined in CATEGORY_GROUPS: {missing}")
            continue

        plot_panel(
            titles=config["titles"],
            ncols=config["ncols"],
            summaries=summaries,
            output_png=OUTPUT_DIR / f"{panel_name}.png",
        )

    # Redundancy between taxonomic levels
    titles = [t for t in REDUNDANCY_TITLES if t in CATEGORY_GROUPS]
    if titles:
        pairs, by_row, matrix = compute_redundancy(summaries, members)

        pairs.to_csv(OUTPUT_DIR / "taxonomy_redundancy_pairs.csv", index=False)
        by_row.to_csv(OUTPUT_DIR / "taxonomy_redundancy_by_row.csv", index=False)
        matrix.to_csv(OUTPUT_DIR / "taxonomy_redundancy_matrix.csv")

        plot_redundancy_matrix(matrix, by_row, OUTPUT_DIR / "taxonomy_redundancy_matrix.png")
        print_redundancy(pairs, by_row)

    print(f"\n[INFO] Results in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()