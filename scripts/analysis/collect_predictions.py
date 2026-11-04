#!/usr/bin/env python3
"""
Collect test predictions of all methods (SIFT, PROVEAN, ESM-1v and the 12 trained models) per split and compute their metrics (mean and SD across splits).
Output: split_<n>_predictions.tsv, metrics_split_<n>.tsv, metrics_mean_std.tsv and predictions_report.xlsx (all as sheets).
Usage: edit the paths below, then run: python collect_predictions.py
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SPLITS_DIR = Path("../data/splits_by_homology")  # change this to your path
SIFT_DIR = Path("../results/sift")  # change this to your path
PROVEAN_DIR = Path("../results/provean")  # change this to your path
ESM1V_SCORES = Path("../results/esm1v/esm1v_scores.tsv")  # change this to your path
TRAINING_DIR = Path("../results/training")  # change this to your path (OUTDIR of training_pipeline/main.py)
OUTDIR = Path("../results/predictions")  # change this to your path
N_SPLITS = 5
PROVEAN_CUTOFF = -2.5

CONFIGS = [
    embedding_type + suffix
    for embedding_type in ("full-mean", "local-mean", "mut-site")
    for suffix in ("", "_delta", "_delta_sift", "_sift")
]

# Output column order of the split_<n> tables
COLS1 = [
    "Number", "Source", "Organism", "Id", "Database", "Sequence", "AA_from", "AA_to",
    "AA_loci", "PMID", "Effect", "Description", "seq_id", "mutation_code", "mutation_id",
    "family_id", "split_group", "y_true", "sift_score", "sift_label", "esm1v_score", "esm1v_label",
]
COLS2 = [
    f"{config.replace('-', '')}_{col}" for config in CONFIGS for col in ("y_pred", "y_proba")
] + ["provean_score", "provean_label"]

ANNOTATION_COLS = [
    "go_C_ids", "go_C_terms", "go_C_evidence",
    "go_P_ids", "go_P_terms", "go_P_evidence",
    "go_F_ids", "go_F_terms", "go_F_evidence",
    "interpro_family_ids", "interpro_family_names",
    "interpro_domain_ids", "interpro_domain_names",
    "interpro_homologous_superfamily_ids", "interpro_homologous_superfamily_names",
    "interpro_conserved_site_ids", "interpro_conserved_site_names",
    "interpro_repeat_ids", "interpro_repeat_names",
    "interpro_active_site_ids", "interpro_active_site_names",
    "interpro_binding_site_ids", "interpro_binding_site_names",
    "interpro_ptm_ids", "interpro_ptm_names",
    "tax_class", "tax_order", "tax_family", "tax_genus", "tax_species",
]

OUTPUT_COLS = COLS1 + COLS2 + ANNOTATION_COLS

def _to_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# SIFT 
def _map_sift_label(x: str) -> str:
    if x == "DELETERIOUS":
        return "functional"
    if x == "TOLERATED":
        return "neutral"
    return "unknown"


def _parse_sift_file(pred_file: Path) -> dict:
    preds = {}
    with pred_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("WARNING!"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            preds[parts[0]] = (_to_float(parts[2]), _map_sift_label(parts[1].upper()))
    return preds


def load_sift(sift_dir: Path) -> pd.DataFrame:
    rows = []
    for pred_file in sift_dir.glob("*.SIFTprediction"):
        seq_id = pred_file.name.removesuffix(".SIFTprediction")
        for code, (score, label) in _parse_sift_file(pred_file).items():
            rows.append({"mutation_id": f"{seq_id}-{code}", "sift_score": score, "sift_label": label})
    return pd.DataFrame(rows, columns=["mutation_id", "sift_score", "sift_label"])


# PROVEAN 
def _parse_provean_file(out_file: Path, cutoff: float) -> dict:
    preds = {}
    in_scores = False
    with out_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("## PROVEAN scores"):
                in_scores = True
                continue
            if not in_scores or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            score = _to_float(parts[1])
            if score is None:
                label = "unknown"
            elif score <= cutoff:
                label = "functional"
            else:
                label = "neutral"
            preds[parts[0]] = (score, label)
    return preds


def load_provean(provean_dir: Path, cutoff: float) -> pd.DataFrame:
    rows = []
    for out_file in provean_dir.glob("*.provean.txt"):
        if out_file.stat().st_size == 0:
            continue
        seq_id = out_file.name.removesuffix(".provean.txt")
        for code, (score, label) in _parse_provean_file(out_file, cutoff).items():
            rows.append({"mutation_id": f"{seq_id}-{code}", "provean_score": score, "provean_label": label})
    return pd.DataFrame(rows, columns=["mutation_id", "provean_score", "provean_label"])


# ESM-1v zero-shot
def _gmm(vals):
    # Two components: the one with the lower mean is functional/deleterious
    try:
        from sklearn.mixture import GaussianMixture

        X = np.array(vals).reshape(-1, 1)
        g = GaussianMixture(n_components=2, random_state=0).fit(X)
        deleterious = int(g.means_.flatten().argmin())
        return deleterious, lambda x: int(g.predict([[x]])[0])

    except Exception:
        # Simple fallback if the GMM cannot be fitted
        s = sorted(vals)
        mid = len(s) // 2
        a = sum(s[:mid]) / mid
        b = sum(s[mid:]) / (len(s) - mid)
        deleterious = 0 if a < b else 1
        return deleterious, lambda x: (
            deleterious if abs(x - a) < abs(x - b) else 1 - deleterious
        )


def load_esm1v(scores_path: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_path, sep="\t")
    scores = df["score"].dropna().tolist()

    # Global GMM fitted on all final scores
    if len(scores) >= 2:
        deleterious, assign = _gmm(scores)
        df["esm1v_label"] = [
            "unknown" if pd.isna(s) else ("functional" if assign(s) == deleterious else "neutral")
            for s in df["score"]
        ]
    else:
        df["esm1v_label"] = "unknown"

    return df.rename(columns={"score": "esm1v_score"})[["mutation_id", "esm1v_score", "esm1v_label"]]


# Training pipeline
def load_training(training_dir: Path, config: str, split: int):
    path = training_dir / config / f"split_{split}" / "models" / "test_predictions.tsv"
    if not path.exists():
        print(f"[COLLECT] missing: {path}")
        return None

    name = config.replace("-", "")  # e.g. mut-site_delta -> mutsite_delta
    df = pd.read_csv(path, sep="\t")[["mutation_id", "y_pred", "y_proba"]]
    return df.rename(columns={"y_pred": f"{name}_y_pred", "y_proba": f"{name}_y_proba"})


def merge(table, preds):
    if preds is None or preds.empty:
        return table
    return table.merge(preds.drop_duplicates("mutation_id"), on="mutation_id", how="left")


# Metrics
# Metrics of one method on the variants it predicted (functional = 1)
def method_metrics(name, y_true, y_pred, score):
    df = pd.DataFrame({"y": y_true, "pred": y_pred, "score": score}).dropna()
    y, pred = df["y"].astype(int), df["pred"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "method": name,
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "auroc": roc_auc_score(y, df["score"]) if y.nunique() == 2 else np.nan,
    }


def split_metrics(table):
    labels = {"functional": 1, "neutral": 0}
    rows = []

    # Classic and zero-shot methods: label -> prediction, negated score -> AUROC
    for name in ("sift", "provean", "esm1v"):
        if f"{name}_label" in table.columns:
            rows.append(method_metrics(
                name,
                table["y_true"],
                table[f"{name}_label"].map(labels),
                -table[f"{name}_score"],
            ))

    # Trained models: y_pred -> prediction, y_proba -> AUROC
    for config in CONFIGS:
        name = config.replace("-", "")
        if f"{name}_y_pred" in table.columns:
            rows.append(method_metrics(
                name, table["y_true"], table[f"{name}_y_pred"], table[f"{name}_y_proba"]
            ))

    return pd.DataFrame(rows)


def mean_std(metrics_by_split):
    stats = pd.concat(metrics_by_split).groupby("method").agg(["mean", "std"])
    stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]
    return stats.reset_index()


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    sift = load_sift(SIFT_DIR) if SIFT_DIR.exists() else None
    provean = load_provean(PROVEAN_DIR, PROVEAN_CUTOFF) if PROVEAN_DIR.exists() else None
    esm1v = load_esm1v(ESM1V_SCORES) if ESM1V_SCORES.exists() else None

    tables = {}
    metrics_by_split = []

    for n in range(1, N_SPLITS + 1):
        table = pd.read_csv(SPLITS_DIR / f"split_{n}" / "test.tsv", sep="\t")
        table["y_true"] = table["Effect"].str.strip().str.lower().map({"functional": 1, "neutral": 0})

        table = merge(table, sift)
        table = merge(table, esm1v)
        for config in CONFIGS:
            table = merge(table, load_training(TRAINING_DIR, config, n))
        table = merge(table, provean)

        table = table[[c for c in OUTPUT_COLS if c in table.columns]]

        out = OUTDIR / f"split_{n}_predictions.tsv"
        table.to_csv(out, sep="\t", index=False)
        print(f"[COLLECT] split_{n}: {table.shape[0]} variants, {table.shape[1]} columns -> {out}")

        metrics = split_metrics(table)
        metrics.to_csv(OUTDIR / f"metrics_split_{n}.tsv", sep="\t", index=False)
        metrics_by_split.append(metrics)

        tables[n] = table

    summary = mean_std(metrics_by_split)
    summary.to_csv(OUTDIR / "metrics_mean_std.tsv", sep="\t", index=False)

    with pd.ExcelWriter(OUTDIR / "predictions_report.xlsx") as writer:
        for n, table in tables.items():
            table.to_excel(writer, sheet_name=f"split_{n}", index=False)
        for n, metrics in enumerate(metrics_by_split, start=1):
            metrics.to_excel(writer, sheet_name=f"metrics_split_{n}", index=False)
        summary.to_excel(writer, sheet_name="metrics_mean_std", index=False)

    print(summary[["method", "accuracy_mean", "f1_mean", "precision_mean", "recall_mean", "auroc_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
