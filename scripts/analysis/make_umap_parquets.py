#!/usr/bin/env python3
"""
Build the parquet inputs of the UMAP panels from the training features.

For one split, each features_<config>.pkl (train and test) is joined with the
split table (train.tsv / test.tsv) by mutation_id and saved as
<config>_<train|test>.parquet with a 'feature' column, plus manifest.tsv.

Usage: edit the paths below, then run: python make_umap_parquets.py
"""
import pickle
from pathlib import Path

import pandas as pd

SPLITS_DIR = Path("../../data/splits_by_homology")  # change this to your path
TRAINING_DIR = Path("../../results/training")  # change this to your path (OUTDIR of plm_emb/main.py)
OUTDIR = Path("../../results/parquet")  # change this to your path
SPLIT = 1

CONFIGS = [
    embedding_type + suffix
    for embedding_type in ("full-mean", "local-mean", "mut-site")
    for suffix in ("", "_delta", "_delta_sift", "_sift")
]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for part in ("train", "test"):
        table = pd.read_csv(SPLITS_DIR / f"split_{SPLIT}" / f"{part}.tsv", sep="\t")
        table = table.drop_duplicates("mutation_id")

        for config in CONFIGS:
            name = config.replace("-", "")  # e.g. mut-site_delta -> mutsite_delta
            features_dir = TRAINING_DIR / config / f"split_{SPLIT}" / part
            pkl_files = sorted(features_dir.glob("features_*.pkl"))
            if not pkl_files:
                print(f"[PARQUET] missing: {features_dir}/features_*.pkl")
                continue

            with pkl_files[0].open("rb") as f:
                obj = pickle.load(f)

            # Rows in the same order as X; split columns added by mutation_id
            meta = obj["meta"].reset_index(drop=True)
            extra_cols = ["mutation_id"] + [c for c in table.columns if c not in meta.columns]
            df = meta.merge(table[extra_cols], on="mutation_id", how="left")

            df["feature"] = [row.tolist() for row in obj["X"]]
            df["feature_dim"] = obj["X"].shape[1]
            df["y"] = obj["y"]
            df["model"] = name
            df["embedding_type"] = obj["embedding_type"]
            df["delta"] = obj["delta"]
            df["use_sift"] = obj["use_sift"]

            out = OUTDIR / f"{name}_{part}.parquet"
            df.to_parquet(out, index=False)
            manifest.append({"model": name, "split": part, "path": out.name})
            print(f"[PARQUET] {out.name}: {len(df)} rows, feature_dim={obj['X'].shape[1]}")

    pd.DataFrame(manifest).to_csv(OUTDIR / "manifest.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
