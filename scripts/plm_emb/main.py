#!/usr/bin/env python3
"""
Training pipeline: ESM-1v embeddings (+ optional SIFT score) -> ML classifiers.

Usage: edit the paths and options below, then run: python main.py
"""
import logging
import subprocess
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from dataset import Prediction, Protein, Variant
from emb_extract import EmbExtract
from features import Feature
from train import Train
from codecarbon import EmissionsTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SPLITS_DIR = Path("../../data/splits_by_homology")  # change this to your path (output of homology_controlled_splits.py)
ESM_MODEL = Path("../../models/esm1v_t33_650M_UR90S_1.pt")  # change this to your path (downloaded ESM-1v weights)
ESM_SCRIPT = Path("extract.py")  # change this to your path
EMB_DIR = Path("../../results/embeddings")  # change this to your path
SIFT_BIN = Path("sift4g")  # change this to your path (sift4g binary)
SIFT_DB = Path("../../databases/uniref90_viridiplantae.fasta")  # change this to your path (SIFT protein database)
SIFT_DIR = Path("../../results/sift")  # change this to your path
SIFT_THREADS = 8  # change this to your number of CPUs
OUTDIR = Path("../../results/training")  # change this to your path

EMBEDDING_TYPE = "mut-site"  # full-mean | local-mean | mut-site
DELTA = False  # True: mutated - wild-type embedding
USE_SIFT = False  # True: append SIFT score to the embedding
WINDOW_SIZE = 13  # local-mean window (residues)
N_SPLITS = 5  # number of split_<n> folders


# Read proteins (seq_id + sequence) and their mutations from FASTA and .subst files
def read_proteins(fasta, subst_dir):
    proteins = []
    for record in SeqIO.parse(str(fasta), "fasta"):
        subst = Path(subst_dir) / f"{record.id}.subst"
        codes = pd.read_csv(subst)["mutant"].astype(str).tolist() if subst.exists() else []
        proteins.append(
            Protein(
                header=record.id,
                sequence=str(record.seq),
                variants=[Variant(code=c, protein_header=record.id) for c in codes],
            )
        )
    return proteins


# CodeCarbon tracker with the same settings as train.py
def tracker(output_dir, project_name, output_file):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return EmissionsTracker(
        output_dir=str(output_dir),
        project_name=project_name,
        output_file=output_file,
        log_level="error",
    )


# Per-token ESM-1v embeddings of mutated (and wild-type) sequences
def extract_embeddings(proteins, model, script, outdir, delta=False):
    outdir = Path(outdir)
    extractor = EmbExtract(
        proteins=proteins,
        method_params={
            "model_embedding": model,
            "script_embedding": script,
            "delta": delta,
        },
        output_dir=outdir,
        shared_dir=outdir,
    )
    return extractor.run()


# Run SIFT (skipped if predictions already exist) and store each score in var.predictions["sift"]
def run_sift(proteins, sift_bin, fasta, database, subst_dir, outdir, threads=1, evalue="0.0001"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not any(outdir.glob("*.SIFTprediction")):
        cmd = [
            str(sift_bin),
            "-q", str(fasta),
            "-d", str(database),
            "--subst", str(subst_dir),
            "--evalue", str(evalue),
            "--out", str(outdir),
            "--outfmt", "bm9",
            "--threads", str(threads),
            "--sub-results",
        ]
        with (outdir / "sift.log").open("w", encoding="utf-8") as log:
            subprocess.run(cmd, stdout=log, stderr=log, text=True, check=True)

    n_missing = 0
    for protein in proteins:
        # One <seq_id>.SIFTprediction file per protein: code, class, score, ...
        scores = {}
        pred_file = outdir / f"{protein.header}.SIFTprediction"
        if pred_file.exists():
            for line in pred_file.open(encoding="utf-8"):
                parts = line.split()
                if len(parts) < 6 or line.startswith("WARNING!"):
                    continue
                try:
                    scores[parts[0]] = (parts[1].upper(), float(parts[2]))
                except ValueError:
                    continue

        for var in protein.variants:
            label, score = scores.get(var.code, ("unknown", None))
            if score is None:
                n_missing += 1
            var.predictions["sift"] = Prediction(method="sift", score=score, label=label)

    print(f"[SIFT] variants without score: {n_missing}")


# X, y and metadata for one split table (train.tsv or test.tsv)
def build_features(proteins, data_tsv, emb_dir, trim_info, outdir,
                   embedding_type, delta=False, use_sift=False, window_size=13):
    feature = Feature(
        proteins=proteins,
        data_tsv=data_tsv,
        method_params={
            "emb_dir": emb_dir,
            "trim_info": trim_info,
            "embedding_type": embedding_type,
            "delta": delta,
            "use_sift": use_sift,
            "window_size": window_size,
        },
        output_dir=outdir,
    )
    return feature.run()


# Grid search with family-grouped CV on train, then evaluation of the best model on test
# (CodeCarbon emissions of training and testing are written by train.py)
def train_evaluate(train_features_path, test_features_path, outdir):
    trainer = Train(
        method_params={
            "train_features_path": train_features_path,
            "test_features_path": test_features_path,
        },
        output_dir=outdir,
    )
    return trainer.run()


def main():
    run_name = EMBEDDING_TYPE + ("_delta" if DELTA else "") + ("_sift" if USE_SIFT else "")

    for n in range(1, N_SPLITS + 1):
        split_dir = SPLITS_DIR / f"split_{n}"
        out_split = OUTDIR / run_name / f"split_{n}"

        features = {}
        for part in ("train", "test"):
            # Train and test inputs of this split (extract <part>_inputs/subst_files.tar.gz first)
            inputs = split_dir / f"{part}_inputs"
            out_part = out_split / part
            proteins = read_proteins(inputs / "sequences.fasta", inputs / "subst_files")

            if USE_SIFT:
                with tracker(out_part, "sift", "emissions_sift.csv"):
                    run_sift(proteins, SIFT_BIN, inputs / "sequences.fasta", SIFT_DB,
                             inputs / "subst_files", SIFT_DIR / f"split_{n}" / part, threads=SIFT_THREADS)

            with tracker(out_part, "embeddings", "emissions_embeddings.csv"):
                # Existing .pt files are skipped, so re-running only computes what is missing
                emb_res = extract_embeddings(proteins, ESM_MODEL, ESM_SCRIPT, EMB_DIR, delta=DELTA)

            with tracker(out_part, "features", "emissions_features.csv"):
                features[part] = build_features(
                    proteins,
                    split_dir / f"{part}.tsv",
                    EMB_DIR,
                    emb_res["trim_info"],
                    out_part,
                    EMBEDDING_TYPE,
                    delta=DELTA,
                    use_sift=USE_SIFT,
                    window_size=WINDOW_SIZE,
                )

        train_evaluate(
            features["train"]["features_path"],
            features["test"]["features_path"],
            out_split,
        )


if __name__ == "__main__":
    main()
