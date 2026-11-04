#!/usr/bin/env python3
"""
Run SIFT (sift4g) on all proteins and their substitutions.

Output: one <seq_id>.SIFTprediction file per protein in OUTDIR, sift.log and
CodeCarbon emissions (emissions_sift.csv).

Usage: edit the paths below, then run from this folder: python run_sift.py
"""
import subprocess
from pathlib import Path

from codecarbon import EmissionsTracker

SIFT_BIN = Path("sift4g")  # change this to your path (sift4g binary)
FASTA = Path("../../data/sequences.fasta")  # change this to your path
SUBST_DIR = Path("../../data/subst_files")  # change this to your path (extract data/subst_files.tar.gz first)
SIFT_DB = Path("../../databases/<database>.fasta")  # change this to your path (SIFT protein database)
OUTDIR = Path("../../results/sift")  # change this to your path
EVALUE = "0.0001"
THREADS = 8  # change this to your number of CPUs


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(SIFT_BIN),
        "-q", str(FASTA),
        "-d", str(SIFT_DB),
        "--subst", str(SUBST_DIR),
        "--evalue", EVALUE,
        "--out", str(OUTDIR),
        "--outfmt", "bm9",
        "--threads", str(THREADS),
        "--sub-results",
    ]
    print("[SIFT] " + " ".join(cmd))

    tracker = EmissionsTracker(
        output_dir=str(OUTDIR),
        project_name="sift",
        output_file="emissions_sift.csv",
        log_level="error",
    )
    tracker.start()
    try:
        with (OUTDIR / "sift.log").open("w", encoding="utf-8") as log:
            subprocess.run(cmd, stdout=log, stderr=log, text=True, check=True)
    finally:
        tracker.stop()

    n_pred = len(list(OUTDIR.glob("*.SIFTprediction")))
    print(f"[SIFT] done | prediction files: {n_pred} | log: {OUTDIR / 'sift.log'}")


if __name__ == "__main__":
    main()
