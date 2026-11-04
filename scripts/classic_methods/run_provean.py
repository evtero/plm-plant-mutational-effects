#!/usr/bin/env python3
"""
Run PROVEAN on all proteins and their substitutions.

For each protein, a temporary FASTA and a temporary substitution file (without the
"mutant" header, stop and synonymous variants) are passed to provean.sh.

Output: one <seq_id>.provean.txt file per protein in OUTDIR and CodeCarbon
emissions (emissions_provean.csv). Existing output files are not recomputed.

Usage: edit the paths below, then run from this folder: python run_provean.py
"""
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from Bio import SeqIO
from codecarbon import EmissionsTracker

PROVEAN_BIN = Path("provean.sh")  # change this to your path (PROVEAN script)
FASTA = Path("../../data/sequences.fasta")  # change this to your path
SUBST_DIR = Path("../../data/subst_files")  # change this to your path (extract data/subst_files.tar.gz first)
OUTDIR = Path("../../results/provean")  # change this to your path
THREADS = 1  # threads per PROVEAN run
WORKERS = 8  # proteins run in parallel; change this to your number of CPUs


# Run PROVEAN for one protein; returns its status
def run_one_protein(seq_id, sequence):
    subst_file = SUBST_DIR / f"{seq_id}.subst"
    if not subst_file.exists():
        return "missing_subst"

    out_file = OUTDIR / f"{seq_id}.provean.txt"
    if out_file.exists() and out_file.stat().st_size > 0:
        return "cached"

    # Keep only missense variants (no header, stop or synonymous)
    variants = []
    for line in subst_file.open(encoding="utf-8"):
        line = line.strip()
        if not line or line.lower() == "mutant" or "*" in line or line[0] == line[-1]:
            continue
        variants.append(line)

    if not variants:
        return "missing_subst"

    fasta = OUTDIR / f"{seq_id}.tmp.fasta"
    tmp_subst = OUTDIR / f"{seq_id}.tmp.subst"

    try:
        fasta.write_text(f">{seq_id}\n{sequence.rstrip('*')}\n", encoding="utf-8")
        tmp_subst.write_text("\n".join(variants) + "\n", encoding="utf-8")

        cmd = [
            str(PROVEAN_BIN),
            "-q", str(fasta),
            "-v", str(tmp_subst),
            "--num_threads", str(THREADS),
        ]
        with out_file.open("w", encoding="utf-8") as out:
            subprocess.run(cmd, stdout=out, stderr=out, text=True, check=True)
        return "ran"
    finally:
        fasta.unlink(missing_ok=True)
        tmp_subst.unlink(missing_ok=True)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    proteins = [(r.id, str(r.seq)) for r in SeqIO.parse(str(FASTA), "fasta")]
    counts = {"ran": 0, "cached": 0, "missing_subst": 0}

    tracker = EmissionsTracker(
        output_dir=str(OUTDIR),
        project_name="provean",
        output_file="emissions_provean.csv",
        log_level="error",
    )
    tracker.start()
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(run_one_protein, seq_id, seq): seq_id
                for seq_id, seq in proteins
            }
            for i, future in enumerate(as_completed(futures), start=1):
                counts[future.result()] += 1
                if i % 100 == 0 or i == len(proteins):
                    print(f"[PROVEAN] {i}/{len(proteins)} proteins")
    finally:
        tracker.stop()

    print(f"[PROVEAN] done | {counts}")


if __name__ == "__main__":
    main()
