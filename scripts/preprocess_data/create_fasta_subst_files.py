#!/usr/bin/env python3
"""
Create a FASTA file (one sequence per seq_id) and one .subst file per mutation_id.

Input:  TSV with seq_id, mutation_id, Sequence, AA_from, AA_loci and AA_to columns.
Output: <outdir>/sequences.fasta
        <outdir>/subst_files/<mutation_id>.subst  (one substitution, e.g. E183D)

Usage: python create_fasta_subst_files.py -i input.tsv -o outdir
"""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Create FASTA and .subst files from a mutation table."
    )
    parser.add_argument("-i", "--input", required=True, help="Input TSV file")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")

    subst_dir = os.path.join(args.outdir, "subst_files")
    os.makedirs(subst_dir, exist_ok=True)

    # One FASTA entry per seq_id
    seqs = df[["seq_id", "Sequence"]].drop_duplicates("seq_id")
    fasta_file = os.path.join(args.outdir, "sequences.fasta")
    with open(fasta_file, "w") as f:
        for seq_id, seq in zip(seqs["seq_id"], seqs["Sequence"]):
            f.write(f">{seq_id}\n{seq}\n")

    # One .subst file per mutation_id
    muts = df.drop_duplicates("mutation_id")
    for _, row in muts.iterrows():
        subst = f"{row['AA_from']}{row['AA_loci']}{row['AA_to']}"
        with open(os.path.join(subst_dir, f"{row['mutation_id']}.subst"), "w") as f:
            f.write(f"{subst}\n")

    print(f"Sequences written: {len(seqs):,} -> {fasta_file}")
    print(f"Subst files written: {len(muts):,} -> {subst_dir}")


if __name__ == "__main__":
    main()
