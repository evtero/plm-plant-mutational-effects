#!/usr/bin/env python3
"""
Protein family-level train/test splits and residual homology check (Supplementary Methods S1).

Steps:
  1. Add seq_id and mutation_id to the enriched dataset.
  2. Write FASTA (one entry per seq_id) and .subst files (one per seq_id).
  3. All-vs-all BLASTP.
  4. Infer protein families following Washburn et al. (2019):
     https://bitbucket.org/bucklerlab/p_strength_prediction
       - src/main/R/Generate_paralogous_gene_pairs.R   (pair filtering)
       - src/main/python/construct_gene_families.py    (graph + MCL clustering)
     Changes: protein-level (seq_id) instead of gene-level; proteins without
     hits are added as nodes so they form their own family.
  5. Family statistics (stats_family_groups.py).
  6. Five random family-level 80/20 train/test splits (seeds 1-5).
  7. Residual homology of each test set against its train set (MMseqs2).

Input:  enriched dataset TSV (output of add_goterms_interpro_metadata.py).
Usage:  edit the paths below, then run: python split_by_homology.py

Outputs (in OUTDIR):
  total_with_seqids.tsv, sequences.fasta, subst_files/   Full dataset with IDs and inputs
  blast_db/, all_vs_all.blast.tsv                        BLASTP database and hits
  result_V3_to_V3_filtered, pairs_failed_filter.tsv      Filtered / removed pairs
  gene_families{11,15,20}.npy                            MCL families (1.1 used downstream)
  total_metadata_seqids_family.tsv                       Dataset with family_id
  family_cluster_stats/, family_cluster_stats.log        Family statistics
  split_<seed>/train.tsv, test.tsv                       Train/test split
  split_<seed>/train_inputs/, test_inputs/               FASTA and .subst files per split
  split_<seed>/homology_check/                           Test vs train MMseqs2 hits, best hits,
                                                         identity bins (Table S1) and summary.log
"""

from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import networkx as nx
import markov_clustering as mc
import scipy.sparse as sp

INPUT_DATASET = Path("../../data/total_benchmark_data_enriched.tsv")  # change this to your path
OUTDIR = Path("../../repo_test/splits_by_homology")  # change this to your path
SCRIPT_INPUTS = Path("create_fasta_subst_files.py")  # change this to your path
SCRIPT_STATS = Path("stats_family_groups.py")  # change this to your path
BLAST_THREADS = 16  # change this to your number of CPUs


def effect_percentages(df):
    counts = df["Effect"].astype(str).value_counts(dropna=False)
    total = len(df)

    functional = counts.get("functional", 0)
    neutral = counts.get("neutral", 0)

    return (
        f"functional={functional} ({100 * functional / total:.2f}%), "
        f"neutral={neutral} ({100 * neutral / total:.2f}%)"
    )


def top_values_sentence(df, col, n=10):
    if col not in df.columns:
        return f"{col}: column not found"

    values = df[col].dropna().astype(str)
    if values.empty:
        return f"{col}: no data"

    top = values.value_counts().head(n)

    return f"{col}: " + ", ".join(
        f"{name} ({count}, {100 * count / len(df):.2f}%)"
        for name, count in top.items()
    )


def print_split_composition(train, test):
    print("\nComposition by Effect")
    print(f"Train Effect: {effect_percentages(train)}")
    print(f"Test Effect : {effect_percentages(test)}")

    for col in ["tax_species", "tax_family", "tax_class", "tax_order"]:
        print(f"\nTop 10 {col}")
        print(f"Train {top_values_sentence(train, col)}")
        print(f"Test  {top_values_sentence(test, col)}")


# Write FASTA and .subst files with create_fasta_subst_files.py
def write_inputs(tsv, outdir):
    subprocess.run(
        ["python", str(SCRIPT_INPUTS), "-i", str(tsv), "-o", str(outdir)],
        check=True,
    )
    return outdir / "sequences.fasta", outdir / "subst_files"


def main():
    outdir = OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_DATASET, sep="\t")

    # Check required columns
    required_cols = ["Sequence", "AA_from", "AA_loci", "AA_to"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # One seq_id per unique sequence (rows with the same sequence share seq_id)
    seq_codes = pd.factorize(df["Sequence"])[0] + 1
    df["seq_id"] = "seqid_" + pd.Series(seq_codes, index=df.index).astype(str).str.zfill(4)

    # Mutation code, e.g. A123C
    df["mutation_code"] = (
        df["AA_from"].astype(str)
        + df["AA_loci"].astype(str)
        + df["AA_to"].astype(str)
    )

    # Unique sequence-mutation ID
    df["mutation_id"] = df["seq_id"] + "-" + df["mutation_code"]

    # Warn about duplicated sequence-mutation rows
    duplicated = df.duplicated(subset=["mutation_id"], keep=False)
    if duplicated.any():
        n_dup_rows = int(duplicated.sum())
        n_dup_ids = int(df.loc[duplicated, "mutation_id"].nunique())
        print(
            f"WARNING: {n_dup_rows} duplicated rows corresponding to "
            f"{n_dup_ids} repeated mutation_id."
        )

    tsv_out = outdir / "total_with_seqids.tsv"
    df.to_csv(tsv_out, sep="\t", index=False)

    fasta_out, subst_dir = write_inputs(tsv_out, outdir)

    print(f"Total rows: {len(df):,}")
    print(f"Unique sequences: {df['seq_id'].nunique():,}")
    print(f"Unique sequence-mutations: {df['mutation_id'].nunique():,}")
    print(f"TSV with IDs: {tsv_out}")
    print(f"FASTA: {fasta_out}")
    print(f"Subst files: {subst_dir}")

    # BLAST database
    blast_db = outdir / "blast_db" / "total_sequences"
    blast_db.parent.mkdir(parents=True, exist_ok=True)

    print("Creating BLAST database...")

    subprocess.run(
        [
            "makeblastdb",
            "-in",
            str(fasta_out),
            "-dbtype",
            "prot",
            "-parse_seqids",
            "-out",
            str(blast_db),
        ],
        check=True,
    )

    # All-vs-all BLASTP
    blast_out = outdir / "all_vs_all.blast.tsv"

    print("Running all-vs-all BLASTP...")

    subprocess.run(
        [
            "blastp",
            "-query",
            str(fasta_out),
            "-db",
            str(blast_db),
            "-out",
            str(blast_out),
            "-outfmt",
            "6",
            "-evalue",
            "0.001",
            "-num_threads",
            str(BLAST_THREADS),
        ],
        check=True,
    )

    print(f"BLAST: {blast_out}")

    # Infer protein families with MCL following Washburn et al.
    # No collapsing to genes: seq_id is used directly as graph node

    print("Inferring family_id with MCL...")

    blast_cols = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
    ]

    blast = pd.read_csv(
        blast_out,
        sep="\t",
        names=blast_cols,
        header=None,
    )

    # Rename columns as in Washburn et al.
    blast = blast.rename(columns={"qseqid": "query", "sseqid": "target"})

    # Filter as in Washburn et al.
    passed = blast[
        (blast["query"] != blast["target"])
        & (blast["evalue"] < 0.001)
        & (blast["bitscore"] > 50)
    ].copy()

    failed = blast[
        ~(
            (blast["query"] != blast["target"])
            & (blast["evalue"] < 0.001)
            & (blast["bitscore"] > 50)
        )
    ].copy()

    # Undirected pairs: A-B == B-A
    passed["qgene"] = passed[["query", "target"]].min(axis=1)
    passed["sgene"] = passed[["query", "target"]].max(axis=1)

    # Best bitscore per pair
    passed_best = (
        passed.sort_values("bitscore", ascending=False)
        .drop_duplicates(subset=["qgene", "sgene"], keep="first")
        .reset_index(drop=True)
    )

    # Save intermediate tables
    passed_best[["qgene", "sgene", "bitscore"]].to_csv(
        outdir / "result_V3_to_V3_filtered",
        sep="\t",
        index=False,
    )

    failed.to_csv(outdir / "pairs_failed_filter.tsv", sep="\t", index=False)

    print(f"Total alignments: {len(blast)}")
    print(f"Passed bitscore and evalue filter: {len(passed)}")
    print(f"Failed filter: {len(failed)}")
    print(f"Final unique pairs: {len(passed_best)}")

    # Weighted graph
    edges_with_weights = [
        (
            passed_best["qgene"][i],
            passed_best["sgene"][i],
            passed_best["bitscore"][i],
        )
        for i in range(len(passed_best))
    ]

    G = nx.Graph()
    G.add_weighted_edges_from(edges_with_weights)

    # Add all sequences so isolated ones also get a family
    all_seq_ids = sorted(df["seq_id"].astype(str).unique())
    G.add_nodes_from(all_seq_ids)

    print(f"Graph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")

    # Sparse matrix for markov_clustering
    nodes = list(G.nodes())

    matrix = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight")
    matrix = sp.csr_matrix(matrix)

    # MCL with different inflation values
    result11 = mc.run_mcl(matrix, inflation=1.1)
    result15 = mc.run_mcl(matrix, inflation=1.5)
    result20 = mc.run_mcl(matrix, inflation=2)

    clusters11 = mc.get_clusters(result11)
    clusters15 = mc.get_clusters(result15)
    clusters20 = mc.get_clusters(result20)

    # Node indices to sequence IDs
    gene_families11 = [[nodes[element] for element in cluster] for cluster in clusters11]
    gene_families15 = [[nodes[element] for element in cluster] for cluster in clusters15]
    gene_families20 = [[nodes[element] for element in cluster] for cluster in clusters20]

    # Save families
    np.save(outdir / "gene_families11.npy", np.array(gene_families11, dtype=object))
    np.save(outdir / "gene_families15.npy", np.array(gene_families15, dtype=object))
    np.save(outdir / "gene_families20.npy", np.array(gene_families20, dtype=object))

    # Inflation 1.1 used downstream, as in Washburn et al.
    seq_to_family = {}

    for i, family in enumerate(gene_families11, start=1):
        family_id = f"family_{i:04d}"
        for seq_id in family:
            seq_to_family[seq_id] = family_id

    df["family_id"] = df["seq_id"].astype(str).map(seq_to_family)

    if df["family_id"].isna().any():
        raise RuntimeError("Some sequences have no family_id.")

    tsv_family_out = outdir / "total_metadata_seqids_family.tsv"
    df.to_csv(tsv_family_out, sep="\t", index=False)

    print(f"Families inflation 1.1: {len(gene_families11)}")
    print(f"Families inflation 1.5: {len(gene_families15)}")
    print(f"Families inflation 2.0: {len(gene_families20)}")
    print(f"Clustered sequences inflation 1.1: {sum(len(f) for f in gene_families11)}")
    print(f"Largest family size inflation 1.1: {max(len(f) for f in gene_families11)}")
    print(f"TSV with family_id: {tsv_family_out}")

    # Family statistics before splitting
    cluster_stats_outdir = outdir / "family_cluster_stats"
    cluster_stats_log = outdir / "family_cluster_stats.log"

    print("Computing family_id statistics...")

    with cluster_stats_log.open("w") as log:
        subprocess.run(
            [
                "python",
                str(SCRIPT_STATS),
                str(tsv_family_out),
                "--group-col",
                "family_id",
                "-o",
                str(cluster_stats_outdir),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    print(f"Cluster stats: {cluster_stats_outdir}")
    print(f"Cluster stats log: {cluster_stats_log}")

    print("\nGenerating train/test splits...")

    test_size = 0.2

    for seed in range(1, 6):

        families = df["family_id"].unique().copy()

        rng = np.random.RandomState(seed)
        rng.shuffle(families)

        n_test = max(1, int(len(families) * test_size))

        test_families = set(families[:n_test])

        train = df[~df["family_id"].isin(test_families)].copy()
        test = df[df["family_id"].isin(test_families)].copy()

        train["split_group"] = 1
        test["split_group"] = 2

        split_dir = outdir / f"split_{seed}"
        split_dir.mkdir(exist_ok=True)

        train.to_csv(split_dir / "train.tsv", sep="\t", index=False)
        test.to_csv(split_dir / "test.tsv", sep="\t", index=False)

        overlap = (
            set(train["family_id"])
            & set(test["family_id"])
        )

        print(f"\nSplit {seed}")
        print(f"Train: {len(train)}")
        print(f"Test : {len(test)}")
        print(f"Train families: {train['family_id'].nunique()}")
        print(f"Test families : {test['family_id'].nunique()}")
        print(f"Family overlap: {len(overlap)}")

        if len(overlap) != 0:
            raise RuntimeError(f"Split {seed}: family_id present in both train and test")

        print_split_composition(train, test)

        # FASTA/subst inputs for train and test
        for split_name, split_tsv in [
            ("train_inputs", split_dir / "train.tsv"),
            ("test_inputs", split_dir / "test.tsv"),
        ]:
            split_inputs_dir = split_dir / split_name
            print(f"Generating inputs for {split_inputs_dir}...")
            write_inputs(split_tsv, split_inputs_dir)

        # ------------------------------------------------------------
        # Residual train/test homology with MMseqs2
        # ------------------------------------------------------------

        homology_dir = split_dir / "homology_check"
        homology_dir.mkdir(exist_ok=True)

        mmseqs_out = homology_dir / "test_vs_train.tsv"

        print("Checking train/test homology with MMseqs2...")

        subprocess.run(
            [
                "mmseqs",
                "easy-search",
                str(split_dir / "test_inputs" / "sequences.fasta"),
                str(split_dir / "train_inputs" / "sequences.fasta"),
                str(mmseqs_out),
                str(homology_dir / "tmp"),
                "--format-output",
                "query,target,pident,alnlen,qlen,tlen,qcov,tcov,evalue,bits",
            ],
            check=True,
        )

        # Read MMseqs2 results
        cols = [
            "query",
            "target",
            "pident",
            "alnlen",
            "qlen",
            "tlen",
            "qcov",
            "tcov",
            "evalue",
            "bits",
        ]

        hits = pd.read_csv(mmseqs_out, sep="\t", names=cols)

        print("\nMMseqs hits:")
        print(hits.head(2))

        # Total test proteins
        total_test = sum(
            line.startswith(">")
            for line in open(split_dir / "test_inputs" / "sequences.fasta")
        )

        # Test proteins with at least one hit
        proteins_with_hits = hits["query"].nunique()

        # Best hit by bitscore
        best_hits = (
            hits.sort_values(["query", "bits"], ascending=[True, False])
            .drop_duplicates("query")
        )

        print("\nBest hit by bitscore:")
        print(best_hits.head(2))

        # Test proteins whose best hit has identity > 40
        best_hits_gt40 = best_hits[best_hits["pident"] > 40]

        pct_gt40_over_hits = (
            100 * len(best_hits_gt40) / len(best_hits)
            if len(best_hits) else 0
        )

        pct_gt40_over_total_test = (
            100 * len(best_hits_gt40) / total_test
            if total_test else 0
        )

        print(
            f"Best-hit pident > 40: {len(best_hits_gt40)} "
            f"({pct_gt40_over_hits:.2f}% of test with hits; "
            f"{pct_gt40_over_total_test:.2f}% of total test)"
        )

        best_hits.to_csv(
            homology_dir / "best_bitscore.tsv",
            sep="\t",
            index=False,
        )

        # Identity distribution of best hits
        bins = pd.cut(
            best_hits["pident"],
            bins=[0, 20, 30, 40, 50, 70, 100],
            labels=["<20", "20-30", "30-40", "40-50", "50-70", ">=70"],
            include_lowest=True,
        )

        identity_bins = (
            bins.value_counts(sort=False)
            .rename_axis("identity_bin")
            .reset_index(name="count")
        )

        # Percentage over test proteins with at least one hit
        identity_bins["percent_over_hits"] = (
            100 * identity_bins["count"] / len(best_hits)
            if len(best_hits) else 0
        )

        # Percentage over all test proteins
        identity_bins["percent_over_total_test"] = (
            100 * identity_bins["count"] / total_test
            if total_test else 0
        )

        identity_bins.to_csv(
            homology_dir / "identity_bins_bitscore.tsv",
            sep="\t",
            index=False,
        )

        print("\nIdentity distribution:")
        print(identity_bins.to_string(index=False))

        # High identity and high coverage
        high_identity = best_hits[
            (best_hits["pident"] >= 50)
            & (best_hits["qcov"] >= 0.8)
            & (best_hits["tcov"] >= 0.8)
        ]

        print("\nHigh identity and high coverage:")
        print(high_identity.head(2))

        high_identity.to_csv(
            homology_dir / "high_identity_high_cov.tsv",
            sep="\t",
            index=False,
        )

        # High identity but low coverage
        partial_hits = best_hits[
            (best_hits["pident"] >= 70)
            & (
                (best_hits["qcov"] < 0.5)
                | (best_hits["tcov"] < 0.5)
            )
        ]

        partial_hits.to_csv(
            homology_dir / "high_identity_low_cov.tsv",
            sep="\t",
            index=False,
        )

        print("\nHigh identity and low coverage:")
        print(partial_hits.head(2))

        pct_with_hits = 100 * proteins_with_hits / total_test if total_test else 0
        pct_without_hits = 100 * (total_test - proteins_with_hits) / total_test if total_test else 0

        # Summary
        with open(homology_dir / "summary.log", "w") as f:
            f.write(f"Test proteins: {total_test}\n")
            f.write(f"Proteins with hits: {proteins_with_hits}\n")
            f.write(f"Proteins without hits: {total_test - proteins_with_hits}\n")
            f.write(f"Percent with hits: {pct_with_hits:.2f}\n")
            f.write(f"Percent without hits: {pct_without_hits:.2f}\n")
            f.write(f"Best-hit pident > 40: {len(best_hits_gt40)}\n")
            f.write(f"Best-hit pident > 40 over test with hits: {pct_gt40_over_hits:.2f}\n")
            f.write(f"Best-hit pident > 40 over total test: {pct_gt40_over_total_test:.2f}\n")


if __name__ == "__main__":
    main()
