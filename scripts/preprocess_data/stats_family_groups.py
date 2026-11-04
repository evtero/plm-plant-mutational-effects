#!/usr/bin/env python3
# Compute cluster statistics from a TSV with metadata

import argparse
import ast
import csv
import os
from collections import Counter, defaultdict

ANNOTATION_COLS_DEFAULT = [
    "interpro_family_names",
    "interpro_domain_names",
    "interpro_homologous_superfamily_names",
    "interpro_conserved_site_names",
    "interpro_repeat_names",
    "interpro_active_site_names",
    "interpro_binding_site_names",
    "interpro_ptm_names",
    "go_F_terms",
    "go_P_terms",
    "go_C_terms",
]

def parse_terms(value):
    if not value or str(value).strip() in ("", "[]", "NA", "nan", "None"):
        return []
    
    # '|' separator, as produced by add_goterms_interpro_metadata.py
    if "|" in str(value):
        return [x.strip() for x in str(value).split("|") if x.strip()]
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return [str(parsed).strip()] if str(parsed).strip() else []
    except Exception:
        clean = value.replace("[", "").replace("]", "").replace('"', "").replace("'", "")
        return [x.strip() for x in clean.split(",") if x.strip()]

def read_clusters(input_file, annotation_cols, group_col):
    clusters = defaultdict(list)
    seen = set()
    stats = defaultdict(int)

    with open(input_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats["total_rows"] += 1
            seq_id = row.get("seq_id", "").strip()

            if not seq_id:
                stats["empty_seq_id"] += 1
                continue
            if seq_id in seen:
                stats["duplicate_seq_rows"] += 1
                continue

            seen.add(seq_id)
            cluster_id = row.get(group_col, "").strip()

            if cluster_id in ("", "NA", "nan", "None"):
                stats["missing_cluster"] += 1
                continue

            clusters[cluster_id].append(row)

    stats["unique_seq_id_retained"] = len(seen)
    stats["clusters_detected"] = len(clusters)
    stats["assigned_sequences"] = sum(len(v) for v in clusters.values())
    return clusters, dict(stats)


def _annotation_availability(clusters, annotation_cols):
    rows = []
    for col in annotation_cols:
        with_ann = without_ann = n_terms = 0
        for seqs in clusters.values():
            for row in seqs:
                terms = parse_terms(row.get(col, ""))
                if terms:
                    with_ann += 1
                    n_terms += len(set(terms))
                else:
                    without_ann += 1
        rows.append({
            "annotation_type": col,
            "with_annotation": with_ann,
            "without_annotation": without_ann,
            "unique_seq_annotations_counted": n_terms,
        })
    return rows


def write_main_cluster_stats(clusters, annotation_cols, outdir):
    summary_rows, details_rows, cluster_rows = [], [], []
    total_seqs = sum(len(v) for v in clusters.values())

    for col in annotation_cols:
        clusters_with_ann = 0
        total_distinct = 0
        max_purities = []

        for cluster_id, rows in clusters.items():
            size = len(rows)
            term_counter = Counter()
            annotated = set()

            for row in rows:
                terms = set(parse_terms(row.get(col, "")))
                if terms:
                    annotated.add(row["seq_id"])
                term_counter.update(terms)

            if not term_counter:
                continue

            clusters_with_ann += 1
            total_distinct += len(term_counter)
            n_annotated = len(annotated)
            dominant, dom_count = term_counter.most_common(1)[0]
            purity_all = dom_count / size
            purity_ann = dom_count / n_annotated if n_annotated else 0
            max_purities.append(purity_all)

            cluster_rows.append({
                "cluster_id": cluster_id,
                "cluster_size": size,
                "annotation_type": col,
                "n_distinct_annotations": len(term_counter),
                "annotated_seq_count": n_annotated,
                "annotation_coverage": n_annotated / size,
                "dominant_annotation": dominant,
                "dominant_count": dom_count,
                "dominant_purity_all_cluster": purity_all,
                "dominant_purity_annotated_only": purity_ann,
            })

            for term, count in term_counter.most_common():
                details_rows.append({
                    "cluster_id": cluster_id,
                    "cluster_size": size,
                    "annotation_type": col,
                    "annotation": term,
                    "count": count,
                    "purity_all_cluster": count / size,
                    "annotated_seq_count": n_annotated,
                    "purity_annotated_only": count / n_annotated if n_annotated else 0,
                })

        summary_rows.append({
            "annotation_type": col,
            "n_clusters": len(clusters),
            "n_sequences": total_seqs,
            "clusters_with_annotation": clusters_with_ann,
            "total_distinct_cluster_terms": total_distinct,
            "mean_max_purity": sum(max_purities) / len(max_purities) if max_purities else 0,
        })

    files = {
        "summary": (os.path.join(outdir, "summary_by_annotation_type.tsv"),
                    ["annotation_type", "n_clusters", "n_sequences",
                     "clusters_with_annotation", "total_distinct_cluster_terms", "mean_max_purity"],
                    summary_rows),
        "details": (os.path.join(outdir, "all_cluster_annotations.tsv"),
                    ["cluster_id", "cluster_size", "annotation_type", "annotation",
                     "count", "purity_all_cluster", "annotated_seq_count", "purity_annotated_only"],
                    details_rows),
        "cluster": (os.path.join(outdir, "cluster_level_summary.tsv"),
                    ["cluster_id", "cluster_size", "annotation_type", "n_distinct_annotations",
                     "annotated_seq_count", "annotation_coverage", "dominant_annotation",
                     "dominant_count", "dominant_purity_all_cluster", "dominant_purity_annotated_only"],
                    cluster_rows),
    }

    for path, fieldnames, data in files.values():
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            w.writeheader()
            w.writerows(data)

    availability_rows = _annotation_availability(clusters, annotation_cols)
    return files["summary"][0], files["details"][0], files["cluster"][0], availability_rows


def write_dominant_purity_bins(cluster_file, outdir):
    output_file = os.path.join(outdir, "dominant_purity_bins.tsv")
    purities = defaultdict(list)

    with open(cluster_file, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                purities[row["annotation_type"]].append(float(row["dominant_purity_all_cluster"]))
            except ValueError:
                continue

    with open(output_file, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["annotation_type", "n_clusters", "pct_purity_1.0",
                    "pct_90_100", "pct_80_90", "pct_below_80"])

        for ann_type in sorted(purities):
            vals = purities[ann_type]
            n = len(vals)
            w.writerow([
                ann_type, n,
                f"{100 * sum(p == 1.0  for p in vals) / n:.2f}",
                f"{100 * sum(0.9 <= p < 1.0 for p in vals) / n:.2f}",
                f"{100 * sum(0.8 <= p < 0.9 for p in vals) / n:.2f}",
                f"{100 * sum(p < 0.8  for p in vals) / n:.2f}",
            ])

    return output_file


def write_annotation_dispersion(details_file, total_file, outdir, group_col):
    output_file = os.path.join(outdir, "annotation_dispersion_species.tsv")

    annotation_clusters = defaultdict(set)
    with open(details_file, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            annotation_clusters[(row["annotation_type"], row["annotation"])].add(row["cluster_id"])

    cluster_species = defaultdict(set)
    seen = set()
    with open(total_file, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            seq_id = row["seq_id"].strip()
            if not seq_id or seq_id in seen:
                continue
            seen.add(seq_id)
            cluster = row[group_col].strip()
            # tax_species es el nombre de columna que genera el enriquecedor
            species = row.get("tax_species", row.get("species", "")).strip()
            if cluster and species:
                cluster_species[cluster].add(species)

    annotation_types = sorted({k[0] for k in annotation_clusters})
    BINS = ["1", "2-5", "6-10", ">10"]

    def bin_label(n):
        if n == 1: return "1"
        if n <= 5: return "2-5"
        if n <= 10: return "6-10"
        return ">10"

    with open(output_file, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["annotation_type", "cluster_bin", "n_annotations",
                    "pct_annotations", "mean_species", "min_species", "max_species"])

        for ann_type in annotation_types:
            ann_this_type = {k: v for k, v in annotation_clusters.items() if k[0] == ann_type}
            n_total = len(ann_this_type)
            if not n_total:
                continue

            bins = defaultdict(list)
            for key, clusters in ann_this_type.items():
                bins[bin_label(len(clusters))].append(key)

            for label in BINS:
                keys = bins.get(label, [])
                species_counts = [
                    len({sp for c in annotation_clusters[k] for sp in cluster_species.get(c, set())})
                    for k in keys
                ]
                if species_counts:
                    mean_s = f"{sum(species_counts) / len(species_counts):.2f}"
                    min_s, max_s = min(species_counts), max(species_counts)
                else:
                    mean_s = min_s = max_s = "NA"

                w.writerow([ann_type, label, len(keys),
                            f"{100 * len(keys) / n_total:.2f}",
                            mean_s, min_s, max_s])

    return output_file


def write_run_summary(outdir, input_file, group_col, stats, availability_rows, output_files):
    run_summary = os.path.join(outdir, "run_summary.log")

    with open(run_summary, "w") as f:
        f.write(f"[INFO] Input file: {input_file}\n")
        f.write(f"[INFO] Output directory: {outdir}\n")
        f.write(f"[INFO] Group column: {group_col}\n\n")

        f.write("[CHECK]\n")
        for key, label in [
            ("total_rows",              "Input rows                      "),
            ("unique_seq_id_retained",  "Unique seq_id retained          "),
            ("duplicate_seq_rows",      "Duplicate rows skipped by seq_id"),
            ("empty_seq_id",            "Rows skipped with empty seq_id  "),
            ("missing_cluster",         "Unique seq_id without cluster   "),
            ("clusters_detected",       "Clusters detected               "),
            ("assigned_sequences",      "Sequences assigned to clusters  "),
        ]:
            f.write(f"{label}: {stats.get(key, 0)}\n")

        f.write("\n[CHECK] Annotation availability by column\n")
        for row in availability_rows:
            f.write(f"{row['annotation_type']}: with_annotation={row['with_annotation']}, "
                    f"without_annotation={row['without_annotation']}, "
                    f"unique_seq_annotations_counted={row['unique_seq_annotations_counted']}\n")

        f.write("\n[OUTPUT]\n")
        for label, path in output_files.items():
            f.write(f"{label}: {path}\n")

    return run_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("total_ids_metadata_clusters",
                        help="TSV con seq_id, columna de grupo, species y columnas de anotacion")
    parser.add_argument("-o", "--outdir", default=None)
    parser.add_argument("--annotation-cols", nargs="+", default=ANNOTATION_COLS_DEFAULT)
    parser.add_argument("--group-col", default="cluster_id")
    args = parser.parse_args()

    input_file = args.total_ids_metadata_clusters
    group_col = args.group_col
    outdir = args.outdir or f"{os.path.splitext(os.path.basename(input_file))[0]}_cluster_stats"
    os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] Input: {input_file} | Output: {outdir} | Group col: {group_col}")

    clusters, stats = read_clusters(input_file, args.annotation_cols, group_col)

    print("\n[CHECK]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    summary_file, details_file, cluster_file, availability_rows = write_main_cluster_stats(
        clusters, args.annotation_cols, outdir
    )

    dominant_bins_file = write_dominant_purity_bins(cluster_file, outdir)
    dispersion_file = write_annotation_dispersion(details_file, input_file, outdir, group_col)

    output_files = {
        "Summary table": summary_file,
        "All annotations table": details_file,
        "Cluster-level table": cluster_file,
        "Dominant purity bins": dominant_bins_file,
        "Annotation dispersion/species diversity": dispersion_file,
    }

    run_summary = write_run_summary(
        outdir=outdir,
        input_file=input_file,
        group_col=group_col,
        stats=stats,
        availability_rows=availability_rows,
        output_files={**output_files, "Run summary": os.path.join(outdir, "run_summary.log")},
    )

    print("\n[OUTPUT]")
    for label, path in output_files.items():
        print(f"  {label}: {path}")
    print(f"  Run summary: {run_summary}")


if __name__ == "__main__":
    main()
