#!/usr/bin/env python3
"""
ESM-1v zero-shot scores for all proteins and their substitutions.

Proteins <= 1,022 aa are scored as a single window. Longer proteins are split into
overlapping 1,022-aa windows (50% overlap); each variant is scored in every window
that contains it and the window scores are combined with a weighted average that
down-weights positions near the window edges. Each window score is the mean over
the ESM-1v models.

Output (OUTDIR):
  <seq_id>/<seq_id>.<start>.<end>.{in,out}.subst   ESM-1v input/output per window
  esm1v.jobs.jsonl, esm1v.out, esm1v.err           Scoring jobs and logs
  esm1v_scores.tsv                                 Final score per variant
  rejected_variants.json                           Variants not scored and why
  emissions_esm1v.csv                              CodeCarbon estimates

Usage: edit the paths below, then run from this folder: python run_esm1v_zeroshot.py
"""
import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Bio import SeqIO
from codecarbon import EmissionsTracker

FASTA = Path("../../data/sequences.fasta")  # change this to your path
SUBST_DIR = Path("../../data/subst_files")  # change this to your path (extract data/subst_files.tar.gz first)
ESM_SCRIPT = Path("predict.py")  # change this to your path (fair-esm examples/variant-prediction/predict.py)
MODELS = [
    Path(f"../../models/esm1v_t33_650M_UR90S_{i}.pt") for i in range(1, 6)
]  # change this to your path (ESM-1v weights 1-5)
OUTDIR = Path("../../results/esm1v")  # change this to your path
SCORING = "wt-marginals"

WINDOW = 1022  # max length for ESM-1v
_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")  # e.g. A123C


# Read proteins (seq_id, sequence) and their variant codes from FASTA and .subst files
def read_proteins(fasta, subst_dir):
    proteins = []
    for record in SeqIO.parse(str(fasta), "fasta"):
        subst = Path(subst_dir) / f"{record.id}.subst"
        codes = []
        if subst.exists():
            for line in subst.open(encoding="utf-8"):
                line = line.strip()
                if line and line.lower() != "mutant":
                    codes.append(line)
        proteins.append((record.id, str(record.seq), codes))
    return proteins


def _parse_mut(code: str):
    m = _MUT_RE.match(code)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def _to_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


# Remove a terminal stop; proteins with internal stops are skipped
def _sanitize_sequence(sequence: str) -> Tuple[str, bool]:
    seq = sequence.strip().upper()
    if seq.endswith("*"):
        seq = seq[:-1]
    return seq, "*" in seq


# Overlapping windows (0-based [start0, end0)) covering a long protein
def _make_sliding_windows(seq_len: int, window: int = WINDOW, overlap: int = WINDOW // 2):
    if seq_len <= window:
        return [(0, seq_len)]

    step = window - overlap
    windows = []
    start = 0

    while start + window < seq_len:
        windows.append((start, start + window))
        start += step

    # Make sure the end of the protein is covered
    windows.append((seq_len - window, seq_len))
    return sorted(set(windows))


# Assign each variant position to every window that contains it
def _windows_for_variant_positions(seq_len: int, positions0: List[int]):
    parts = []
    for start0, end0 in _make_sliding_windows(seq_len):
        part_positions0 = [p for p in positions0 if start0 <= p < end0]
        if part_positions0:
            parts.append((start0, end0, part_positions0))
    return parts


# Weight of a position within a window: lower near the window edges
def _window_weight(local_pos1: int, window_len: int) -> float:
    if window_len < WINDOW:
        return 1.0

    i = local_pos1
    if i < 256:
        return 1.0 / (1.0 + math.exp(-(i - 128) / 16.0))
    if i < (WINDOW - 256):
        return 1.0
    return 1.0 / (1.0 + math.exp((i - (WINDOW - 128)) / 16.0))


# Scoring jobs (one per window) for one protein
def _jobs_for_protein(seq_id, sequence, codes, outdir, rejected):
    prot_outdir = outdir / seq_id
    prot_outdir.mkdir(parents=True, exist_ok=True)

    if not codes:
        return []

    sequence, internal_stop = _sanitize_sequence(sequence)
    if internal_stop:
        print(f"[ESM1v] protein skipped (internal stop): {seq_id}")
        return []

    seq_len = len(sequence)

    # Group variants by 0-based position
    pos0_to_codes = defaultdict(list)
    for code in codes:
        parsed = _parse_mut(code)
        if not parsed:
            rejected[seq_id][code] = {"reason": "invalid_variant_format"}
            continue

        ref, pos1, alt = parsed
        if pos1 < 1 or pos1 > seq_len:
            rejected[seq_id][code] = {
                "reason": "position_outside_sequence",
                "position": pos1,
                "sequence_length": seq_len,
            }
            continue

        pos0_to_codes[pos1 - 1].append(code)

    positions0 = sorted(pos0_to_codes)

    # Short proteins: one window; long proteins: overlapping windows
    if seq_len <= WINDOW:
        parts = [(0, seq_len, positions0)]
    else:
        parts = _windows_for_variant_positions(seq_len, positions0)

    jobs = []
    for start0, end0, part_positions0 in parts:
        part_len = end0 - start0
        if part_len <= 0:
            continue

        subst_in = prot_outdir / f"{seq_id}.{start0}.{end0}.in.subst"

        with subst_in.open("w", encoding="utf-8") as f:
            f.write("mutant\n")
            for pos0 in part_positions0:
                for code in pos0_to_codes[pos0]:
                    ref, pos1, alt = _parse_mut(code)

                    # Global 1-based position -> local 1-based position in the window
                    local_pos1 = pos1 - start0
                    if not 1 <= local_pos1 <= part_len:
                        raise ValueError(
                            f"[ESM1v] bad local_pos1={local_pos1} for {seq_id} {code} "
                            f"(window=[{start0}:{end0}], seq_len={seq_len})"
                        )

                    got = sequence[start0 + local_pos1 - 1]
                    if got != ref:
                        rejected[seq_id][code] = {
                            "reason": "wild_type_mismatch",
                            "expected_residue": ref,
                            "sequence_residue": got,
                        }
                        continue

                    f.write(f"{ref}{local_pos1}{alt}\n")

        out_subst = prot_outdir / f"{seq_id}.{start0}.{end0}.out.subst"

        # Skip windows already scored
        if out_subst.exists() and out_subst.stat().st_size > 0:
            continue

        jobs.append(
            {
                "sequence": sequence[start0:end0],
                "dms_input": str(subst_in),
                "dms_output": str(out_subst),
                "mutation_col": "mutant",
                "offset_idx": 1,
                "scoring_strategy": SCORING,
                "window_start": start0,
                "window_end": end0,
                "seq_len_full": seq_len,
            }
        )

    return jobs


# Mean score over the ESM-1v model columns for each variant of one output file
def _read_csv_means(path: Path) -> Dict[str, Optional[float]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or "mutant" not in header:
            return {}

        mutant_idx = header.index("mutant")
        out = {}
        for row in reader:
            if len(row) <= mutant_idx:
                continue
            scores = [_to_float(x) for x in row[mutant_idx + 1:]]
            scores = [s for s in scores if s is not None]
            out[row[mutant_idx]] = sum(scores) / len(scores) if scores else None
        return out


# Final score per variant: weighted average over the windows that contain it
def collect_scores(proteins, outdir):
    rows = []
    for seq_id, _, codes in proteins:
        variant_to_scores = defaultdict(list)

        for out_path in sorted((outdir / seq_id).glob(f"{seq_id}.*.*.out.subst")):
            if out_path.stat().st_size == 0:
                continue

            parts = out_path.name.split(".")
            start0, end0 = int(parts[-4]), int(parts[-3])
            window_len = end0 - start0

            # Local variant -> global variant
            for local_var, mean_score in _read_csv_means(out_path).items():
                parsed = _parse_mut(local_var)
                if mean_score is None or not parsed:
                    continue
                ref, local_pos1, alt = parsed
                global_var = f"{ref}{local_pos1 + start0}{alt}"
                variant_to_scores[global_var].append(
                    (mean_score, _window_weight(local_pos1, window_len))
                )

        for code in codes:
            pairs = variant_to_scores.get(code, [])
            den = sum(w for _, w in pairs)
            score = sum(s * w for s, w in pairs) / den if den else None
            rows.append(
                {
                    "seq_id": seq_id,
                    "variant": code,
                    "mutation_id": f"{seq_id}-{code}",
                    "score": score,
                    "n_windows": len(pairs),
                }
            )
    return rows


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    proteins = read_proteins(FASTA, SUBST_DIR)
    rejected = defaultdict(dict)

    # One job per window
    jobs_path = OUTDIR / "esm1v.jobs.jsonl"
    jobs = []
    for seq_id, sequence, codes in proteins:
        jobs.extend(_jobs_for_protein(seq_id, sequence, codes, OUTDIR, rejected))
    with jobs_path.open("w", encoding="utf-8") as jf:
        for job in jobs:
            jf.write(json.dumps(job) + "\n")

    print(f"[ESM1v] proteins={len(proteins)} | jobs={len(jobs)}")

    if jobs:
        tracker = EmissionsTracker(
            output_dir=str(OUTDIR),
            project_name="esm1v",
            output_file="emissions_esm1v.csv",
            log_level="error",
        )
        tracker.start()
        try:
            # predict.py (fair-esm) scores one window per call with all models
            with (OUTDIR / "esm1v.out").open("w", encoding="utf-8") as out, \
                 (OUTDIR / "esm1v.err").open("w", encoding="utf-8") as err:
                for i, job in enumerate(jobs, start=1):
                    cmd = [
                        "python", str(ESM_SCRIPT),
                        "--model-location", *[str(m) for m in MODELS],
                        "--sequence", job["sequence"],
                        "--dms-input", job["dms_input"],
                        "--dms-output", job["dms_output"],
                        "--mutation-col", "mutant",
                        "--offset-idx", "1",
                        "--scoring-strategy", SCORING,
                    ]
                    subprocess.run(cmd, stdout=out, stderr=err, text=True, check=True)
                    if i % 50 == 0 or i == len(jobs):
                        print(f"[ESM1v] {i}/{len(jobs)} jobs")
        finally:
            tracker.stop()

    rows = collect_scores(proteins, OUTDIR)
    with (OUTDIR / "esm1v_scores.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seq_id", "variant", "mutation_id", "score", "n_windows"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)

    (OUTDIR / "rejected_variants.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")

    n_scored = sum(r["score"] is not None for r in rows)
    print(f"[ESM1v] done | variants={len(rows)} | scored={n_scored} | rejected={sum(len(v) for v in rejected.values())}")


if __name__ == "__main__":
    main()
