from collections.abc import Set
from pathlib import Path
import re
import subprocess
import logging
from typing import List, Tuple

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from dataset import Protein

logger = logging.getLogger(__name__)

def _compute_window_single_mut(pos1: int, total_length: int, max_len: int = 1022):
    # If the protein fits entirely in ESM, no trimming is needed
    if total_length <= max_len:
        return 0, total_length

    # Center the window around the mutation
    center = pos1 - 1
    start = max(0, center - max_len // 2)
    end = min(total_length, start + max_len)

    # Adjust the window if it falls near the end of the sequence
    if end - start < max_len:
        start = max(0, end - max_len)

    return start, end

def mutate_sequence(
    proteins: List[Protein],
    out_fasta: Path,
    delta: bool = False,
    max_len: int = 1022,
):
    mut_records = []
    wt_records = []
    trim_info = {}
    seen_wt = set()

    wt_fasta = out_fasta.with_name("wildtype.fasta") if delta else None

    for prot in proteins:
        wt_seq = prot.sequence.rstrip("*")
        # ESM-1v does not support sequences with internal stop codons
        if "*" in wt_seq:
            logger.warning(
                "[mutate_sequence] protein skipped: %s | "
                "reason=internal_stop_in_reference_sequence",
                prot.header,
            )
            continue
        L = len(wt_seq)

        for var in prot.get_variants():
            mut = var.code

            # validate the mutant format (e.g., "A123C")
            try:
                wt, pos1, alt = _parse_mutant(mut)
            except ValueError:
                logger.warning(
                    "[mutate_sequence] invalid mutant: %s | %s", prot.header, mut
                )
                continue

            # discard mutations that are out of range
            if pos1 < 1 or pos1 > L:
                logger.warning(
                    "[mutate_sequence] out of range: %s | %s", prot.header, mut
                )
                continue

            # check that the wild-type amino acid matches the expected one in the sequence
            if wt_seq[pos1 - 1] != wt:
                logger.warning(
                    "[mutate_sequence] WT mismatch: %s | %s | expected=%s found=%s",
                    prot.header,
                    mut,
                    wt,
                    wt_seq[pos1 - 1],
                )
                continue

            # compute the window around the mutation, trimming if necessary
            start, end = _compute_window_single_mut(pos1, L, max_len=max_len)
            trimmed = not (start == 0 and end == L)

            # local position of the mutation within the extracted window (1-based)
            specific_tok = pos1 - start

            mut_uid = f"{prot.header}-{mut}"

            # build the mutated sequence by replacing the amino acid at the specified position
            mut_seq = list(wt_seq)
            mut_seq[pos1 - 1] = alt
            mut_seq = "".join(mut_seq)

            # the real trimming is applied when preparing the ESM batch, so we save the full sequence here
            mut_records.append(SeqRecord(Seq(mut_seq), id=mut_uid, description=""))

            # save info for mapping embeddings and positions
            trim_info[mut_uid] = {
                "header": prot.header,
                "variant": mut,
                "variant_uid": mut_uid,
                "position": pos1,
                "start": start,
                "end": end,
                "trimmed": trimmed,
                "specific_tok": specific_tok,
                "mut_pt": f"{mut_uid}.pt",
                "wt_pt": f"{prot.header}.pt" if not trimmed else f"{mut_uid}.pt",
            }

            if delta:
                # in delta mode, we also need the WT sequence for the same window
                wt_uid = prot.header if not trimmed else mut_uid

                # avoid duplicating WT for untrimmed proteins
                if wt_uid not in seen_wt:
                    seen_wt.add(wt_uid)
                    wt_records.append(SeqRecord(Seq(wt_seq), id=wt_uid, description=""))

    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(mut_records, str(out_fasta), "fasta")

    if delta and wt_fasta is not None:
        SeqIO.write(wt_records, str(wt_fasta), "fasta")

    return {
        "out_fasta": str(out_fasta),
        "wt_fasta": str(wt_fasta) if wt_fasta else None,
        "trim_info": trim_info,
    }

def extract_embedding_batch(
    fasta: Path,
    trim_info: dict,
    needed_uids: Set[str] | None,
    model_path: Path,
    script_path: Path,
    out_dir: Path,
    batch_size_seqs: int = 256,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    tool_log = out_dir / "emb_extraction.log"

    jobs = []
    skipped_existing = 0

    for record in SeqIO.parse(str(fasta), "fasta"):
        uid = record.id

        # Mutant case: the uid coincides directly with trim_info
        if uid in trim_info:
            info = trim_info[uid]
            variant_uid = uid

        else:
            # WT case not trimmed: the uid is only the protein header
            matches = [
                info
                for info in trim_info.values()
                if info["header"] == uid and not info["trimmed"]
            ]

            # keep only the matches that are in needed_uids if needed_uids is provided
            if needed_uids is not None:
                matches = [
                    info for info in matches if info["variant_uid"] in needed_uids
                ]

            if not matches:
                logger.warning("[extract_embedding] missing trim info for uid=%s", uid)
                continue

            info = matches[0]
            variant_uid = info["variant_uid"]

        if needed_uids is not None and variant_uid not in needed_uids:
            continue

        # do not re-extract embeddings if the .pt file already exists and is non-empty
        out_pt = out_dir / f"{uid}.pt"
        if out_pt.exists() and out_pt.stat().st_size > 0:
            skipped_existing += 1
            continue

        seq = str(record.seq)

        # apply trimming only when preparing the ESM batch
        if info["trimmed"]:
            seq = seq[info["start"] : info["end"]]

        jobs.append((uid, seq))

    logger.info(
        "[extract_embedding] candidates=%d | skipped_existing=%d | out_dir=%s",
        len(jobs),
        skipped_existing,
        out_dir,
    )

    if not jobs:
        return

    n_batches = (len(jobs) + batch_size_seqs - 1) // batch_size_seqs

    for bi in range(n_batches):
        chunk = jobs[bi * batch_size_seqs : (bi + 1) * batch_size_seqs]
        batch_fasta = out_dir / f"_batch_{bi:04d}.fasta"

        # temporary FASTA file
        with batch_fasta.open("w", encoding="utf-8") as fh:
            for uid, seq in chunk:
                fh.write(f">{uid}\n{seq}\n")

        cmd = [
            "python",
            str(script_path),
            str(model_path),
            str(batch_fasta),
            str(out_dir),
            "--repr_layers",
            "33",
            "--include",
            "per_tok",
        ]

        logger.info(
            "[extract_embedding] batch %d/%d | n_seq=%d | outdir=%s",
            bi + 1,
            n_batches,
            len(chunk),
            out_dir,
        )

        try:
            # accumulate log files 
            with tool_log.open("a", encoding="utf-8") as out:
                out.write(f"\n=== batch {bi + 1}/{n_batches} | n_seq={len(chunk)} ===\n")
                out.write("CMD: " + " ".join(cmd) + "\n")
                subprocess.run(cmd, check=True, stdout=out, stderr=out, text=True)
        finally:
            # The temporary FASTA file is not preserved after extracting embeddings
            batch_fasta.unlink(missing_ok=True)

def _parse_mutant(mut: str) -> Tuple[str, int, str]:
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", str(mut))
    if not m:
        raise ValueError(f"Invalid mutant format: {mut}")

    return m.group(1), int(m.group(2)), m.group(3)

