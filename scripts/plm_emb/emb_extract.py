from __future__ import annotations

import logging
from pathlib import Path
from time import time

from emb_utils import (
    mutate_sequence,
    extract_embedding_batch,
)

logger = logging.getLogger(__name__)

class EmbExtract:
    name = "embeddings_extraction"

    def __init__(self, proteins, method_params, output_dir, shared_dir):
        self.proteins = proteins
        self.method_params = method_params
        self.output_dir = Path(output_dir)
        self.shared_dir = Path(shared_dir)

    def prepare(self) -> None:
        logger.info("[%s] start | Extract per-token ESM1v embeddings", self.name)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.delta = bool(self.method_params.get("delta", False))

        self.mutated_fasta = self.output_dir / "mutated.fasta"
        self.emb_dir = self.shared_dir
        self.mut_emb_dir = self.emb_dir / "mut"
        self.wt_emb_dir = self.emb_dir / "wt"

        # Create directories if they don't exist
        self.emb_dir.mkdir(parents=True, exist_ok=True)
        self.mut_emb_dir.mkdir(parents=True, exist_ok=True)
        if self.delta:
            self.wt_emb_dir.mkdir(parents=True, exist_ok=True)

        t0 = time()

        res = mutate_sequence(
            proteins=self.proteins,
            out_fasta=self.mutated_fasta,
            delta=self.delta,
            max_len=int(self.method_params.get("max_len", 1022)),
        )

        self.wt_fasta = Path(res["wt_fasta"]) if res["wt_fasta"] else None
        self.trim_info = res["trim_info"]

        self._write_trim_info()

        logger.info(
            "[%s] prepare done | mut_fasta=%s | wt_fasta=%s | variants=%d | elapsed_s=%.2f",
            self.name,
            self.mutated_fasta,
            self.wt_fasta,
            len(self.trim_info),
            time() - t0,
        )

    def extract(self) -> None:
        needed_uids = None

        extract_embedding_batch(
            fasta=self.mutated_fasta,
            trim_info=self.trim_info,
            needed_uids=needed_uids,
            model_path=Path(self.method_params["model_embedding"]),
            script_path=Path(self.method_params["script_embedding"]),
            out_dir=self.mut_emb_dir,
            batch_size_seqs=int(self.method_params.get("batch_size_seqs", 256)),
        )

        if self.delta and self.wt_fasta is not None:
            extract_embedding_batch(
                fasta=self.wt_fasta,
                trim_info=self.trim_info,
                needed_uids=needed_uids,
                model_path=Path(self.method_params["model_embedding"]),
                script_path=Path(self.method_params["script_embedding"]),
                out_dir=self.wt_emb_dir,
                batch_size_seqs=int(self.method_params.get("batch_size_seqs", 256)),
            )

        logger.info("[%s] extract done", self.name)

    def run(self):
        self.prepare()
        self.extract()

        return {
            "mutated_fasta": self.mutated_fasta,
            "wt_fasta": self.wt_fasta,
            "emb_dir": self.emb_dir,
            "mut_emb_dir": self.mut_emb_dir,
            "wt_emb_dir": self.wt_emb_dir if self.delta else None,
            "trim_info": self.trim_info,
            "embedding_type": "per_tok",
            "delta": self.delta,
        }

    def _write_trim_info(self) -> None:
        path = self.output_dir / "trim_info.tsv"

        with path.open("w", encoding="utf-8") as f:
            f.write(
                "variant_uid\theader\tvariant\ttrimmed\tstart\tend\t"
                "specific_tok\tmut_pt\twt_pt\n"
            )

            for uid, info in self.trim_info.items():
                f.write(
                    f"{uid}\t"
                    f"{info['header']}\t"
                    f"{info['variant']}\t"
                    f"{info['trimmed']}\t"
                    f"{info['start']}\t"
                    f"{info['end']}\t"
                    f"{info['specific_tok']}\t"
                    f"{info['mut_pt']}\t"
                    f"{info['wt_pt']}\n"
                )

