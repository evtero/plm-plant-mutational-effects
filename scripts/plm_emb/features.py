from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

class Feature:
    name = "feature"

    def __init__(
        self,
        proteins,
        data_tsv: Path | str,
        method_params: dict,
        output_dir: Path,
    ) -> None:
        self.proteins = proteins
        self.data_tsv = Path(data_tsv)
        self.method_params = method_params
        self.output_dir = Path(output_dir)

        self.emb_dir = Path(method_params["emb_dir"])
        self.trim_info: dict = method_params["trim_info"]

        self.embedding_type: str = method_params.get("embedding_type", "full-mean")
        self.delta: bool = bool(method_params.get("delta", False))
        self.use_sift: bool = bool(method_params.get("use_sift", False))
        self.window_size: int = int(method_params.get("window_size", 13))
        
        if self.embedding_type not in {"full-mean", "local-mean", "mut-site"}:
            raise ValueError(f"Unknown embedding_type: {self.embedding_type!r}")

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.data_tsv, sep="\t")

        required_cols = {"mutation_id", "seq_id", "Effect"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns in data_tsv: {sorted(missing_cols)}"
            )

        X: list[np.ndarray] = []
        y: list[float] = []
        meta_rows: list[dict] = []

        n_missing_trim = 0
        n_missing_mut_pt = 0
        n_missing_wt_pt = 0
        n_missing_sift = 0
        n_bad_label = 0

        missing = sum(1 for uid in df["mutation_id"] if uid not in self.trim_info)
        logger.info(
            "[feature] metadata_uids=%d | trim_info_uids=%d | missing=%d",
            len(df),
            len(self.trim_info),
            missing,
        )

        for _, row in df.iterrows():
            uid = row["mutation_id"]
            header = row["seq_id"]

            info = self.trim_info.get(uid)
            if info is None:
                n_missing_trim += 1
                continue

            mut_pt = self.emb_dir / "mut" / info["mut_pt"]
            if not mut_pt.exists():
                n_missing_mut_pt += 1
                continue

            # specific_tok in trim_info is 1-based (position - start), converted to 0-based
            mut_emb = self._build_embedding(mut_pt, int(info["specific_tok"]))

            if self.delta:
                wt_pt = self.emb_dir / "wt" / info["wt_pt"]
                if not wt_pt.exists():
                    n_missing_wt_pt += 1
                    continue
                wt_emb = self._build_embedding(wt_pt, int(info["specific_tok"]))
                emb = mut_emb - wt_emb
            else:
                emb = mut_emb

            if self.use_sift:
                sift_score = self._get_sift_score(header, info["variant"])
                if sift_score is None:
                    n_missing_sift += 1
                    continue
                emb = np.concatenate([emb, np.array([sift_score], dtype=float)])

            label = self._effect_to_y(row["Effect"])
            if label is None:
                n_bad_label += 1
                continue

            X.append(emb)
            y.append(label)

            meta_row = {
                "mutation_id": uid,
                "seq_id": header,
                "mutation_code": row["mutation_code"],
                "Effect": row["Effect"],
                "family_id": row["family_id"],
                "split_group": row["split_group"],
                "variant": info["variant"],
                "trimmed": info["trimmed"],
                "start": info["start"],
                "end": info["end"],
                "specific_tok": info["specific_tok"],
            }
            meta_rows.append(meta_row)

        logger.info(
            "[feature] skipped | missing_trim=%d | missing_mut_pt=%d | "
            "missing_wt_pt=%d | missing_sift=%d | bad_label=%d",
            n_missing_trim,
            n_missing_mut_pt,
            n_missing_wt_pt,
            n_missing_sift,
            n_bad_label,
        )

        if not X:
            raise RuntimeError(
                "[feature] No features were built. "
                "Check missing embeddings, metadata mutation_id, labels, and SIFT availability."
            )

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        meta_df = pd.DataFrame(meta_rows)

        if X_arr.ndim != 2:
            raise RuntimeError(f"[feature] Expected 2D X, got shape={X_arr.shape}")

        if len(X_arr) != len(y_arr) or len(X_arr) != len(meta_df):
            raise RuntimeError(
                f"[feature] Feature/y/meta length mismatch: "
                f"X={len(X_arr)}, y={len(y_arr)}, meta={len(meta_df)}"
            )

        logger.info("[%s] X shape: %s | y shape: %s", self.name, X_arr.shape, y_arr.shape)

        suffix = self.embedding_type
        if self.delta:
            suffix = f"delta_{suffix}"
        if self.use_sift:
            suffix = f"{suffix}_sift"
        
        features_path = self.output_dir / f"features_{suffix}.pkl"
        meta_path = self.output_dir / f"features_{suffix}_meta.tsv"

        with features_path.open("wb") as f:
            pickle.dump(
                {
                    "X": X_arr,
                    "y": y_arr,
                    "meta": meta_df,
                    "embedding_type": self.embedding_type,
                    "delta": self.delta,
                    "use_sift": self.use_sift,
                },
                f,
            )

        meta_df.to_csv(meta_path, sep="\t", index=False)

        logger.info(
            "[%s] done | X=%s | y=%s | features=%s | meta=%s",
            self.name,
            X_arr.shape,
            y_arr.shape,
            features_path,
            meta_path,
        )

        return {
            "X": X_arr,
            "y": y_arr,
            "meta": meta_df,
            "features_path": features_path,
            "meta_path": meta_path,
        }

    def _build_embedding(self, pt_path: Path, specific_tok: int) -> np.ndarray:
        """specific_tok is 1-based (position within the extracted window); converted to 0-based."""
        per_tok = torch.load(pt_path, map_location="cpu")["representations"][33].numpy()
        # Conversion explicitly 1-based → 0-based
        idx = specific_tok - 1

        if idx < 0 or idx >= per_tok.shape[0]:
            raise ValueError(
                f"specific_tok={specific_tok} (idx={idx}) out of bounds "
                f"for {pt_path} with len={per_tok.shape[0]}"
            )

        if self.embedding_type == "full-mean":
            return per_tok.mean(axis=0)

        if self.embedding_type == "mut-site":
            return per_tok[idx]

        if self.embedding_type == "local-mean":
            half = self.window_size // 2
            start = max(0, idx - half)
            end = min(per_tok.shape[0], idx + half + 1)
            return per_tok[start:end].mean(axis=0)

        raise ValueError(f"Unknown embedding_type: {self.embedding_type!r}")

    def _effect_to_y(self, effect) -> float | None:
        effect = str(effect).strip().lower()
        if effect == "functional":
            return 1.0
        if effect == "neutral":
            return 0.0
        return None

    def _get_sift_score(self, header: str, mut: str) -> float | None:
        for protein in self.proteins:
            if protein.header != header:
                continue
            for var in protein.get_variants():
                if var.code != mut:
                    continue
                pred = var.predictions.get("sift")
                if pred is None or pred.score is None:
                    return None
                return float(pred.score)
        return None

