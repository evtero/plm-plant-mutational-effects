# dataset.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass(frozen=True)
class Prediction:
    method: str
    score: Optional[float]
    label: str
    raw: Optional[dict] = None


@dataclass
class Variant:
    code: str
    protein_header: str
    predictions: Dict[str, Prediction] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.protein_header}-{self.code}"

    def get_protein_header(self) -> str:
        return self.protein_header


@dataclass
class Protein:
    header: str
    sequence: str
    uniprot_ids: List[str] = field(default_factory=list)
    variants: List[Variant] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.header

    def get_variants(self) -> List[Variant]:
        return self.variants
