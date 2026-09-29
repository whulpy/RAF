from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np




@dataclass(slots=True)
class SourceProfile:
    source_id: str
    name: str
    connector_type: str
    modality: str
    domain: str
    license: str | None = None
    homepage: str | None = None
    description: str | None = None
    group_attributes: list[str] = field(default_factory=list)
    connector_version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_unit: str | None = None
    access_type: str = "unknown"
    api_base_url: str | None = None
    supports_search: bool = False
    supports_pagination: bool = False
    requires_auth: bool = False
    available_metadata: list[str] = field(default_factory=list)
    documentation_url: str | None = None
    compatibility_score: float | None = None
    compatibility_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RawSample:
    source_id: str
    external_id: str
    content_url: str | None = None
    local_path: Path | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    endpoint: str | None = None


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LabelResult:
    labels: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RAFCandidate:
    candidate_id: str
    source_id: str
    external_id: str
    modality: str
    raw_path: Path
    processed_path: Path | None
    embedding: np.ndarray
    group_labels: dict[str, Any]
    metadata: dict[str, Any]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        self.embedding = np.asarray(self.embedding, dtype=np.float32).reshape(-1)

    def to_manifest_record(self, embedding_index: int | None = None) -> dict[str, Any]:
        record = {
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "external_id": self.external_id,
            "modality": self.modality,
            "raw_path": str(self.raw_path),
            "processed_path": str(self.processed_path) if self.processed_path is not None else None,
            "group_labels": self.group_labels,
            "metadata": self.metadata,
            "provenance": self.provenance,
            "embedding_dimension": int(self.embedding.size),
        }
        if embedding_index is not None:
            record["embedding_index"] = embedding_index
        return record
