from __future__ import annotations

import csv
import json
from pathlib import Path

from .base import GroupLabelProvider
from ..models import LabelResult, RawSample


class ManifestLabelProvider(GroupLabelProvider):
    def __init__(self, path: str | Path, *, id_column: str = "external_id", label_columns: list[str] | None = None):
        self.path = Path(path)
        self.id_column = id_column
        self.label_columns = label_columns
        self._rows = self._load()

    def _load(self) -> dict[str, dict[str, str]]:
        if self.path.suffix.lower() == ".jsonl":
            with self.path.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        else:
            with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        return {str(row[self.id_column]): row for row in rows if row.get(self.id_column) is not None}

    def label(self, sample: RawSample) -> LabelResult:
        row = self._rows.get(sample.external_id, {})
        keys = self.label_columns or [key for key in row if key != self.id_column]
        labels = {key: str(row[key]) for key in keys if row.get(key) not in (None, "")}
        return LabelResult(labels, {"provider": "manifest", "manifest": str(self.path)})
