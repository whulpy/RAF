from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from .base import SourceConnector
from .state import JsonState
from ..exceptions import ConfigurationError, SourceExhaustedError
from ..models import RawSample, SourceProfile


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class LocalFolderConnector(SourceConnector):
    def __init__(
        self,
        profile: SourceProfile,
        root: str | Path,
        *,
        metadata_csv: str | Path | None = None,
        path_column: str = "path",
        id_column: str = "id",
        seed: int = 0,
        state_path: str | Path | None = None,
    ) -> None:
        self._profile = profile
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ConfigurationError(f"local source directory does not exist: {self.root}")
        self._rows = self._load_rows(metadata_csv, path_column, id_column)
        order = list(range(len(self._rows)))
        random.Random(seed).shuffle(order)
        self._state_store = JsonState(state_path)
        state = self._state_store.load({"cursor": 0, "order": order})
        self._order = [int(value) for value in state.get("order", order)]
        if sorted(self._order) != list(range(len(self._rows))):
            self._order = order
            state["cursor"] = 0
        self._cursor = int(state.get("cursor", 0))

    @property
    def profile(self) -> SourceProfile:
        return self._profile

    def _load_rows(self, metadata_csv: str | Path | None, path_column: str, id_column: str) -> list[dict[str, Any]]:
        if metadata_csv:
            manifest = Path(metadata_csv).expanduser().resolve()
            with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            result: list[dict[str, Any]] = []
            for row_number, row in enumerate(rows, start=2):
                raw_path = row.get(path_column)
                if not raw_path:
                    raise ConfigurationError(f"missing {path_column!r} in {manifest}:{row_number}")
                path = (self.root / raw_path).resolve()
                if self.root not in path.parents and path != self.root:
                    raise ConfigurationError(f"manifest path escapes source root: {raw_path}")
                metadata = dict(row)
                external_id = str(row.get(id_column) or raw_path)
                result.append({"path": path, "external_id": external_id, "metadata": metadata})
            return result
        paths = sorted(path for path in self.root.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS)
        return [
            {"path": path, "external_id": path.relative_to(self.root).as_posix(), "metadata": {}}
            for path in paths
        ]

    def fetch_one(self) -> RawSample:
        if self._cursor >= len(self._order):
            raise SourceExhaustedError(f"source exhausted: {self.profile.source_id}")
        row = self._rows[self._order[self._cursor]]
        self._cursor += 1
        self.save_state()
        return RawSample(
            source_id=self.profile.source_id,
            external_id=row["external_id"],
            local_path=row["path"],
            raw_metadata=dict(row["metadata"]),
            endpoint=str(self.root),
        )

    def save_state(self) -> None:
        self._state_store.save({"cursor": self._cursor, "order": self._order})
