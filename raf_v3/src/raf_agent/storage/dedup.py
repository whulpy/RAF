from __future__ import annotations

import sqlite3
from pathlib import Path


class DedupStore:
    """Persistent two-key deduplication for source IDs and content hashes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS samples (source_id TEXT NOT NULL, external_id TEXT NOT NULL, "
            "content_sha256 TEXT NOT NULL UNIQUE, candidate_id TEXT NOT NULL, "
            "PRIMARY KEY (source_id, external_id))"
        )
        self._db.commit()

    def has_external_id(self, source_id: str, external_id: str) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM samples WHERE source_id=? AND external_id=?", (source_id, external_id)
        ).fetchone()
        return row is not None

    def seen_external_id(self, source_id: str, external_id: str) -> bool:
        return self.has_external_id(source_id, external_id)

    def has_content_hash(self, content_sha256: str) -> bool:
        return self._db.execute(
            "SELECT 1 FROM samples WHERE content_sha256=?", (content_sha256,)
        ).fetchone() is not None

    def seen_hash(self, content_hash: str) -> bool:
        return self.has_content_hash(content_hash)

    def add(self, source_id: str, external_id: str, content_sha256: str, candidate_id: str) -> None:
        self._db.execute(
            "INSERT INTO samples(source_id, external_id, content_sha256, candidate_id) VALUES(?,?,?,?)",
            (source_id, external_id, content_sha256, candidate_id),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
