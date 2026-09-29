from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AcquisitionLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        # Vocabulary is intentionally acquisition-only. RAF decisions do not belong here.
        if event not in {"candidate_ready", "skipped", "source_error"}:
            raise ValueError(f"unsupported acquisition event: {event}")
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, "status": event, **fields}
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def record(self, status: str, **fields: Any) -> None:
        self.write(status, **fields)
