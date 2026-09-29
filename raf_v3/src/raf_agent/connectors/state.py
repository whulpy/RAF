from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


class JsonState:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None

    def load(self, default: dict[str, Any]) -> dict[str, Any]:
        if self.path is None or not self.path.exists():
            return dict(default)
        with self.path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else dict(default)

    def save(self, value: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(6):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    # Antivirus/indexing processes can briefly hold a freshly
                    # replaced JSON file on Windows. Keep the atomic commit and
                    # retry the same temporary file instead of truncating state.
                    time.sleep(0.05 * (attempt + 1))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
