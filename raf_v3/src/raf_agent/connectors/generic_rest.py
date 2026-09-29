from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any

from .base import SourceConnector
from .http import RobustHttpClient
from .state import JsonState
from ..exceptions import ConfigurationError, SourceExhaustedError, SourceUnavailableError
from ..models import RawSample, SourceProfile


def _at(value: Any, path: str | None, default: Any = None) -> Any:
    if not path:
        return value
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part, default)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else default
        else:
            return default
    return current


class GenericRESTConnector(SourceConnector):
    """Config-driven JSON API connector with persistent pagination state."""

    def __init__(
        self,
        profile: SourceProfile,
        endpoint: str,
        *,
        method: str = "GET",
        items_path: str = "rows",
        id_path: str = "row_idx",
        url_path: str = "row.image.src",
        metadata_path: str | None = "row",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        auth_env: dict[str, str] | None = None,
        pagination: dict[str, Any] | None = None,
        state_path: str | Path | None = None,
        http: RobustHttpClient | None = None,
    ) -> None:
        self._profile = profile
        self.endpoint = endpoint
        self.method = method.upper()
        self.items_path = items_path
        self.id_path = id_path
        self.url_path = url_path
        self.metadata_path = metadata_path
        self.params = dict(params or {})
        self.headers = dict(headers or {})
        for header, env_name in (auth_env or {}).items():
            value = os.environ.get(env_name)
            if not value:
                raise ConfigurationError(f"missing credential environment variable: {env_name}")
            self.headers[header] = value
        self.pagination = {
            "mode": "offset",
            "parameter": "offset",
            "start": 0,
            "step": 100,
            "limit_parameter": "length",
            "page_size": 100,
            **(pagination or {}),
        }
        self._state_store = JsonState(state_path)
        self._state = self._state_store.load(
            {"position": self.pagination["start"], "seen_external_ids": [], "buffer": []}
        )
        self._seen = set(str(value) for value in self._state.get("seen_external_ids", []))
        self._buffer: list[dict[str, Any]] = [
            item for item in self._state.get("buffer", []) if isinstance(item, dict)
        ]
        self._http = http or RobustHttpClient()

    @property
    def profile(self) -> SourceProfile:
        return self._profile

    def _fill(self) -> None:
        if self._state.get("exhausted", False):
            raise SourceExhaustedError(f"source exhausted: {self.profile.source_id}")
        mode = self.pagination.get("mode", "offset")
        if mode == "random_offset":
            draw_index = int(self._state.get("random_draw_index", 0))
            seed = int(self.pagination.get("seed", 0))
            maximum = int(self.pagination["max_offset"])
            digest = hashlib.sha256(f"{seed}:{draw_index}".encode("utf-8")).digest()
            position = int.from_bytes(digest[:8], "big") % (maximum + 1)
            self._state["random_draw_index"] = draw_index + 1
        else:
            position = self._state.get("position", self.pagination["start"])
        substitutions = {"page": position, "offset": position, "cursor": position}
        params = {
            key: (value.format(**substitutions) if isinstance(value, str) and "{" in value else value)
            for key, value in self.params.items()
        }
        if self.pagination.get("parameter"):
            params[self.pagination["parameter"]] = position
        if self.pagination.get("limit_parameter"):
            params[self.pagination["limit_parameter"]] = self.pagination["page_size"]
        response = self._http.request(self.method, self.endpoint, params=params, headers=self.headers)
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise SourceUnavailableError(f"source returned invalid JSON: {self.profile.source_id}") from exc
        items = _at(payload, self.items_path, [])
        if not isinstance(items, list):
            raise SourceUnavailableError(f"source returned a non-list items field: {self.profile.source_id}")
        if not items:
            if mode != "random_offset":
                self._state["exhausted"] = True
                self.save_state()
                raise SourceExhaustedError(f"source exhausted: {self.profile.source_id}")
            raise SourceUnavailableError(f"source returned no items: {self.profile.source_id}")
        self._buffer.extend(item for item in items if isinstance(item, dict))
        if mode == "cursor":
            cursor = _at(payload, self.pagination.get("next_cursor_path"))
            if cursor is None:
                self._state["exhausted"] = True
            else:
                self._state["position"] = cursor
        elif mode != "random_offset":
            self._state["position"] = int(position) + int(self.pagination.get("step", len(items)))
        self.save_state()

    def fetch_one(self) -> RawSample:
        for _ in range(1000):
            if not self._buffer:
                self._fill()
            item = self._buffer.pop(0)
            self.save_state()
            external_id = _at(item, self.id_path)
            content_url = _at(item, self.url_path)
            if external_id is None or not content_url:
                continue
            external_id = str(external_id)
            if external_id in self._seen:
                continue
            self._seen.add(external_id)
            self.save_state()
            metadata = _at(item, self.metadata_path, {})
            if not isinstance(metadata, dict):
                metadata = {"value": metadata}
            return RawSample(
                source_id=self.profile.source_id,
                external_id=external_id,
                content_url=str(content_url),
                raw_metadata=metadata,
                endpoint=self.endpoint,
            )
        raise SourceUnavailableError(f"too many duplicate or invalid records: {self.profile.source_id}")

    def save_state(self) -> None:
        self._state["seen_external_ids"] = sorted(self._seen)
        self._state["buffer"] = self._buffer
        self._state_store.save(self._state)
