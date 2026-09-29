from __future__ import annotations

from .base import SourceConnector
from ..exceptions import ConfigurationError, UnknownSourceError
from ..models import SourceProfile


class SourceRegistry:
    def __init__(self) -> None:
        self._connectors: dict[str, SourceConnector] = {}

    def register(self, connector: SourceConnector, *, replace: bool = False) -> None:
        source_id = connector.profile.source_id
        if source_id in self._connectors and not replace:
            raise ConfigurationError(f"source already registered: {source_id}")
        self._connectors[source_id] = connector

    def get(self, source_id: str) -> SourceConnector:
        try:
            return self._connectors[source_id]
        except KeyError as exc:
            raise UnknownSourceError(f"unknown source: {source_id}") from exc

    def list(self) -> list[SourceProfile]:
        return [self._connectors[key].profile for key in sorted(self._connectors)]

    def list_profiles(self) -> list[SourceProfile]:
        return self.list()

    def close(self) -> None:
        for connector in self._connectors.values():
            connector.close()
