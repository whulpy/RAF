from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import RawSample, SourceProfile


class SourceConnector(ABC):
    """A stateful source that yields one raw item at a time."""

    @property
    @abstractmethod
    def profile(self) -> SourceProfile:
        raise NotImplementedError

    @abstractmethod
    def fetch_one(self) -> RawSample:
        raise NotImplementedError

    @abstractmethod
    def save_state(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.save_state()

    def prepare(self) -> dict[str, object]:
        """Prepare a source that needs an installation step.

        Online record APIs normally need no preparation.  Bulk-download
        connectors override this method so callers can install them before a
        RAF run instead of paying the setup cost on the first pull.
        """
        return {"source_id": self.profile.source_id, "ready": True, "action": "none"}

    def fetch_many(self, n: int) -> list[RawSample]:
        return [self.fetch_one() for _ in range(n)]


DataSourceConnector = SourceConnector
