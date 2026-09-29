"""The single integration boundary between the image agent and RAF."""
from __future__ import annotations

from raf.data import DataSource, SourceManager

from ..exceptions import MissingRequiredLabelError, SourceExhaustedError
from ..models import RAFCandidate


class RAFAdapter:
    @staticmethod
    def to_raf_input(candidate: RAFCandidate, sensitive_attribute: str = "group", *,
                     feature_col: str = "embedding", sensitive_col: str = "group") -> dict:
        group = candidate.group_labels.get(sensitive_attribute)
        if group is None:
            raise MissingRequiredLabelError(f"Missing required group attribute {sensitive_attribute!r}")
        return {"record_id": candidate.candidate_id, "source_id": candidate.source_id,
                feature_col: candidate.embedding, sensitive_col: group,
                "raw_path": str(candidate.raw_path), "processed_path": str(candidate.processed_path),
                "provenance": dict(candidate.provenance)}


class RAFExternalDataSource(DataSource):
    def __init__(self, agent, source_id: str, *, cost: float = 1.0, sensitive_attribute: str = "group",
                 feature_col: str = "embedding", sensitive_col: str = "group") -> None:
        super().__init__(source_id, cost)
        self.agent = agent
        self.sensitive_attribute = sensitive_attribute
        self.feature_col = feature_col
        self.sensitive_col = sensitive_col

    def sample_one(self):
        if self.exhausted:
            return None
        try:
            candidate = self.agent.fetch_one(self.name)
        except SourceExhaustedError:
            self.exhausted = True
            return None
        self.num_samples_drawn += 1
        return RAFAdapter.to_raf_input(candidate, self.sensitive_attribute,
                                       feature_col=self.feature_col, sensitive_col=self.sensitive_col)


class RAFExternalSourceManager(SourceManager):
    def __init__(self, agent, sources):
        super().__init__(sources)
        self.agent = agent

    def close_all(self) -> None:
        try:
            super().close_all()
        finally:
            self.agent.close()


def build_raf_source_manager(agent, *, costs: dict[str, float] | None = None, sensitive_attribute: str = "group",
                             feature_col: str = "embedding", sensitive_col: str = "group"):
    costs = costs or {}
    return RAFExternalSourceManager(agent, [
        RAFExternalDataSource(agent, profile.source_id, cost=float(costs.get(profile.source_id, 1.0)),
                              sensitive_attribute=sensitive_attribute, feature_col=feature_col, sensitive_col=sensitive_col)
        for profile in agent.list_sources()
    ])
