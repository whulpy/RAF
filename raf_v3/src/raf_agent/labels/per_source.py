from __future__ import annotations

from .base import GroupLabelProvider
from .metadata import SourceMetadataLabelProvider
from ..models import LabelResult, RawSample


class PerSourceMetadataLabelProvider(GroupLabelProvider):
    """Apply a different metadata schema for each registered source."""

    def __init__(self, mappings: dict[str, dict]):
        self.providers = {
            source_id: SourceMetadataLabelProvider(source_mappings)
            for source_id, source_mappings in mappings.items()
        }

    def label(self, sample: RawSample) -> LabelResult:
        provider = self.providers.get(sample.source_id)
        if provider is None:
            return LabelResult({}, {"provider": "per_source_metadata", "reason": "source_mapping_missing"})
        result = provider.label(sample)
        result.provenance["provider"] = "per_source_metadata"
        result.provenance["source_id"] = sample.source_id
        return result
