from __future__ import annotations

from typing import Any

from .base import GroupLabelProvider
from ..models import LabelResult, RawSample


def _at(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


class SourceMetadataLabelProvider(GroupLabelProvider):
    def __init__(self, mappings: dict[str, Any]):
        self.mappings = dict(mappings)

    def label(self, sample: RawSample) -> LabelResult:
        labels = {}
        provenance_mappings = {}
        for output_name, specification in self.mappings.items():
            if isinstance(specification, str):
                metadata_path = specification
                value_map = {}
            else:
                metadata_path = str(specification["path"])
                value_map = {str(key): str(value) for key, value in specification.get("value_map", {}).items()}
            value = _at(sample.raw_metadata, metadata_path)
            if value is not None and str(value).strip():
                labels[output_name] = value_map.get(str(value), str(value))
            provenance_mappings[output_name] = {"path": metadata_path, "value_map": value_map}
        return LabelResult(labels, {"provider": "source_metadata", "mappings": provenance_mappings})
