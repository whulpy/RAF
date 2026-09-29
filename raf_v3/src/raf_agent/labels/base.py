from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import LabelResult, RawSample


class GroupLabelProvider(ABC):
    @abstractmethod
    def label(self, sample: RawSample) -> LabelResult:
        raise NotImplementedError

    def get_labels(self, raw_sample: RawSample, processed_path: Path | None = None) -> LabelResult:
        return self.label(raw_sample)
