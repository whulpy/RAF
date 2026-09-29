from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode_batch(self, paths: Sequence[Path]) -> np.ndarray:
        raise NotImplementedError

    def encode(self, path: Path) -> np.ndarray:
        matrix = self.encode_batch([path])
        return np.asarray(matrix[0], dtype=np.float32)
