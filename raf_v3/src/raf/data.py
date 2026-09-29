"""Read existing datasets and expose source samplers; no dataset construction."""
from __future__ import annotations

import abc
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import SourceSpec


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError("Use Parquet or JSONL with an embedding array and an explicit group label")


class DataSource(abc.ABC):
    """One fixed RAF arm. Return None only when the source is exhausted."""

    def __init__(self, name: str, cost: float = 1.0) -> None:
        if not name or not math.isfinite(cost) or cost <= 0:
            raise ValueError("Source name must be nonempty and cost finite and positive")
        self.name = name
        self.cost = float(cost)
        self.exhausted = False
        self.num_samples_drawn = 0

    @abc.abstractmethod
    def sample_one(self) -> dict[str, Any] | None:
        """Return one existing sample with embedding and group information."""
        raise NotImplementedError

    def reset(self) -> None:
        self.exhausted = False
        self.num_samples_drawn = 0

    def close(self) -> None:
        pass


class InMemorySource(DataSource):
    """Uniform random sampling of supplied records, without replacement by default."""

    def __init__(self, name: str, cost: float, records: Sequence[dict[str, Any]], *,
                 seed: int | None = None, with_replacement: bool = False) -> None:
        super().__init__(name, cost)
        self._records = [dict(record) for record in records]
        self._seed = seed
        self._with_replacement = with_replacement
        self.reset()

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self._seed)
        self._remaining = list(range(len(self._records)))
        if not self._with_replacement:
            self._rng.shuffle(self._remaining)

    def sample_one(self) -> dict[str, Any] | None:
        if self.exhausted or not self._records or (not self._with_replacement and not self._remaining):
            self.exhausted = True
            return None
        if self._with_replacement:
            index = int(self._rng.integers(0, len(self._records)))
        else:
            pick = int(self._rng.integers(0, len(self._remaining)))
            index = self._remaining[pick]
            self._remaining[pick] = self._remaining[-1]
            self._remaining.pop()
        self.num_samples_drawn += 1
        return dict(self._records[index])


class ParquetSource(InMemorySource):
    """Local adapter for an existing Parquet/JSONL source, loaded into memory."""

    def __init__(self, spec: SourceSpec, *, feature_col: str = "embedding", sensitive_col: str = "group") -> None:
        frame = load_dataset(spec.path)
        for column in (spec.feature_col, spec.sensitive_col):
            if column not in frame:
                raise ValueError(f"Source {spec.name!r} is missing column {column!r}")
        records = frame.to_dict("records")
        for record in records:
            record[feature_col] = record[spec.feature_col]
            record[sensitive_col] = record[spec.sensitive_col]
        super().__init__(spec.name, spec.cost, records, seed=spec.seed, with_replacement=spec.with_replacement)


class SourceManager:
    def __init__(self, sources: Sequence[DataSource]) -> None:
        self.sources = list(sources)
        if not self.sources:
            raise ValueError("At least one source is required")
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("Source names must be unique")

    def __len__(self) -> int:
        return len(self.sources)

    def get(self, index: int) -> DataSource:
        return self.sources[index]

    def costs(self) -> np.ndarray:
        costs = np.asarray([s.cost for s in self.sources], dtype=float)
        if not np.isfinite(costs).all() or (costs <= 0).any():
            raise ValueError("Every source cost must be finite and positive")
        return costs

    def close_all(self) -> None:
        for source in self.sources:
            source.close()


def load_sources(path: str | Path, *, feature_col: str = "embedding", sensitive_col: str = "group",
                 seed: int = 42) -> SourceManager:
    """Instantiate samplers from explicit paths; never construct or rebalance data."""
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Source configuration must be a JSON list")
    sources = []
    for index, item in enumerate(payload):
        settings = {"feature_col": feature_col, "sensitive_col": sensitive_col,
                    "seed": seed + index + 1, **item}
        source_path = Path(settings["path"]).expanduser()
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        settings["path"] = str(source_path.resolve())
        sources.append(ParquetSource(SourceSpec(**settings), feature_col=feature_col, sensitive_col=sensitive_col))
    return SourceManager(sources)
