"""Portable configuration for the RAF runtime."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RAFConfig:
    sensitive_col: str = "group"
    feature_col: str = "embedding"
    n_clusters: int = 20
    random_state: int = 42
    cluster_random_state: int | None = None
    cluster_subsample_ratio: float | None = 0.1
    cluster_subsample_size: int | None = None
    cluster_assign_batch_size: int = 4096
    fair_kappa: float = 0.0
    fair_epsilon_p: float = 1e-6
    fair_alpha_eps: float = 1.0
    fair_eps_min: float = 0.1
    policy_warmup_rounds: int = 0
    eps: float = 0.0
    batch_size: int = 4096
    hybrid_neighbor_k: int = 50
    raf_redundancy_elimination: bool = False
    raf_redundancy_exact_check: bool = True
    raf_redundancy_tolerance: float = 1e-8
    hnsw_single_index: bool = False
    hnsw_delta_capacity: int = 4096
    hnsw_rebuild_delta_threshold: int = 2048
    hnsw_ef_search: int = 100
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    max_steps: int = 200_000
    max_cost: float = 20_000.0
    max_accepted: int | None = None
    demand_decay_per_candidate: float = 0.2

    def __post_init__(self) -> None:
        positive_ints = (
            "n_clusters", "cluster_assign_batch_size", "batch_size", "hybrid_neighbor_k",
            "hnsw_delta_capacity", "hnsw_rebuild_delta_threshold", "hnsw_ef_search",
            "hnsw_m", "hnsw_ef_construction", "max_steps",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_accepted is not None and (
            isinstance(self.max_accepted, bool) or not isinstance(self.max_accepted, int) or self.max_accepted <= 0
        ):
            raise ValueError("max_accepted must be a positive integer or null")
        for name in ("max_cost", "fair_epsilon_p", "demand_decay_per_candidate"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("eps", "fair_kappa", "fair_alpha_eps", "raf_redundancy_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.fair_eps_min <= 1:
            raise ValueError("fair_eps_min must be between zero and one")
        if self.policy_warmup_rounds < 0:
            raise ValueError("policy_warmup_rounds must be nonnegative")
        if not self.sensitive_col or not self.feature_col or self.sensitive_col == self.feature_col:
            raise ValueError("feature_col and sensitive_col must be distinct, nonempty names")

    @classmethod
    def from_json(cls, path: str | Path) -> "RAFConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceSpec:
    name: str
    path: str
    cost: float = 1.0
    feature_col: str = "embedding"
    sensitive_col: str = "group"
    seed: int | None = None
    with_replacement: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.path:
            raise ValueError("A source needs a name and a path")
        if not math.isfinite(self.cost) or self.cost <= 0:
            raise ValueError("Source cost must be finite and positive")
