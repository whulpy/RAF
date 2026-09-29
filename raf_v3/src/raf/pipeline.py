"""RAF's complete streaming assembly loop, without experimental baselines."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .bandit import BanditState, FairnessWeightedEpsGreedyPolicy
from .clustering import ClusterSet, fit_clusters
from .config import RAFConfig
from .data import SourceManager
from .demand import GroupLabeler, build_counts_from_labels, build_group_labeler, recompute_q_from_counts
from .selection import SelectState, build_initial_select_states, split_major_minor, try_accept_incremental_hybrid
from .vector_ops import extract_embeddings


@dataclass
class PreparedState:
    query_df: pd.DataFrame
    cluster_set: ClusterSet
    group_labeler: GroupLabeler
    demand: np.ndarray
    major_vectors: np.ndarray
    major_norm_sq: np.ndarray
    select_states: dict[Any, SelectState]


@dataclass
class RAFRunResult:
    """Accepted counts distinguish cumulative events from final retained records."""
    sampled: int = 0
    candidates: int = 0
    accepted: int = 0
    retained: int = 0
    removed: int = 0
    skipped_not_minor: int = 0
    skipped_no_demand: int = 0
    total_cost: float = 0.0
    total_gain: float = 0.0
    valuation_s: float = 0.0
    elapsed_s: float = 0.0
    stop_reason: str = "max_steps"
    source_sampled: dict[str, int] = field(default_factory=dict)
    source_candidates: dict[str, int] = field(default_factory=dict)
    accepted_records: list[dict[str, Any]] = field(default_factory=list)
    augmented_df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def summary(self) -> dict[str, Any]:
        return {name: value for name, value in vars(self).items()
                if name not in {"accepted_records", "augmented_df"}}


class RAFPipeline:
    def __init__(self, config: RAFConfig | None = None, *,
                 progress_callback: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.config = config or RAFConfig()
        self.progress_callback = progress_callback

    def prepare(self, query_df: pd.DataFrame) -> PreparedState:
        cfg = self.config
        frame = query_df.copy().reset_index(drop=True)
        if frame.empty:
            raise ValueError("The initial dataset must not be empty")
        for name in (cfg.feature_col, cfg.sensitive_col):
            if name not in frame:
                raise ValueError(f"The initial dataset is missing {name!r}")
        if frame[cfg.sensitive_col].isna().any():
            raise ValueError("Every initial sample requires an explicit group label")
        vectors = extract_embeddings(frame, feature_col=cfg.feature_col)
        if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("Embeddings must be nonempty, finite vectors of equal dimension")
        group_labeler = build_group_labeler(frame, cfg.sensitive_col)
        if group_labeler.num_groups() < 2:
            raise ValueError("RAF requires at least two observed groups in the initial dataset")
        cluster_set, labels = fit_clusters(
            vectors, cfg.n_clusters,
            random_state=cfg.random_state if cfg.cluster_random_state is None else cfg.cluster_random_state,
            subsample_ratio=cfg.cluster_subsample_ratio, subsample_size=cfg.cluster_subsample_size,
            assign_batch_size=cfg.cluster_assign_batch_size,
        )
        counts = build_counts_from_labels(labels, frame[cfg.sensitive_col].to_numpy(), group_labeler.values)
        # Empty preclusters still have a row, so a returned cluster ID is always valid.
        if len(counts) < len(cluster_set.centers):
            counts = np.pad(counts, ((0, len(cluster_set.centers) - len(counts)), (0, 0)))
        demand = recompute_q_from_counts(counts)
        major_df, minor_dfs, _ = split_major_minor(frame, cfg.sensitive_col)
        major_labels = cluster_set.assign_clusters(extract_embeddings(major_df, feature_col=cfg.feature_col),
                                                  batch_size=cfg.cluster_assign_batch_size)
        minor_labels = {
            group: cluster_set.assign_clusters(extract_embeddings(df, feature_col=cfg.feature_col),
                                               batch_size=cfg.cluster_assign_batch_size)
            for group, df in minor_dfs.items()
        }
        major_vectors, major_norm_sq, states = build_initial_select_states(
            major_df, minor_dfs, feature_col=cfg.feature_col, metric="euclidean", batch_size=cfg.batch_size,
            major_cluster_labels=major_labels, minor_cluster_labels_by_group=minor_labels,
            use_hnsw=True, single_hnsw_index=cfg.hnsw_single_index,
            delta_capacity=cfg.hnsw_delta_capacity, hnsw_rebuild_delta_threshold=cfg.hnsw_rebuild_delta_threshold,
            hnsw_ef_search=cfg.hnsw_ef_search, hnsw_m=cfg.hnsw_m, hnsw_ef_construction=cfg.hnsw_ef_construction,
        )
        return PreparedState(frame, cluster_set, group_labeler, demand, major_vectors, major_norm_sq, states)

    def run(self, *, query_df: pd.DataFrame, source_manager: SourceManager) -> RAFRunResult:
        """Assemble an augmented dataset; always close source resources on exit."""
        start = time.perf_counter()
        try:
            return self._run(query_df, source_manager, start)
        finally:
            source_manager.close_all()

    def _run(self, query_df: pd.DataFrame, sources: SourceManager, start: float) -> RAFRunResult:
        cfg = self.config
        prepared = self.prepare(query_df)
        costs = sources.costs()
        state = BanditState(len(sources), prepared.demand.size, num_groups=prepared.demand.shape[1])
        policy = FairnessWeightedEpsGreedyPolicy(
            alpha_eps=cfg.fair_alpha_eps, eps_min=cfg.fair_eps_min, kappa=cfg.fair_kappa,
            epsilon_p=cfg.fair_epsilon_p, seed=cfg.random_state,
        )
        result = RAFRunResult(source_sampled={s.name: 0 for s in sources.sources},
                              source_candidates={s.name: 0 for s in sources.sources})
        active_records: dict[tuple[Any, int], dict[str, Any]] = {}
        for _ in range(cfg.max_steps):
            if result.total_cost >= cfg.max_cost:
                result.stop_reason = "budget"
                break
            if cfg.max_accepted is not None and len(active_records) >= cfg.max_accepted:
                result.stop_reason = "retained_target"
                break
            available = np.asarray([not source.exhausted for source in sources.sources], dtype=bool)
            if not available.any():
                result.stop_reason = "sources_exhausted"
                break
            # Enforce the budget before acquiring, including unequal source costs.
            available &= costs <= cfg.max_cost - result.total_cost
            if not available.any():
                result.stop_reason = "budget"
                break
            pending = np.flatnonzero(available & (state.N_i < cfg.policy_warmup_rounds))
            if len(pending):
                pending = pending[state.N_i[pending] == state.N_i[pending].min()]
                index = int(pending[state.t % len(pending)])
            else:
                index = policy.select_source(state=state, q_flat=prepared.demand.reshape(-1), costs=costs,
                                             available_mask=available, current_cost=result.total_cost,
                                             max_cost=cfg.max_cost)
            source = sources.get(index)
            sample = source.sample_one()
            if sample is None:
                source.exhausted = True
                continue
            result.sampled += 1
            result.source_sampled[source.name] += 1
            result.total_cost += float(costs[index])
            if cfg.feature_col not in sample or cfg.sensitive_col not in sample:
                raise ValueError(f"Source {source.name!r} returned a record without the configured embedding/group")
            vector = np.asarray(sample[cfg.feature_col], dtype=np.float32)
            if vector.shape != (prepared.major_vectors.shape[1],) or not np.isfinite(vector).all():
                raise ValueError(f"Source {source.name!r} returned an incompatible embedding")
            group = sample[cfg.sensitive_col]
            group_id = prepared.group_labeler.get_group(sample, cfg.sensitive_col)
            cluster = prepared.cluster_set.assign_cluster(vector)
            combination = None if group_id is None or cluster is None else cluster * prepared.demand.shape[1] + group_id
            state.update_observation(index, combination)
            if group not in prepared.select_states:
                result.skipped_not_minor += 1
                continue
            if group_id is None or cluster is None or prepared.demand[cluster, group_id] <= 0:
                result.skipped_no_demand += 1
                continue
            result.candidates += 1
            result.source_candidates[source.name] += 1
            state.mark_effective(index)
            prepared.demand[cluster, group_id] = max(
                0.0, prepared.demand[cluster, group_id] - cfg.demand_decay_per_candidate
            )
            valuation_start = time.perf_counter()
            decision = try_accept_incremental_hybrid(
                prepared.select_states[group], prepared.major_vectors, vector,
                major_norm_sq=prepared.major_norm_sq, eps=cfg.eps, metric="euclidean",
                neighbor_k=cfg.hybrid_neighbor_k, candidate_cluster=cluster, allowed_clusters=None,
                redundancy_elimination=cfg.raf_redundancy_elimination,
                redundancy_exact_check=cfg.raf_redundancy_exact_check,
                redundancy_tolerance=cfg.raf_redundancy_tolerance,
            )
            result.valuation_s += time.perf_counter() - valuation_start
            if decision.accept:
                result.accepted += 1
                result.total_gain += float(decision.gain)
                record = dict(sample)
                record.setdefault("source_id", source.name)
                active_records[(group, int(decision.accepted_minor_id))] = record
                for identifier in decision.redundant_minor_ids:
                    if active_records.pop((group, int(identifier)), None) is not None:
                        result.removed += 1
            if self.progress_callback is not None:
                self.progress_callback({"sampled": result.sampled, "candidates": result.candidates,
                                        "accepted": result.accepted, "retained": len(active_records),
                                        "total_cost": result.total_cost})
        result.accepted_records = list(active_records.values())
        result.retained = len(result.accepted_records)
        assert result.accepted - result.removed == result.retained
        frames = [prepared.query_df]
        if result.accepted_records:
            frames.append(pd.DataFrame(result.accepted_records))
        result.augmented_df = pd.concat(frames, ignore_index=True)
        result.elapsed_s = time.perf_counter() - start
        return result
