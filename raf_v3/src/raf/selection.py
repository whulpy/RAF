"""RAF MaxSim valuation, posting lists, HNSW search, and redundancy removal."""
from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from itertools import chain
from typing import Any, Optional
import numpy as np
import pandas as pd
from .vector_ops import compute_maxsim, extract_embeddings

HAS_HNSWLIB = importlib.util.find_spec("hnswlib") is not None


@dataclass
class SelectState:
    minor_vectors: np.ndarray
    max_sims: np.ndarray
    argmax_idx: np.ndarray
    total: float
    inverted_lists: list[list[int]]
    pos: np.ndarray
    next_minor_id: int
    initial_minor_count: int
    minor_active: np.ndarray
    major_to_cluster: Optional[np.ndarray] = None
    minor_to_cluster: Optional[np.ndarray] = None
    hnsw_main: Any = None
    hnsw_delta: Any = None
    hnsw_main_deleted: int = 0
    hnsw_delta_deleted: int = 0
    hnsw_main_count: int = 0
    hnsw_main_capacity: Optional[int] = None
    hnsw_single_index: bool = False
    delta_capacity: Optional[int] = None
    hnsw_rebuild_delta_threshold: Optional[int] = None
    minor_seen_stamp: Optional[np.ndarray] = None
    major_seen_stamp: Optional[np.ndarray] = None
    minor_visit_stamp: int = 1
    major_visit_stamp: int = 1
    hnsw_ef_search: Optional[int] = None
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200


@dataclass
class SelectResult:
    accept: bool
    gain: float
    num_winners: int
    reason: str
    total_before: float
    total_after: float
    num_candidates: Optional[int] = None
    profile: Optional[dict[str, float]] = None
    accepted_minor_id: Optional[int] = None
    redundant_minor_ids: tuple[int, ...] = ()


def _init_profile() -> dict[str, float]:
    return {
        "knn": 0.0,
        "knn_main": 0.0,
        "knn_delta": 0.0,
        "cand": 0.0,
        "sim": 0.0,
        "winner": 0.0,
        "update": 0.0,
    }


def split_major_minor(df: pd.DataFrame, sensitive_col: str) -> tuple[pd.DataFrame, dict[object, pd.DataFrame], object]:
    counts = df[sensitive_col].value_counts(dropna=True)
    if counts.empty:
        raise ValueError(f"No valid values found in '{sensitive_col}'.")
    major_value = counts.idxmax()
    major_df = df[df[sensitive_col] == major_value].copy().reset_index(drop=True)
    minor_dfs = {
        value: df[df[sensitive_col] == value].copy().reset_index(drop=True)
        for value in counts.index
        if value != major_value
    }
    return major_df, minor_dfs, major_value


def _build_inv_pos(argmax_idx: np.ndarray, minor_count: int) -> tuple[list[list[int]], np.ndarray]:
    inv = [[] for _ in range(minor_count)]
    pos = np.full((argmax_idx.shape[0],), -1, dtype=np.int64)
    for major_id, minor_id in enumerate(argmax_idx.tolist()):
        if minor_id < 0:
            continue
        pos[major_id] = len(inv[minor_id])
        inv[minor_id].append(int(major_id))
    return inv, pos


def _active_minor_vectors(state: SelectState) -> np.ndarray:
    active = int(state.next_minor_id)
    if active <= 0:
        return np.empty((0, state.minor_vectors.shape[1]), dtype=np.float32)
    return np.asarray(state.minor_vectors[:active], dtype=np.float32)


def _active_minor_clusters(state: SelectState) -> Optional[np.ndarray]:
    if state.minor_to_cluster is None:
        return None
    active = int(state.next_minor_id)
    return np.asarray(state.minor_to_cluster[:active], dtype=np.int32)


def _ensure_minor_capacity(state: SelectState, min_capacity: int) -> None:
    current_capacity = int(state.minor_vectors.shape[0])
    if current_capacity >= int(min_capacity):
        return
    new_capacity = max(current_capacity * 2, int(min_capacity), current_capacity + 256, 256)
    new_minor_vectors = np.empty((new_capacity, state.minor_vectors.shape[1]), dtype=np.float32)
    if state.next_minor_id > 0:
        new_minor_vectors[: state.next_minor_id] = state.minor_vectors[: state.next_minor_id]
    state.minor_vectors = new_minor_vectors
    new_minor_active = np.zeros((new_capacity,), dtype=bool)
    active_limit = min(int(state.next_minor_id), int(state.minor_active.shape[0]))
    if active_limit > 0:
        new_minor_active[:active_limit] = state.minor_active[:active_limit]
    state.minor_active = new_minor_active
    if state.minor_to_cluster is not None:
        new_minor_to_cluster = np.full((new_capacity,), -1, dtype=np.int32)
        if state.next_minor_id > 0:
            new_minor_to_cluster[: state.next_minor_id] = state.minor_to_cluster[: state.next_minor_id]
        state.minor_to_cluster = new_minor_to_cluster
    if state.minor_seen_stamp is not None:
        new_minor_seen_stamp = np.zeros((new_capacity,), dtype=np.int32)
        limit = min(int(state.next_minor_id), int(state.minor_seen_stamp.shape[0]))
        if limit > 0:
            new_minor_seen_stamp[:limit] = state.minor_seen_stamp[:limit]
        state.minor_seen_stamp = new_minor_seen_stamp


def _swap_delete(inv: list[list[int]], pos: np.ndarray, bucket_id: int, major_id: int) -> None:
    idx = int(pos[major_id])
    if idx < 0:
        return
    bucket = inv[bucket_id]
    last_major = bucket[-1]
    bucket[idx] = last_major
    bucket.pop()
    pos[last_major] = idx
    pos[major_id] = -1


def _append_to_bucket(inv: list[list[int]], pos: np.ndarray, bucket_id: int, major_id: int) -> None:
    pos[major_id] = len(inv[bucket_id])
    inv[bucket_id].append(int(major_id))


def _ensure_hnswlib() -> Any:
    if not HAS_HNSWLIB:
        raise ImportError("hnswlib is required for incremental_hybrid mode.")
    import hnswlib  # type: ignore

    return hnswlib


def _build_hnsw_index(
    vectors: np.ndarray,
    *,
    ids: Optional[np.ndarray] = None,
    ef_search: Optional[int] = None,
    m: int = 16,
    ef_construction: int = 200,
    max_elements: Optional[int] = None,
) -> Any:
    if vectors.size == 0:
        return None
    hnswlib = _ensure_hnswlib()
    index = hnswlib.Index(space="l2", dim=vectors.shape[1])
    index.init_index(
        max_elements=int(max_elements) if max_elements is not None else int(vectors.shape[0]),
        ef_construction=int(ef_construction),
        M=int(m),
    )
    if ef_search is not None:
        index.set_ef(int(ef_search))
    if ids is None:
        ids = np.arange(vectors.shape[0], dtype=np.int64)
    index.add_items(vectors.astype(np.float32, copy=False), ids.astype(np.int64, copy=False))
    return index


def _initial_main_capacity(initial_count: int, delta_capacity: Optional[int], single_index: bool) -> int:
    initial_count = int(initial_count)
    if single_index:
        growth = int(delta_capacity) if delta_capacity is not None and int(delta_capacity) > 0 else max(initial_count, 2048)
        return max(initial_count + growth, initial_count * 2, initial_count + 256, 256)
    return max(initial_count, 1)


def _create_empty_hnsw_index(
    dim: int,
    *,
    max_elements: int,
    ef_search: Optional[int] = None,
    m: int = 16,
    ef_construction: int = 200,
) -> Any:
    hnswlib = _ensure_hnswlib()
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=int(max_elements), ef_construction=int(ef_construction), M=int(m))
    if ef_search is not None:
        index.set_ef(int(ef_search))
    return index


def build_initial_select_states(major_df: pd.DataFrame, minor_dfs: dict[object, pd.DataFrame], *, feature_col: Optional[str], metric: str, batch_size: int, major_cluster_labels: Optional[np.ndarray]=None, minor_cluster_labels_by_group: Optional[dict[object, np.ndarray]]=None, use_hnsw: bool=False, single_hnsw_index: bool=False, delta_capacity: Optional[int]=None, hnsw_rebuild_delta_threshold: Optional[int]=None, hnsw_ef_search: Optional[int]=60, hnsw_m: int=16, hnsw_ef_construction: int=200) -> tuple[np.ndarray, np.ndarray, dict[object, SelectState]]:
    major_vectors = extract_embeddings(major_df, feature_col=feature_col)
    major_norm_sq = np.sum(major_vectors * major_vectors, axis=1, dtype=np.float32)
    states: dict[object, SelectState] = {}
    for group_value, minor_df in minor_dfs.items():
        minor_vectors = extract_embeddings(minor_df, feature_col=feature_col)
        maxsim = compute_maxsim(major_vectors, minor_vectors, metric=metric, batch_size=batch_size, return_argmax=True)
        max_sims = np.asarray(maxsim['max_sims'], dtype=np.float32)
        argmax_idx = np.asarray(maxsim['argmax_idx'], dtype=np.int64)
        initial_total = float(maxsim['total'])
        inv, pos = _build_inv_pos(argmax_idx, minor_vectors.shape[0])
        hnsw_main = None
        hnsw_delta = None
        if use_hnsw:
            main_capacity = _initial_main_capacity(minor_vectors.shape[0], delta_capacity, single_hnsw_index)
            hnsw_main = _build_hnsw_index(minor_vectors, ef_search=hnsw_ef_search, m=hnsw_m, ef_construction=hnsw_ef_construction, max_elements=main_capacity)
            if not single_hnsw_index and delta_capacity is not None and (delta_capacity > 0) and (minor_vectors.shape[1] > 0):
                hnsw_delta = _create_empty_hnsw_index(minor_vectors.shape[1], max_elements=delta_capacity, ef_search=hnsw_ef_search, m=hnsw_m, ef_construction=hnsw_ef_construction)
        states[group_value] = SelectState(minor_vectors=minor_vectors, max_sims=max_sims, argmax_idx=argmax_idx, total=initial_total, inverted_lists=inv, pos=pos, next_minor_id=int(minor_vectors.shape[0]), initial_minor_count=int(minor_vectors.shape[0]), minor_active=np.ones((minor_vectors.shape[0],), dtype=bool), major_to_cluster=None if major_cluster_labels is None else np.asarray(major_cluster_labels, dtype=np.int32), minor_to_cluster=None if minor_cluster_labels_by_group is None else np.asarray(minor_cluster_labels_by_group.get(group_value, np.full((minor_vectors.shape[0],), -1, dtype=np.int32)), dtype=np.int32), hnsw_main=hnsw_main, hnsw_delta=hnsw_delta, hnsw_main_count=int(minor_vectors.shape[0]) if hnsw_main is not None else 0, hnsw_main_capacity=main_capacity if use_hnsw else None, hnsw_single_index=bool(single_hnsw_index), delta_capacity=delta_capacity, hnsw_rebuild_delta_threshold=hnsw_rebuild_delta_threshold, minor_seen_stamp=None if single_hnsw_index else np.zeros((minor_vectors.shape[0],), dtype=np.int32), major_seen_stamp=None if single_hnsw_index else np.zeros((major_vectors.shape[0],), dtype=np.int32), minor_visit_stamp=1, major_visit_stamp=1, hnsw_ef_search=hnsw_ef_search, hnsw_m=hnsw_m, hnsw_ef_construction=hnsw_ef_construction)
    return (major_vectors, major_norm_sq, states)


def _sim_to_candidate(
    major_vectors: np.ndarray,
    candidate_vector: np.ndarray,
    *,
    major_norm_sq: Optional[np.ndarray] = None,
) -> np.ndarray:
    candidate_vector = np.asarray(candidate_vector, dtype=np.float32)
    candidate_norm = float(np.dot(candidate_vector, candidate_vector))
    if major_norm_sq is None:
        major_norm = np.sum(major_vectors * major_vectors, axis=1, dtype=np.float32)
    else:
        major_norm = np.asarray(major_norm_sq, dtype=np.float32)
    return -(major_norm + candidate_norm - 2.0 * (major_vectors @ candidate_vector))


def _ensure_delta_capacity(state: SelectState) -> None:
    if state.hnsw_delta is None or state.delta_capacity is None:
        return
    current = int(getattr(state.hnsw_delta, "get_current_count", lambda: 0)())
    if current < int(state.delta_capacity):
        return
    new_capacity = max(int(state.delta_capacity) * 2, int(state.delta_capacity) + 1)
    resize = getattr(state.hnsw_delta, "resize_index", None)
    if resize is None:
        state.hnsw_delta = None
        return
    resize(int(new_capacity))
    state.delta_capacity = int(new_capacity)


def _ensure_hnsw_main_capacity(state: SelectState) -> None:
    if state.hnsw_main is None or state.hnsw_main_capacity is None:
        return
    current = int(state.next_minor_id)
    if current < int(state.hnsw_main_capacity):
        return
    new_capacity = max(int(state.hnsw_main_capacity) * 2, int(state.hnsw_main_capacity) + 1, current + 1)
    resize = getattr(state.hnsw_main, "resize_index", None)
    if resize is None:
        _rebuild_hnsw_main_from_active_minors(state)
        state.hnsw_main_capacity = max(new_capacity, int(state.next_minor_id))
        return
    resize(int(new_capacity))
    state.hnsw_main_capacity = int(new_capacity)


def _rebuild_hnsw_main_from_active_minors(state: SelectState) -> None:
    if state.hnsw_main is None:
        return
    active_ids = np.flatnonzero(
        np.asarray(state.minor_active[: state.next_minor_id], dtype=bool)
    ).astype(np.int64)
    if active_ids.size == 0:
        state.hnsw_main = None
        state.hnsw_delta = None
        state.hnsw_main_count = 0
        return
    active_minor_vectors = np.asarray(state.minor_vectors[active_ids], dtype=np.float32)
    active_count = int(active_ids.shape[0])
    if state.hnsw_main_capacity is None:
        state.hnsw_main_capacity = max(
            active_count,
            active_count + (
                int(state.delta_capacity)
                if state.delta_capacity is not None and int(state.delta_capacity) > 0
                else max(active_count, 2048)
            ),
            active_count * 2,
            256,
        )
    elif int(state.hnsw_main_capacity) < active_count:
        state.hnsw_main_capacity = max(
            active_count,
            int(state.hnsw_main_capacity) * 2,
            active_count + 256,
        )
    state.hnsw_main = _build_hnsw_index(
        active_minor_vectors,
        ids=active_ids,
        ef_search=state.hnsw_ef_search,
        m=state.hnsw_m,
        ef_construction=state.hnsw_ef_construction,
        max_elements=state.hnsw_main_capacity,
    )
    state.hnsw_main_deleted = 0
    state.hnsw_delta_deleted = 0
    # This is the ID boundary between the rebuilt main index and the new delta,
    # not the number of live elements in the main index.
    state.hnsw_main_count = int(state.next_minor_id)
    if (not state.hnsw_single_index) and state.delta_capacity is not None and state.delta_capacity > 0:
        state.hnsw_delta = _create_empty_hnsw_index(
            active_minor_vectors.shape[1],
            max_elements=state.delta_capacity,
            ef_search=state.hnsw_ef_search,
            m=state.hnsw_m,
            ef_construction=state.hnsw_ef_construction,
        )
    else:
        state.hnsw_delta = None


def _maybe_rebuild_hnsw_indexes(state: SelectState) -> None:
    if state.hnsw_single_index:
        return
    threshold = state.hnsw_rebuild_delta_threshold
    if state.hnsw_main is None or state.hnsw_delta is None or threshold is None or int(threshold) <= 0:
        return
    delta_count = int(getattr(state.hnsw_delta, "get_current_count", lambda: 0)())
    if delta_count < int(threshold):
        return
    _rebuild_hnsw_main_from_active_minors(state)


def _add_candidate_to_delta_index(state: SelectState, candidate_vector: np.ndarray, candidate_id: int) -> None:
    if state.hnsw_single_index:
        if state.hnsw_main is None:
            return
        _ensure_hnsw_main_capacity(state)
        state.hnsw_main.add_items(
            np.asarray(candidate_vector, dtype=np.float32)[None, :],
            np.asarray([candidate_id], dtype=np.int64),
        )
        state.hnsw_main_count = max(int(state.hnsw_main_count), int(candidate_id) + 1)
        return
    if state.hnsw_delta is None:
        return
    _ensure_delta_capacity(state)
    if state.hnsw_delta is None:
        return
    state.hnsw_delta.add_items(
        np.asarray(candidate_vector, dtype=np.float32)[None, :],
        np.asarray([candidate_id], dtype=np.int64),
    )
    _maybe_rebuild_hnsw_indexes(state)


def _append_candidate_to_state(
    state: SelectState,
    candidate_vector: np.ndarray,
    winner_ids: np.ndarray,
    winner_sims: np.ndarray,
    *,
    candidate_cluster: Optional[int] = None,
) -> tuple[int, tuple[int, ...]]:
    new_id = int(state.next_minor_id)
    _ensure_minor_capacity(state, new_id + 1)
    state.minor_vectors[new_id] = np.asarray(candidate_vector, dtype=np.float32)
    state.minor_active[new_id] = True
    state.inverted_lists.append([])
    if state.minor_to_cluster is not None:
        append_cluster = -1 if candidate_cluster is None else int(candidate_cluster)
        state.minor_to_cluster[new_id] = append_cluster
    touched_old_ids: set[int] = set()
    for major_id in winner_ids.tolist():
        old_minor = int(state.argmax_idx[major_id])
        if old_minor >= 0:
            touched_old_ids.add(old_minor)
            _swap_delete(state.inverted_lists, state.pos, old_minor, major_id)
        _append_to_bucket(state.inverted_lists, state.pos, new_id, major_id)
    if winner_ids.size > 0:
        state.max_sims[winner_ids] = winner_sims
        state.argmax_idx[winner_ids] = new_id
    state.next_minor_id = new_id + 1
    # A rebuild triggered by this insertion must include the newly appended ID.
    _add_candidate_to_delta_index(state, candidate_vector, new_id)
    return new_id, tuple(sorted(touched_old_ids))


def _mark_minor_inactive(state: SelectState, minor_id: int) -> None:
    minor_id = int(minor_id)
    if minor_id < 0 or minor_id >= int(state.next_minor_id):
        return
    if not state.minor_active[minor_id]:
        return
    state.minor_active[minor_id] = False
    # A label may live in either index depending on whether a delta rebuild has
    # occurred. Trying both keeps stable IDs without online array compaction.
    for index, counter in ((state.hnsw_main, "hnsw_main_deleted"), (state.hnsw_delta, "hnsw_delta_deleted")):
        if index is None:
            continue
        mark_deleted = getattr(index, "mark_deleted", None)
        if mark_deleted is None:
            continue
        try:
            mark_deleted(minor_id)
            setattr(state, counter, getattr(state, counter) + 1)
        except RuntimeError:
            pass


def _retire_empty_posting_minors(
    state: SelectState,
    major_vectors: np.ndarray,
    touched_minor_ids: tuple[int, ...],
    *,
    major_norm_sq: Optional[np.ndarray],
    tolerance: float,
    exact_check: bool,
) -> tuple[tuple[int, ...], dict[str, float]]:
    profile = {
        "redundancy_candidates": 0.0,
        "redundancy_exact_checks": 0.0,
        "redundancy_removed": 0.0,
        "redundancy_check_s": 0.0,
    }
    candidates = [
        int(minor_id)
        for minor_id in touched_minor_ids
        if int(minor_id) >= int(state.initial_minor_count)
        and int(minor_id) < int(state.next_minor_id)
        and bool(state.minor_active[int(minor_id)])
        and len(state.inverted_lists[int(minor_id)]) == 0
    ]
    profile["redundancy_candidates"] = float(len(candidates))
    if not candidates:
        return (), profile

    check_start = time.perf_counter()
    retired: list[int] = []
    for minor_id in candidates:
        safe_to_remove = True
        if exact_check:
            profile["redundancy_exact_checks"] += 1.0
            sims = _sim_to_candidate(
                major_vectors,
                state.minor_vectors[minor_id],
                major_norm_sq=major_norm_sq,
            )
            safe_to_remove = not bool(
                np.any(sims > (state.max_sims + float(tolerance)))
            )
        if safe_to_remove:
            _mark_minor_inactive(state, minor_id)
            retired.append(minor_id)
    profile["redundancy_removed"] = float(len(retired))
    profile["redundancy_check_s"] = float(time.perf_counter() - check_start)
    return tuple(retired), profile


def _query_neighbors_from_hnsw(index: Any, candidate_vector: np.ndarray, k: int,
                               deleted_count: int = 0) -> list[int]:
    if index is None or k <= 0:
        return []
    current = int(getattr(index, "get_current_count", lambda: 0)())
    # hnswlib's current count includes labels marked as deleted.
    current -= int(deleted_count)
    if current <= 0:
        return []
    k = min(int(k), current)
    labels, _ = index.knn_query(np.asarray(candidate_vector, dtype=np.float32)[None, :], k=k)
    return [int(x) for x in labels[0].tolist()]


def _query_neighbors_from_vector_slice(
    vectors: np.ndarray,
    candidate_vector: np.ndarray,
    *,
    start_id: int,
    k: int,
) -> list[int]:
    if vectors.size == 0 or k <= 0:
        return []
    candidate_vector = np.asarray(candidate_vector, dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)
    k = min(int(k), int(vectors.shape[0]))
    if k <= 0:
        return []
    dists = np.sum((vectors - candidate_vector[None, :]) ** 2, axis=1, dtype=np.float32)
    if k >= int(vectors.shape[0]):
        order = np.argsort(dists)
    else:
        top_idx = np.argpartition(dists, kth=k - 1)[:k]
        order = top_idx[np.argsort(dists[top_idx])]
    return [int(start_id + idx) for idx in order.tolist()]


def _flatten_neighbor_buckets(
    state: SelectState,
    neighbors: list[int],
    *,
    allowed_clusters: Optional[set[int]] = None,
) -> np.ndarray:
    valid_neighbors = [
        int(minor_id)
        for minor_id in neighbors
        if 0 <= int(minor_id) < len(state.inverted_lists)
        and int(minor_id) < int(state.minor_active.shape[0])
        and bool(state.minor_active[int(minor_id)])
    ]
    if not valid_neighbors:
        return np.zeros((0,), dtype=np.int64)

    if allowed_clusters is None or state.major_to_cluster is None:
        total_len = sum(len(state.inverted_lists[minor_id]) for minor_id in valid_neighbors)
        if total_len <= 0:
            return np.zeros((0,), dtype=np.int64)
        # Each major point belongs to exactly one owner bucket, so a direct flatten is enough.
        return np.fromiter(
            chain.from_iterable(state.inverted_lists[minor_id] for minor_id in valid_neighbors),
            dtype=np.int64,
            count=total_len,
        )

    major_clusters = state.major_to_cluster
    filtered = [
        int(major_id)
        for minor_id in valid_neighbors
        for major_id in state.inverted_lists[minor_id]
        if int(major_clusters[int(major_id)]) in allowed_clusters
    ]
    if not filtered:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(filtered, dtype=np.int64)


def _unique_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(int(value))
        out.append(int(value))
    return out


def _advance_stamp(buffer: Optional[np.ndarray], current_stamp: int) -> int:
    if buffer is None:
        return int(current_stamp)
    next_stamp = int(current_stamp) + 1
    if next_stamp >= np.iinfo(buffer.dtype).max:
        buffer.fill(0)
        next_stamp = 1
    return next_stamp


def _hybrid_major_candidates(
    state: SelectState,
    candidate_vector: np.ndarray,
    *,
    neighbor_k: int,
    allowed_clusters: Optional[set[int]],
) -> tuple[np.ndarray, list[int], dict[str, float]]:
    profile = _init_profile()
    if state.hnsw_single_index:
        t0_main = time.perf_counter()
        neighbors = _query_neighbors_from_hnsw(
            state.hnsw_main, candidate_vector, neighbor_k,
            state.hnsw_main_deleted,
        )
        profile["knn_main"] = time.perf_counter() - t0_main
        profile["knn"] = profile["knn_main"]
        t0_cand = time.perf_counter()
        if not neighbors:
            profile["cand"] = time.perf_counter() - t0_cand
            return np.zeros((0,), dtype=np.int64), [], profile
        cand_arr = _flatten_neighbor_buckets(
            state,
            [int(minor_id) for minor_id in neighbors],
            allowed_clusters=allowed_clusters,
        )
        profile["cand"] = time.perf_counter() - t0_cand
        if cand_arr.size == 0:
            return np.zeros((0,), dtype=np.int64), [int(n) for n in neighbors], profile
        return cand_arr, [int(n) for n in neighbors], profile

    t0_main = time.perf_counter()
    main_neighbors = _query_neighbors_from_hnsw(
        state.hnsw_main, candidate_vector, neighbor_k,
        state.hnsw_main_deleted,
    )
    profile["knn_main"] = time.perf_counter() - t0_main

    t0_delta = time.perf_counter()
    try:
        delta_neighbors = _query_neighbors_from_hnsw(
            state.hnsw_delta, candidate_vector, neighbor_k,
            state.hnsw_delta_deleted,
        )
    except RuntimeError:
        delta_start = int(state.hnsw_main_count)
        delta_end = int(state.next_minor_id)
        delta_neighbors = _query_neighbors_from_vector_slice(
            state.minor_vectors[delta_start:delta_end],
            candidate_vector,
            start_id=delta_start,
            k=neighbor_k,
        )
    profile["knn_delta"] = time.perf_counter() - t0_delta
    profile["knn"] = profile["knn_main"] + profile["knn_delta"]

    t0_cand = time.perf_counter()
    state.minor_visit_stamp = _advance_stamp(state.minor_seen_stamp, state.minor_visit_stamp)
    state.major_visit_stamp = _advance_stamp(state.major_seen_stamp, state.major_visit_stamp)
    minor_stamp = int(state.minor_visit_stamp)
    major_stamp = int(state.major_visit_stamp)

    neighbors: list[int] = []
    for minor_id in main_neighbors + delta_neighbors:
        minor_id = int(minor_id)
        if minor_id < 0 or minor_id >= int(state.next_minor_id):
            continue
        if minor_id >= int(state.minor_active.shape[0]) or not bool(state.minor_active[minor_id]):
            continue
        if state.minor_seen_stamp is not None:
            if state.minor_seen_stamp[minor_id] == minor_stamp:
                continue
            state.minor_seen_stamp[minor_id] = minor_stamp
        neighbors.append(minor_id)
    if not neighbors:
        profile["cand"] = time.perf_counter() - t0_cand
        return np.zeros((0,), dtype=np.int64), neighbors, profile

    candidates: list[int] = []
    major_clusters = state.major_to_cluster
    for minor_id in neighbors:
        if 0 <= minor_id < len(state.inverted_lists):
            for major_id in state.inverted_lists[minor_id]:
                major_id = int(major_id)
                if state.major_seen_stamp is not None:
                    if state.major_seen_stamp[major_id] == major_stamp:
                        continue
                    state.major_seen_stamp[major_id] = major_stamp
                if allowed_clusters is not None and major_clusters is not None:
                    if int(major_clusters[major_id]) not in allowed_clusters:
                        continue
                candidates.append(major_id)
    profile["cand"] = time.perf_counter() - t0_cand
    return np.asarray(candidates, dtype=np.int64), neighbors, profile


def try_accept_incremental_hybrid(
    state: SelectState,
    major_vectors: np.ndarray,
    candidate_vector: np.ndarray,
    *,
    major_norm_sq: Optional[np.ndarray] = None,
    eps: float = 0.0,
    metric: str = "euclidean",
    neighbor_k: int = 20,
    candidate_cluster: Optional[int] = None,
    allowed_clusters: Optional[set[int]] = None,
    redundancy_elimination: bool = False,
    redundancy_exact_check: bool = True,
    redundancy_tolerance: float = 1e-8,
) -> SelectResult:
    profile = _init_profile()
    if major_vectors.size == 0:
        return SelectResult(False, 0.0, 0, "empty major set", state.total, state.total, num_candidates=0, profile=profile)
    if metric.strip().lower() != "euclidean":
        raise ValueError("Only 'euclidean' metric is currently supported.")
    candidate_vector = np.asarray(candidate_vector, dtype=np.float32)
    cand_arr, neighbors, hybrid_profile = _hybrid_major_candidates(
        state,
        candidate_vector,
        neighbor_k=neighbor_k,
        allowed_clusters=allowed_clusters,
    )
    for key, value in hybrid_profile.items():
        profile[key] += float(value)
    if len(neighbors) == 0:
        return SelectResult(False, 0.0, 0, "no neighbors", state.total, state.total, num_candidates=0, profile=profile)
    if cand_arr.size == 0:
        return SelectResult(False, 0.0, 0, "no candidates", state.total, state.total, num_candidates=0, profile=profile)
    t0_sim = time.perf_counter()
    sims = _sim_to_candidate(
        major_vectors[cand_arr],
        candidate_vector,
        major_norm_sq=None if major_norm_sq is None else np.asarray(major_norm_sq, dtype=np.float32)[cand_arr],
    )
    profile["sim"] += time.perf_counter() - t0_sim
    base = state.max_sims[cand_arr]
    t0_winner = time.perf_counter()
    winners_local = sims > (base + float(eps))
    profile["winner"] += time.perf_counter() - t0_winner
    if not np.any(winners_local):
        return SelectResult(False, 0.0, 0, "no improvement", state.total, state.total, num_candidates=int(cand_arr.shape[0]), profile=profile)
    winner_ids = cand_arr[winners_local]
    total_after = float(state.total + np.sum(sims[winners_local] - base[winners_local], dtype=np.float64))
    gain = total_after - state.total
    t0_update = time.perf_counter()
    new_id, touched_old_ids = _append_candidate_to_state(
        state,
        candidate_vector,
        winner_ids.astype(np.int64),
        sims[winners_local].astype(np.float32),
        candidate_cluster=candidate_cluster,
    )
    profile["update"] += time.perf_counter() - t0_update
    state.total = total_after
    redundant_minor_ids: tuple[int, ...] = ()
    if redundancy_elimination:
        redundant_minor_ids, redundancy_profile = _retire_empty_posting_minors(
            state,
            major_vectors,
            touched_old_ids,
            major_norm_sq=major_norm_sq,
            tolerance=float(redundancy_tolerance),
            exact_check=bool(redundancy_exact_check),
        )
        profile.update(redundancy_profile)
        profile["redundancy_gross_accepted"] = 1.0
    return SelectResult(
        True,
        gain,
        int(winner_ids.shape[0]),
        "accepted",
        total_after - gain,
        total_after,
        num_candidates=int(cand_arr.shape[0]),
        profile=profile,
        accepted_minor_id=new_id,
        redundant_minor_ids=redundant_minor_ids,
    )
