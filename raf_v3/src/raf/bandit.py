"""Demand-aware source selection used by RAF."""
from __future__ import annotations

import abc
import math
from typing import Optional
import numpy as np


class BanditState:
    def __init__(self, num_sources: int, num_combinations: int, num_groups: Optional[int] = None) -> None:
        self.num_sources = num_sources
        self.num_combinations = num_combinations
        self.num_groups = num_groups
        self.N_i = np.zeros((num_sources,), dtype=np.int64)
        self.N_i_p = np.zeros((num_sources, num_combinations), dtype=np.int64)
        self.Y_i = np.zeros((num_sources,), dtype=np.int64)
        self.t = 0

    def update_observation(self, source_idx: int, combination_idx: Optional[int]) -> None:
        self.N_i[source_idx] += 1
        self.t += 1
        if combination_idx is not None and 0 <= combination_idx < self.num_combinations:
            self.N_i_p[source_idx, combination_idx] += 1

    def mark_effective(self, source_idx: int) -> None:
        self.Y_i[source_idx] += 1

    def estimate_p_hat(self) -> np.ndarray:
        return self.N_i_p / np.maximum(1, self.N_i[:, None])

    def estimate_group_p_hat(self) -> np.ndarray:
        if self.num_groups is None or self.num_groups <= 0:
            raise ValueError("num_groups is required for group-level source estimation.")
        if self.num_combinations % self.num_groups != 0:
            raise ValueError("num_combinations must be divisible by num_groups.")
        by_group = self.N_i_p.reshape(self.num_sources, -1, self.num_groups).sum(axis=1)
        return by_group / np.maximum(1, self.N_i[:, None])

    def group_counts(self) -> np.ndarray:
        if self.num_groups is None or self.num_groups <= 0:
            raise ValueError("num_groups is required for group-level source estimation.")
        if self.num_combinations % self.num_groups != 0:
            raise ValueError("num_combinations must be divisible by num_groups.")
        return self.N_i_p.reshape(self.num_sources, -1, self.num_groups).sum(axis=1)


class BanditPolicy(abc.ABC):
    @abc.abstractmethod
    def select_source(
        self,
        *,
        state: BanditState,
        q_flat: np.ndarray,
        costs: np.ndarray,
        available_mask: np.ndarray,
        current_cost: float = 0.0,
        max_cost: Optional[float] = None,
    ) -> int:
        raise NotImplementedError

    def on_observation(self, state: BanditState, source_idx: int) -> None:
        return


class FairnessWeightedEpsGreedyPolicy(BanditPolicy):
    def __init__(
        self,
        *,
        alpha_eps: float = 1.0,
        eps_min: float = 0.1,
        kappa: float = 1.0,
        epsilon_p: float = 1e-6,
        enable_pruning: bool = False,
        prune_budget_interval: float = 0.1,
        prune_fraction: float = 0.1,
        min_active_sources: int = 1,
        prune_use_uniqueness: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.alpha_eps = float(alpha_eps)
        self.eps_min = float(eps_min)
        self.kappa = float(kappa)
        self.epsilon_p = float(epsilon_p)
        self.enable_pruning = bool(enable_pruning)
        self.prune_budget_interval = float(prune_budget_interval)
        self.prune_fraction = float(prune_fraction)
        self.min_active_sources = max(1, int(min_active_sources))
        self.prune_use_uniqueness = bool(prune_use_uniqueness)
        self.disabled = np.zeros(0, dtype=bool)
        self.last_pruned_sources: list[int] = []
        self._next_prune_checkpoint = self.prune_budget_interval
        self.rng = np.random.default_rng(seed)

    def epsilon_t(self, t: int) -> float:
        if t <= 1:
            return 1.0
        return max(self.eps_min, min(1.0, self.alpha_eps * (math.log(t) / t) ** (1.0 / 3.0)))

    def _ensure_capacity(self, num_sources: int) -> None:
        if self.disabled.shape == (num_sources,):
            return
        self.disabled = np.zeros((num_sources,), dtype=bool)

    def _has_unique_combo(
        self,
        *,
        source_idx: int,
        q_flat: np.ndarray,
        p_hat: np.ndarray,
        active_mask: np.ndarray,
    ) -> bool:
        positive = np.flatnonzero(np.asarray(q_flat, dtype=float) > 0)
        if positive.size == 0:
            return False
        for combo_idx in positive.tolist():
            if p_hat[source_idx, combo_idx] <= self.epsilon_p:
                continue
            other_mask = active_mask.copy()
            other_mask[source_idx] = False
            if not np.any(p_hat[other_mask, combo_idx] > self.epsilon_p):
                return True
        return False

    def _should_prune(self, *, current_cost: float, max_cost: Optional[float]) -> bool:
        if not self.enable_pruning:
            return False
        if max_cost is None or max_cost <= 0:
            return False
        if self.prune_budget_interval <= 0 or self.prune_fraction <= 0:
            return False
        budget_ratio = float(current_cost) / float(max_cost)
        if budget_ratio + 1e-12 < self._next_prune_checkpoint:
            return False
        while budget_ratio + 1e-12 >= self._next_prune_checkpoint:
            self._next_prune_checkpoint += self.prune_budget_interval
        return True

    def _apply_pruning(
        self,
        *,
        q_flat: np.ndarray,
        p_hat: np.ndarray,
        available_mask: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        self.last_pruned_sources = []
        active_mask = available_mask & (~self.disabled)
        current_active = int(active_mask.sum())
        if current_active <= self.min_active_sources:
            return

        prune_count = int(math.floor(current_active * self.prune_fraction))
        prune_count = min(prune_count, current_active - self.min_active_sources)
        if prune_count <= 0:
            return

        ranked = np.flatnonzero(active_mask)
        ranked = ranked[np.argsort(scores[ranked], kind="stable")]
        for source_idx in ranked.tolist():
            if len(self.last_pruned_sources) >= prune_count:
                break
            if self.prune_use_uniqueness and self._has_unique_combo(
                source_idx=source_idx,
                q_flat=q_flat,
                p_hat=p_hat,
                active_mask=active_mask,
            ):
                continue
            self.disabled[source_idx] = True
            active_mask[source_idx] = False
            self.last_pruned_sources.append(int(source_idx))

    def select_source(
        self,
        *,
        state: BanditState,
        q_flat: np.ndarray,
        costs: np.ndarray,
        available_mask: np.ndarray,
        current_cost: float = 0.0,
        max_cost: Optional[float] = None,
    ) -> int:
        self._ensure_capacity(state.num_sources)
        candidates = np.flatnonzero(available_mask)
        if len(candidates) == 0:
            raise RuntimeError("No available sources.")

        p_hat = state.estimate_p_hat()
        effective_mask = np.asarray(available_mask, dtype=bool) & (~self.disabled)
        if not np.any(effective_mask):
            effective_mask = np.asarray(available_mask, dtype=bool)

        reward_weights = np.asarray(q_flat, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            active_costs = np.where(effective_mask, costs, np.inf)
            a_hat = np.min(active_costs[:, None] / np.maximum(p_hat, self.epsilon_p), axis=0)
        a_hat[~np.isfinite(a_hat)] = 0.0
        w = reward_weights * a_hat
        s = (p_hat * w[None, :]).sum(axis=1) / np.maximum(costs, 1e-12)

        n_i = np.maximum(1.0, state.N_i.astype(float))
        p_eff_hat = state.Y_i.astype(float) / n_i
        if self.kappa <= 0.0:
            p_eff = p_eff_hat
        else:
            var_hat = p_eff_hat * (1.0 - p_eff_hat) / n_i
            p_eff = p_eff_hat + self.kappa * np.sqrt(np.maximum(0.0, var_hat))
        scores = s * p_eff

        if self._should_prune(current_cost=current_cost, max_cost=max_cost):
            self._apply_pruning(
                q_flat=q_flat,
                p_hat=p_hat,
                available_mask=np.asarray(available_mask, dtype=bool),
                scores=scores,
            )
            effective_mask = np.asarray(available_mask, dtype=bool) & (~self.disabled)
            if not np.any(effective_mask):
                effective_mask = np.asarray(available_mask, dtype=bool)

        candidates = np.flatnonzero(effective_mask)
        if len(candidates) == 0:
            raise RuntimeError("No available sources.")
        if self.rng.random() < self.epsilon_t(max(1, state.t + 1)):
            return int(self.rng.choice(candidates))

        scores[~effective_mask] = -np.inf
        best = np.flatnonzero(scores == np.max(scores))
        return int(self.rng.choice(best))
