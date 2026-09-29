from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..connectors.state import JsonState
from ..models import ValidationResult


class RepresentationGuard:
    """Checks encoder compatibility and monitors centroid drift without RAF state."""

    def __init__(
        self,
        *,
        provider_name: str,
        expected_model: str | None = None,
        expected_dimension: int | None = None,
        require_unit_norm: bool = True,
        norm_tolerance: float = 1e-3,
        reference_centroid: np.ndarray | None = None,
        max_centroid_cosine_distance: float | None = None,
        drift_min_samples: int = 25,
        drift_policy: str = "warn",
        state_path: str | Path | None = None,
    ) -> None:
        if drift_policy not in {"warn", "reject"}:
            raise ValueError("drift_policy must be warn or reject")
        self.provider_name = provider_name
        self.expected_model = expected_model
        self.expected_dimension = expected_dimension
        self.require_unit_norm = require_unit_norm
        self.norm_tolerance = norm_tolerance
        self.reference_centroid = None if reference_centroid is None else np.asarray(reference_centroid, np.float32).reshape(-1)
        self.max_centroid_cosine_distance = max_centroid_cosine_distance
        self.drift_min_samples = drift_min_samples
        self.drift_policy = drift_policy
        self._state_store = JsonState(state_path)
        self._state = self._state_store.load({"count": 0, "sum": []})

    def validate_and_observe(self, vector: np.ndarray) -> ValidationResult:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        details: dict[str, Any] = {"provider": self.provider_name, "dimension": int(vector.size)}
        if self.expected_model and self.provider_name != self.expected_model:
            return ValidationResult(False, "embedding_model_mismatch", details)
        if self.expected_dimension is not None and vector.size != self.expected_dimension:
            return ValidationResult(False, "embedding_dimension_mismatch", details)
        if not np.isfinite(vector).all():
            return ValidationResult(False, "embedding_non_finite", details)
        norm = float(np.linalg.norm(vector))
        details["norm"] = norm
        if self.require_unit_norm and abs(norm - 1.0) > self.norm_tolerance:
            return ValidationResult(False, "embedding_not_normalized", details)
        count = int(self._state.get("count", 0))
        running_sum = np.asarray(self._state.get("sum") or np.zeros(vector.size), dtype=np.float64)
        if running_sum.size != vector.size:
            return ValidationResult(False, "drift_state_dimension_mismatch", details)
        running_sum += vector
        count += 1
        self._state = {"count": count, "sum": running_sum.tolist()}
        self._state_store.save(self._state)
        details["drift_sample_count"] = count
        if self.reference_centroid is not None and count >= self.drift_min_samples:
            centroid = running_sum / count
            denominator = float(np.linalg.norm(centroid) * np.linalg.norm(self.reference_centroid))
            distance = 1.0 if denominator == 0 else 1.0 - float(np.dot(centroid, self.reference_centroid) / denominator)
            details["centroid_cosine_distance"] = distance
            threshold = self.max_centroid_cosine_distance
            if threshold is not None and distance > threshold:
                details["drift_alert"] = True
                if self.drift_policy == "reject":
                    return ValidationResult(False, "embedding_centroid_drift", details)
        return ValidationResult(True, details=details)
