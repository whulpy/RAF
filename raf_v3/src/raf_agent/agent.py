from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .connectors import SourceConnector, SourceRegistry
from .embeddings import EmbeddingProvider, RepresentationGuard
from .exceptions import (
    AcquisitionExhaustedError,
    DuplicateSampleError,
    InvalidSampleError,
    MissingRequiredLabelError,
    SampleDownloadError,
    SourceAuthenticationError,
    SourceExhaustedError,
    SourceUnavailableError,
)
from .labels import GroupLabelProvider
from .models import RAFCandidate, RawSample, SourceProfile
from .preprocessing import ImagePreprocessor
from .storage import AcquisitionLog, ContentCache, DedupStore
from .validation import ImageValidator


class ExternalDataAgent:
    """Acquire and prepare data without participating in RAF's decisions."""

    def __init__(
        self,
        *,
        registry: SourceRegistry,
        cache: ContentCache,
        dedup: DedupStore,
        validator: ImageValidator,
        preprocessor: ImagePreprocessor,
        embedding_provider: EmbeddingProvider,
        label_provider: GroupLabelProvider,
        acquisition_log: AcquisitionLog,
        representation_guard: RepresentationGuard | None = None,
        required_group_attributes: list[str] | None = None,
        missing_label_policy: str = "skip",
        max_attempts_per_fetch: int = 12,
    ) -> None:
        if missing_label_policy == "allow":
            missing_label_policy = "keep_unknown"
        if missing_label_policy not in {"skip", "error", "keep_unknown"}:
            raise ValueError("missing_label_policy must be skip, keep_unknown, or error")
        self.registry = registry
        self.cache = cache
        self.dedup = dedup
        self.validator = validator
        self.preprocessor = preprocessor
        self.embedding_provider = embedding_provider
        self.label_provider = label_provider
        self.acquisition_log = acquisition_log
        self.representation_guard = representation_guard
        self.required_group_attributes = list(required_group_attributes or [])
        self.missing_label_policy = missing_label_policy
        self.max_attempts_per_fetch = max_attempts_per_fetch
        self._closed = False

    @classmethod
    def from_config(cls, config_path: str | Path, output_dir: str | Path):
        from .config import build_agent, load_config

        return build_agent(load_config(config_path), output_dir)

    def register_source(self, connector: SourceConnector, *, replace: bool = False) -> None:
        self.registry.register(connector, replace=replace)

    def list_sources(self) -> list[SourceProfile]:
        return self.registry.list()





    def fetch_one(self, source_id: str) -> RAFCandidate:
        """Fetch exactly one valid candidate from the source selected by RAF.

        All retries remain on ``source_id``.  This method has no LLM calls and no
        access to RAF state, group deficits, rewards, or accept/reject decisions.
        """
        connector = self.registry.get(source_id)
        failures: list[str] = []
        for attempt in range(1, self.max_attempts_per_fetch + 1):
            raw: RawSample | None = None
            try:
                raw = connector.fetch_one()
                if raw.source_id != source_id:
                    raise SourceUnavailableError(
                        f"connector boundary violation: requested {source_id}, received {raw.source_id}"
                    )
                if self.dedup.has_external_id(source_id, raw.external_id):
                    raise DuplicateSampleError("duplicate_external_id")
                cached = self.cache.materialize(local_path=raw.local_path, url=raw.content_url)
                if self.dedup.has_content_hash(cached.sha256):
                    raise DuplicateSampleError("duplicate_content_hash")
                validation = self.validator.validate(cached.path)
                if not validation.valid:
                    raise InvalidSampleError(validation.reason or "validation_failed")
                processed_path = self.preprocessor.process(cached.path, cached.sha256)
                labels = self.label_provider.get_labels(raw, processed_path)
                missing = [name for name in self.required_group_attributes if not labels.labels.get(name)]
                if missing and self.missing_label_policy == "error":
                    raise MissingRequiredLabelError(f"missing required labels: {', '.join(missing)}")
                if missing and self.missing_label_policy == "skip":
                    raise SampleDownloadError(f"missing_required_labels:{','.join(missing)}")
                if missing and self.missing_label_policy == "keep_unknown":
                    labels.labels.update({name: "__unknown__" for name in missing})
                embedding = np.asarray(self.embedding_provider.encode(processed_path), dtype=np.float32).reshape(-1)
                if embedding.size != self.embedding_provider.dimension or not np.isfinite(embedding).all():
                    raise SampleDownloadError("invalid_embedding")
                representation = None
                if self.representation_guard is not None:
                    representation = self.representation_guard.validate_and_observe(embedding)
                    if not representation.valid:
                        raise SampleDownloadError(representation.reason or "representation_alignment_failed")
                candidate_id = hashlib.sha256(
                    f"{source_id}\0{raw.external_id}\0{cached.sha256}".encode("utf-8")
                ).hexdigest()
                provenance: dict[str, Any] = {
                    "source_id": source_id,
                    "external_id": raw.external_id,
                    "original_url": raw.content_url,
                    "endpoint": raw.endpoint,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "license": connector.profile.license,
                    "content_sha256": cached.sha256,
                    "connector": connector.profile.connector_type,
                    "connector_type": connector.profile.connector_type,
                    "connector_version": connector.profile.connector_version,
                    "api_endpoint": raw.endpoint,
                    "connector_query": connector.profile.metadata.get("fixed_query"),
                    "preprocessing_signature": self.preprocessor.signature,
                    "embedding_model": self.embedding_provider.name,
                    "labeling": labels.provenance,
                }
                candidate = RAFCandidate(
                    candidate_id=candidate_id,
                    source_id=source_id,
                    external_id=raw.external_id,
                    modality=connector.profile.modality,
                    raw_path=cached.path,
                    processed_path=processed_path,
                    embedding=embedding,
                    group_labels=labels.labels,
                    metadata={
                        "source_metadata": _json_safe(raw.raw_metadata),
                        "validation": validation.details,
                        "representation": representation.details if representation is not None else None,
                    },
                    provenance=provenance,
                )
                self.dedup.add(source_id, raw.external_id, cached.sha256, candidate_id)
                connector.save_state()
                self.acquisition_log.write(
                    "candidate_ready", source_id=source_id, external_id=raw.external_id, candidate_id=candidate_id
                )
                return candidate
            except SourceExhaustedError:
                raise
            except SourceAuthenticationError as exc:
                self.acquisition_log.write("source_error", source_id=source_id, attempt=attempt, reason=str(exc))
                if raw is not None and not connector.profile.requires_auth and raw.content_url != raw.endpoint:
                    # Public dataset APIs may return short-lived signed asset
                    # URLs. A stale cached URL is a bad record, not evidence
                    # that the public source itself now requires credentials.
                    failures.append("expired_or_rejected_public_asset_url")
                    continue
                raise
            except SourceUnavailableError as exc:
                failures.append(str(exc))
                self.acquisition_log.write("source_error", source_id=source_id, attempt=attempt, reason=str(exc))
            except MissingRequiredLabelError as exc:
                self.acquisition_log.write(
                    "skipped",
                    source_id=source_id,
                    external_id=raw.external_id if raw else None,
                    attempt=attempt,
                    reason=str(exc),
                )
                raise
            except (DuplicateSampleError, InvalidSampleError, SampleDownloadError) as exc:
                failures.append(str(exc))
                self.acquisition_log.write(
                    "skipped",
                    source_id=source_id,
                    external_id=raw.external_id if raw else None,
                    attempt=attempt,
                    reason=str(exc),
                )
            finally:
                connector.save_state()
        detail = failures[-1] if failures else "unknown failure"
        raise AcquisitionExhaustedError(
            f"could not prepare a candidate from source {source_id!r} after "
            f"{self.max_attempts_per_fetch} attempts; last error: {detail}"
        )

    def close(self) -> None:
        if self._closed:
            return
        self.registry.close()
        self.dedup.close()
        self._closed = True


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if not isinstance(item, (bytes, bytearray))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value if not isinstance(item, (bytes, bytearray))]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
