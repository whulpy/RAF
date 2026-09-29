"""Configure acquisition from existing image sources, without setup workflows."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from .agent import ExternalDataAgent
from .connectors import GenericRESTConnector, LocalFolderConnector, SourceRegistry
from .connectors.http import RobustHttpClient
from .embeddings import OpenCLIPEmbeddingProvider, RepresentationGuard
from .exceptions import ConfigurationError
from .labels import ManifestLabelProvider, PerSourceMetadataLabelProvider, SourceMetadataLabelProvider
from .models import SourceProfile
from .preprocessing import ImagePreprocessor
from .storage import AcquisitionLog, ContentCache, DedupStore
from .validation import ImageValidator


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    config = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ConfigurationError("Agent configuration must be a mapping")
    config["_config_path"] = str(path)
    return config


def build_agent(config: dict[str, Any], output_dir: str | Path) -> ExternalDataAgent:
    base = Path(config.get("_config_path", ".")).parent
    output = Path(output_dir).resolve()
    resolve = lambda value: (base / Path(value).expanduser()).resolve()
    request = config.get("request", {})
    http = RobustHttpClient(timeout=float(request.get("timeout_seconds", 30)),
                            retries=int(request.get("max_retries", 4)),
                            backoff=float(request.get("backoff_seconds", 0.5)),
                            min_interval=float(request.get("min_interval_seconds", 0)))
    registry = SourceRegistry()
    for item in config.get("sources", []):
        identifier = str(item["source_id"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier) or identifier in {".", ".."}:
            raise ConfigurationError("source_id must contain only letters, digits, dots, underscores, or hyphens")
        kind = item["type"]
        profile = SourceProfile(source_id=identifier, name=item.get("name", identifier),
                                connector_type=kind, modality="image", domain=item.get("domain", "unspecified"),
                                license=item.get("license"), homepage=item.get("homepage"),
                                group_attributes=list(item.get("group_attributes", [])),
                                requires_auth=bool(item.get("auth_env")),
                                metadata={"fixed_query": dict(item.get("params", {}))})
        state_path = output / "state" / "sources" / f"{identifier}.json"
        if kind == "local_folder":
            connector = LocalFolderConnector(profile, resolve(item["root"]),
                metadata_csv=resolve(item["metadata_csv"]) if item.get("metadata_csv") else None,
                path_column=item.get("path_column", "path"), id_column=item.get("id_column", "id"),
                seed=int(item.get("seed", config.get("random_seed", 42))), state_path=state_path)
        elif kind == "generic_rest":
            headers = item.get("headers", {})
            protected = {"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie"}
            if any(key.lower() in protected and value for key, value in headers.items()):
                raise ConfigurationError("Put credential environment-variable names in auth_env, not secret values in headers")
            connector = GenericRESTConnector(profile, item["endpoint"], method=item.get("method", "GET"),
                items_path=item.get("items_path", "rows"), id_path=item.get("id_path", "id"),
                url_path=item.get("url_path", "image_url"), metadata_path=item.get("metadata_path", "metadata"),
                params=item.get("params"), headers=headers, auth_env=item.get("auth_env"),
                pagination=item.get("pagination"), state_path=state_path, http=http)
        else:
            raise ConfigurationError(f"Unsupported runtime connector: {kind}")
        registry.register(connector)
    if not registry.list():
        raise ConfigurationError("Configure at least one existing image source")

    embedding_cfg = config.get("embedding", {})
    if embedding_cfg.get("type") != "clip" or not embedding_cfg.get("model"):
        raise ConfigurationError("Configure embedding.type=clip and the same model used for the initial dataset")
    embedding = OpenCLIPEmbeddingProvider(embedding_cfg["model"], device=embedding_cfg.get("device"),
                                          batch_size=int(embedding_cfg.get("batch_size", 16)))
    label_cfg = config.get("labels", {})
    if label_cfg.get("type") == "source_metadata":
        labels = SourceMetadataLabelProvider(label_cfg.get("mappings", {}))
    elif label_cfg.get("type") == "per_source_metadata":
        labels = PerSourceMetadataLabelProvider(label_cfg.get("mappings", {}))
    elif label_cfg.get("type") == "manifest":
        labels = ManifestLabelProvider(resolve(label_cfg["path"]),
            id_column=label_cfg.get("id_column", "external_id"), label_columns=label_cfg.get("label_columns"))
    else:
        raise ConfigurationError("Use source_metadata, per_source_metadata, or manifest labels")
    validation = config.get("validation", {})
    preprocessing = config.get("preprocessing", {})
    alignment = config.get("alignment", {})
    guard = RepresentationGuard(provider_name=embedding.name,
        expected_model=alignment.get("expected_model", embedding.name),
        expected_dimension=alignment.get("expected_dimension"),
        require_unit_norm=bool(alignment.get("require_unit_norm", True)),
        norm_tolerance=float(alignment.get("norm_tolerance", 1e-3)),
        state_path=output / "state" / "representation.json")
    runtime = config.get("runtime", {})
    attempts = int(runtime.get("max_attempts_per_fetch", 12))
    if attempts <= 0:
        raise ConfigurationError("max_attempts_per_fetch must be positive")
    return ExternalDataAgent(registry=registry,
        cache=ContentCache(output / "images" / "raw", max_bytes=int(validation.get("max_bytes", 30 * 1024 * 1024)), http=http),
        dedup=DedupStore(output / "state" / "dedup.sqlite3"),
        validator=ImageValidator(min_width=int(validation.get("min_width", 32)),
                                 min_height=int(validation.get("min_height", 32))),
        preprocessor=ImagePreprocessor(output / "images" / "processed", size=tuple(preprocessing.get("size", [224, 224])),
                                       mode=preprocessing.get("mode", "center_crop")),
        embedding_provider=embedding, label_provider=labels,
        acquisition_log=AcquisitionLog(output / "logs" / "acquisition.jsonl"), representation_guard=guard,
        required_group_attributes=list(label_cfg.get("required", ["group"])), missing_label_policy="skip",
        max_attempts_per_fetch=attempts)
