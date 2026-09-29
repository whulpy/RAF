"""Command-line entry point for one RAF assembly run."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAFConfig
from .data import load_dataset, load_sources
from .pipeline import RAFPipeline


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _parquet_frame(frame: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    frame = frame.copy()
    # Keep rich provenance portable across heterogeneous external sources.
    for name in frame.columns:
        if name != feature_col and any(isinstance(value, dict) for value in frame[name]):
            frame[name] = frame[name].map(
                lambda value: json.dumps(value, default=_json_default, ensure_ascii=False)
                if isinstance(value, dict) else value
            )
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAF: budget-constrained dataset assembly for fair clustering")
    parser.add_argument("--query", required=True, type=Path, help="Existing initial dataset (.parquet or .jsonl)")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sources", type=Path, help="JSON list of existing local source paths and costs")
    inputs.add_argument("--agent-config", type=Path, help="YAML/JSON configuration for existing image sources")
    parser.add_argument("--config", type=Path, help="RAF parameters as JSON")
    parser.add_argument("--output", required=True, type=Path, help="New or empty output directory")
    parser.add_argument("--budget", type=float, help="Override max_cost")
    parser.add_argument("--seed", type=int, help="Override random_state")
    parser.add_argument("--group-col", help="Override sensitive_col")
    parser.add_argument("--feature-col", help="Override feature_col")
    parser.add_argument("--n-clusters", type=int, help="Override the number of preclusters")
    parser.add_argument("--agent-state", type=Path, help="Persistent image-agent cache/state directory")
    parser.add_argument("--group-attribute", default="group", help="Agent label to use as the RAF group")
    args = parser.parse_args(argv)
    cfg = RAFConfig.from_json(args.config) if args.config else RAFConfig()
    overrides = {name: value for name, value in {
        "max_cost": args.budget, "random_state": args.seed, "sensitive_col": args.group_col,
        "feature_col": args.feature_col, "n_clusters": args.n_clusters,
    }.items() if value is not None}
    cfg = replace(cfg, **overrides)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be a new or empty directory; use a separate --agent-state for persistent acquisition state")
    query = load_dataset(args.query)
    if args.sources:
        manager = load_sources(args.sources, feature_col=cfg.feature_col, sensitive_col=cfg.sensitive_col,
                               seed=cfg.random_state)
    else:
        try:
            from raf_agent.config import build_agent, load_config
            from raf_agent.integration import build_raf_source_manager
        except ImportError as exc:
            raise RuntimeError('Image acquisition requires pip install ".[clip]"') from exc
        agent_cfg = load_config(args.agent_config)
        required = agent_cfg.setdefault("labels", {}).setdefault("required", [])
        if args.group_attribute not in required:
            required.append(args.group_attribute)
        agent = build_agent(agent_cfg, args.agent_state or output / "acquisition")
        try:
            manager = build_raf_source_manager(agent,
                costs={item["source_id"]: float(item.get("cost", 1.0)) for item in agent_cfg["sources"]},
                sensitive_attribute=args.group_attribute, feature_col=cfg.feature_col, sensitive_col=cfg.sensitive_col)
        except Exception:
            agent.close()
            raise
    result = RAFPipeline(cfg).run(query_df=query, source_manager=manager)
    output.mkdir(parents=True, exist_ok=True)
    _parquet_frame(result.augmented_df, cfg.feature_col).to_parquet(output / "augmented.parquet", index=False)
    accepted = pd.DataFrame(result.accepted_records) if result.accepted_records else result.augmented_df.iloc[:0]
    _parquet_frame(accepted, cfg.feature_col).to_parquet(output / "accepted.parquet", index=False)
    summary = {"config": cfg.to_dict(), **result.summary()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                   default=_json_default), encoding="utf-8")
    print(json.dumps(result.summary(), indent=2, default=_json_default))
    return 0
