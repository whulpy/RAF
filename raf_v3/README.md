# RAF

RAF assembles an augmented dataset for fair clustering by selecting external
sources under an acquisition budget and valuing retrieved minority samples.
Candidate valuation uses the initial dataset's representation and minority
coverage; it does not run a downstream clustering algorithm for each candidate.

This repository contains the RAF runtime and an optional image-acquisition
agent. It takes an **existing initial dataset and existing sources** as inputs.
Dataset construction, source/query generators, benchmark baselines, downstream
evaluation suites, experimental results, and pretrained model files are not
included.

## Install

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
# Activate the environment using your shell's usual command.
python -m pip install .
python -m raf --help
```

For acquisition from existing image folders or configured image APIs using CLIP:

```bash
python -m pip install ".[clip]"
```

The `[agent]` extra installs image preparation and acquisition dependencies for
applications that supply their own `EmbeddingProvider`. The `[clip]` extra also
installs PyTorch and Transformers. CLIP weights are loaded on first use. HNSW
requires `hnswlib`; if your platform has no matching wheel, its installation
requires a C++ build toolchain.

## Run with existing embeddings

The initial dataset and each local source must be Parquet or JSONL files with:

| Field | Meaning |
| --- | --- |
| `embedding` | A finite, nonempty numeric vector; every vector has the same dimension |
| `group` | An explicit group label with consistent semantics across inputs |
| Other columns | Optional record identifiers and metadata, retained in the output |

Use the same encoder and preprocessing for all embeddings. Group labels are
provided by the input data; RAF does not infer them from embeddings. At least
two groups must be represented in the initial dataset.

1. Copy `configs/sources.example.json` and set paths to your existing source files.
2. Copy `configs/raf.example.json` and adjust the budget, column names, and preclusters.
3. Run:

```bash
python -m raf --query /path/to/query.parquet --sources /path/to/sources.json --config configs/raf.example.json --output /path/to/new-run
```

On Windows, quote paths containing spaces. No machine-specific drive or directory
is assumed. Relative source paths are resolved against the source configuration
file, not the shell's working directory.

The equivalent console command is `raf`. Common overrides are `--budget`,
`--seed`, `--group-col`, `--feature-col`, and `--n-clusters`.

Each source configuration declares a positive per-sample cost. Local sources
are loaded into memory and sampled uniformly without replacement by default.
For larger repositories or remote services, implement `DataSource.sample_one()`
or use the optional external agent. The local adapter simulates the retrieval
interface; it does not claim to avoid reading a local source file into memory.

## Outputs

The output directory must be new or empty. A successful run writes:

| File | Contents |
| --- | --- |
| `augmented.parquet` | Initial dataset plus the final retained external samples |
| `accepted.parquet` | Final retained external samples only |
| `summary.json` | Resolved RAF configuration, counts, costs, timing, and stop reason |

`summary.accepted` counts cumulative acceptance events. `summary.retained`
counts samples remaining after optional redundancy removal. Thus
`accepted - removed = retained`. The file `accepted.parquet` contains the
retained records, not the historical acceptance events.

The budget counts every successfully retrieved sample, including majority
samples and candidates later rejected. RAF considers only sources affordable
under the remaining budget. An optional `max_accepted` stops when that many
external records remain active. `max_steps` also bounds attempts, and finite
source exhaustion ends acquisition cleanly.

## Python API

```python
from raf import RAFConfig, RAFPipeline, load_dataset, load_sources

config = RAFConfig(
    sensitive_col="group",
    feature_col="embedding",
    n_clusters=10,
    max_cost=1000,
    raf_redundancy_elimination=True,
)
query = load_dataset("/path/to/query.parquet")
sources = load_sources("/path/to/sources.json", seed=config.random_state)
result = RAFPipeline(config).run(query_df=query, source_manager=sources)
print(result.summary())
augmented = result.augmented_df
```

`InMemorySource` accepts records supplied by your application.
`DataSource` is the interface for custom acquisition adapters. Return `None`
only when the selected source is exhausted; raise an error for transport or
schema failures. `RAFPipeline.run()` closes its source manager on exit.

## Optional image-acquisition agent

The runtime boundary is:

```text
RAF selects a source
    -> agent.fetch_one(source_id)
    -> acquire -> validate -> deduplicate -> read labels -> preprocess -> embed
    -> RAF matches demand -> values candidate -> accepts or rejects
```

The agent receives a source ID, not RAF's demand matrix, source rewards, or
valuation results. It never selects a different source or makes the RAF
acceptance decision. This release includes existing-folder and configurable
REST connectors, persistent deduplication, image validation, metadata labels,
CLIP embedding, and representation checks.

Configure your existing sources using `configs/agent.example.yaml`, with the
same encoder, preprocessing, and labels used for the initial dataset:

```bash
python -m raf --query /path/to/query.parquet --agent-config /path/to/agent.yaml --config configs/raf.example.json --agent-state /path/to/acquisition-state --output /path/to/new-run
```

Use `--group-attribute` if the agent's label name differs from `group`.
Source costs are declared in the agent configuration. The accounting unit is
one prepared sample returned to RAF; provider charges for HTTP retries or image
downloads are not automatically measured. Persistent state resumes acquisition
cursors and deduplication, not the RAF bandit or valuation state.

Local image sources use a seeded random permutation. REST sources must expose
an appropriate sampling interface; ordinary sequential pagination is not
uniform random sampling. See [runtime details](docs/runtime.md) before adapting
an API. Source discovery, bulk download/installers, initial-dataset bootstrap,
label inference, and LLM workflows are outside this runtime.

## Code layout

```text
configs/                    Input/configuration templates; no data
docs/runtime.md             Algorithm semantics and extension interfaces
src/raf/
    config.py               Runtime parameters and source specifications
    data.py                 Existing-data readers and source samplers
    clustering.py           Initial preclusters and radius-based assignment
    demand.py               Cluster-group demand
    bandit.py               Demand-aware source selection
    selection.py            MaxSim valuation, HNSW, posting lists, redundancy
    pipeline.py             Complete streaming RAF execution
    cli.py                  Command-line entry and output export
src/raf_agent/
    agent.py                Same-source acquisition/preparation
    connectors/             Existing-folder and REST access
    embeddings/             Encoder interface, CLIP, representation checks
    labels/                 Explicit metadata/manifest labels
    preprocessing/          Image preprocessing
    storage/                Runtime cache, deduplication, acquisition log
    validation/             Image validation
    integration/            Adapter to RAF's source interface
```

This release is extracted from `raf_standalone` and the current `raf_agent`
runtime inside `RAF_source_agent`. It has no imports from either original
directory. Defaults and implementation details are described in
[runtime details](docs/runtime.md); this is a runtime distribution, not a claim
to reproduce a particular experimental table.

## License

The code is distributed under the [MIT License](LICENSE). External datasets,
model weights, and installed dependencies retain their own licenses; none are
bundled here.
