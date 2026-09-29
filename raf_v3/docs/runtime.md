# Runtime details

## Core algorithm

1. Precluster the initial embeddings with the original MiniBatchKMeans routine.
   For each cluster, the matching radius is the 95th percentile of initial
   within-cluster distances. Retrieved samples outside every radius do not
   reach valuation.
2. Count each group in each precluster. For counts `n[c, g]`, initialize
   `q[c, g] = max_h n[c, h] - n[c, g]`.
3. Choose one globally largest group in the initial data as the reference
   majority group. Each other initially observed group has its own minority
   valuation state. A group unseen in the initial dataset is not added as a
   new minority state during acquisition.
4. Select an affordable source using the demand-aware epsilon-greedy policy.
   Record every returned sample's cluster/group observation. A retrieved
   minority sample becomes a candidate only if it matches a precluster and
   the corresponding residual demand is positive.
5. Mark a demand-hit candidate as effective for source estimation. Decrease
   its demand cell by `demand_decay_per_candidate`, clamped at zero, before
   valuation; this decrease does not depend on eventual acceptance.
6. Value the candidate using the existing HNSW-assisted MaxSim update and
   posting lists. Accept only positive gain above `eps`. With redundancy
   removal enabled, check empty posting lists and optionally verify against
   all majority reference vectors before retiring an acquired sample.
7. Continue until the budget, retained-sample target, attempt limit, or source
   availability ends the run. Return the initial dataset plus active accepted
   records.

For each minority group, coverage uses negative squared Euclidean similarity:

```text
coverage(M) = sum over majority x of max over minority y in M (-||x - y||^2).
```

The HNSW path limits the majority entries considered for a candidate through
nearby minority owners' posting lists. It is an approximate acceleration and
is not a guarantee that every beneficial sample is accepted. Internal
preclusters represent acquisition demands; no downstream clustering is run
during candidate valuation.

The source policy retains the current implementation's acquisition-cost
weights, estimated effective-hit multiplier, and exploration schedule:

```text
epsilon(t) = max(eps_min, min(1, alpha * (log(t) / t)^(1/3))), for t > 1.
```

Here the effective-hit count means candidates admitted by the demand gate,
not accepted samples. Pruning, alternative bandit policies, and valuation
baselines are not exposed by the public runtime configuration.

## Parameters

The Python defaults preserve the source implementation's main RAF settings:

| Parameter | Default | Role |
| --- | ---: | --- |
| `n_clusters` | 20 | Number of initial preclusters |
| `cluster_subsample_ratio` | 0.1 | Subsampling for fitting preclusters |
| `max_cost` | 20000 | Finite acquisition budget |
| `max_steps` | 200000 | Maximum acquisition attempts |
| `max_accepted` | null | Optional active retained-sample target |
| `demand_decay_per_candidate` | 0.2 | Decrement for an admitted candidate |
| `fair_eps_min` | 0.1 | Exploration floor |
| `fair_alpha_eps` | 1.0 | Exploration multiplier |
| `fair_kappa` | 0.0 | Effective-hit uncertainty multiplier |
| `hybrid_neighbor_k` | 50 | Minority neighbors considered for valuation |
| `hnsw_single_index` | false | Single index versus main/delta indexes |
| `raf_redundancy_elimination` | false | Retire redundant acquired records |
| `raf_redundancy_exact_check` | true | Verify retirement against the majority set |

The example configuration explicitly enables redundancy removal and sets ten
preclusters. The demand decrement is an algorithm parameter, not a statement
that one admitted candidate resolves one sample of a count deficit. Set it
explicitly when comparing runs. No dataset-specific group merges are applied.

Initial minority samples are never retired. Redundancy checks preserve the
specified coverage objective within the configured tolerance; they do not
guarantee unchanged downstream clustering or fairness metrics.

Source and clustering seeds are exposed. HNSW is approximate and its parallel
construction can change neighbor sets between runs. A fixed seed should not
be interpreted as a guarantee of bit-for-bit identical accepted samples.

`valuation_s` sums only candidate valuation calls, including their internal
updates and redundancy checks. It excludes initialization, acquisition,
candidate cluster assignment, serialization, and downstream clustering.

## Adding a source adapter

Implement the interface in `raf.data.DataSource`:

```python
from raf import DataSource

class RepositorySource(DataSource):
    def sample_one(self):
        # Ask this source's existing sampling interface for one record.
        # Return a dict with the configured embedding and group columns.
        # Return None only for genuine source exhaustion.
        raise NotImplementedError
```

Supply instances in `SourceManager`. The pipeline takes responsibility for
closing the manager when the run exits. Every validly returned record counts
toward the budget, even when it is outside the modeled demand or is rejected
by valuation. Schema/representation failures raise errors rather than creating
placeholder vectors or fabricated group labels.

## Image-agent configuration

`LocalFolderConnector` reads existing images and an optional existing metadata
CSV. It never creates a source dataset. Metadata labels may also come from an
existing manifest. `GenericRESTConnector` reads a configured JSON endpoint and
follows configured field paths.

For a REST endpoint with stable random access, configure `pagination.mode` as
`random_offset`, set `max_offset` to the largest valid offset, and use
`page_size: 1`. The uniformity assumption depends on the provider's offset
semantics and the population remaining stable. The connector rejects repeated
external IDs and the agent rejects duplicate content. Offset and cursor modes
are available for APIs that only stream sequentially, but must not be described
as uniform sampling from the complete source.

Use `auth_env` to map header names to environment-variable names. The
configuration stores those names, not credential values. Runtime state and
provenance may contain external IDs, paths, and provider URLs; they are written
only to the user-selected runtime directory and are not part of the package.

The image agent requires an explicit embedding model. There is no dummy encoder
in the release. The initial dataset must use the same preprocessing and model;
dimension/model-name checks alone cannot establish semantic compatibility.
Do not use the cache from an unrelated encoder, preprocessing, or source
configuration. The built-in persistent store resumes acquisition state only.

Applications with a different encoder can construct `ExternalDataAgent`
directly using the `EmbeddingProvider` interface. The agent never calls a
language model or interprets demographic traits from image appearance.

## Extraction and integration notes

Core preclustering, demand computation, source scoring, MaxSim valuation, and
redundancy logic originate from `raf_standalone`. Image connectors and preparation
originate from `RAF_source_agent/raf_agent`. Experimental orchestration and
legacy compatibility APIs were removed. The streaming pipeline and CLI expose
only the retained runtime path.

Integration fixes enforce affordability before retrieval; distinguish finite
source exhaustion from transport failures; deliver the last cursor page before
exhaustion; bound duplicate-only REST scans; clamp HNSW queries to live elements;
and include newly appended samples in index rebuilds. These fixes are confined
to this distribution. They do not modify the original source directories.

Package validation and temporary inputs/results are maintained outside this
release tree. No validation fixtures, benchmark outputs, caches, credentials,
datasets, or model binaries are included.
