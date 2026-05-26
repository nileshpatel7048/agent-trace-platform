# Architecture

This document is the system design contract for `agent-trace-platform`. If you can defend everything in this doc, you can defend the project.

## 1. Problem statement

Production agent systems (ReAct loops, multi-agent orchestration, tool-using LLMs) emit per-step events that are structurally similar to ML feature logs:

- High volume (10s of GB/day per medium-scale deployment).
- Mixed-cardinality dimensions (agent_id, run_id, model, tool_name).
- Need both real-time inspection ("what is run X doing right now?") and offline analytics ("which tool calls regress on model upgrade?").
- Need deterministic replay for evaluation (the same problem as eliminating training-serving skew in ML feature platforms).

Existing solutions split into two camps:

1. **APM/observability tools** (Datadog, Honeycomb, LangSmith). Great at the inspection axis. Weak at offline analytics over months of data, and not designed for deterministic replay.
2. **Eval frameworks** (Inspect, promptfoo, ragas). Great at scoring a fixed eval set. Weak at ingesting production traffic and replaying *production* runs against a new model.

This platform sits between them: production trace ingest at OTel scale, Iceberg-backed durable storage for cheap long-window analytics, and a replay engine that re-runs stored agent runs against an alternate model with cached tool outputs so the eval is apples-to-apples.

## 2. High-level architecture

```
                                    ┌────────────────────────┐
   Agents (any framework) ───OTLP──▶│  Ingest API (FastAPI)  │
   (LangGraph, CrewAI, raw)         │  /v1/traces            │
                                    └─────────┬──────────────┘
                                              │ batched writes
                                              ▼
                                    ┌────────────────────────┐
                                    │  Trace Store           │
                                    │  Iceberg (PyIceberg)   │
                                    │  Catalog: sqlite/Glue  │
                                    │  Partitioned by        │
                                    │   day, agent_id        │
                                    └─────────┬──────────────┘
                                              │ DuckDB query engine
                              ┌───────────────┼───────────────────────┐
                              ▼               ▼                       ▼
                    ┌────────────────┐ ┌──────────────┐    ┌────────────────────┐
                    │  Query API     │ │  Replay API  │    │  Eval API          │
                    │  /v1/runs      │ │  /v1/runs/:id│    │  /v1/runs/:id/eval │
                    │  /v1/spans     │ │   /replay    │    │  rubric=...        │
                    └────────────────┘ └──────┬───────┘    └─────────┬──────────┘
                                              │                      │
                                              ▼                      ▼
                                    ┌────────────────────────────────────┐
                                    │ Model adapters: Anthropic, OpenAI, │
                                    │ local vLLM. Tool cache layer.      │
                                    └────────────────────────────────────┘
```

## 3. Span schema

Every event in the system is an OTel span with required + optional attributes. We extend OTel rather than invent a new format because (a) existing agent frameworks already emit OTel and (b) it composes with the rest of the observability stack.

### Required attributes

| Attribute              | Type       | Meaning |
|------------------------|------------|---------|
| `trace_id`             | uuid       | OTel-level trace id; one per `run`         |
| `span_id`              | uuid       | this step                                  |
| `parent_span_id`       | uuid?      | for tool-call nesting                      |
| `agent.id`             | string     | logical agent name (e.g., "support-bot")   |
| `run.id`               | uuid       | application-level conversation/run id      |
| `step.idx`             | int        | monotonic step within run                  |
| `span.kind`            | enum       | `llm` \| `tool` \| `agent` \| `internal`   |
| `ts.start_ns`          | int64      | start ns since epoch                       |
| `ts.end_ns`            | int64      | end ns since epoch                         |

### Conditional attributes — `span.kind == llm`

| Attribute              | Type       | Meaning |
|------------------------|------------|---------|
| `llm.model`            | string     | e.g. `claude-sonnet-4-6`                   |
| `llm.input`            | json       | full request body (messages)               |
| `llm.output`           | json       | response body                              |
| `llm.input_tokens`     | int        |                                            |
| `llm.output_tokens`    | int        |                                            |
| `llm.cost_usd`         | decimal    |                                            |

### Conditional attributes — `span.kind == tool`

| Attribute       | Type   | Meaning |
|-----------------|--------|---------|
| `tool.name`     | string | e.g. `web_search`                          |
| `tool.args`     | json   | call args                                  |
| `tool.output`   | json   | result                                     |
| `tool.error`    | string?| if errored                                 |

`tool.output` is what enables deterministic replay. Without it, replaying a run that called `web_search("AAPL stock price today")` would get a *different* price tomorrow and the eval would be polluted by tool variance instead of measuring model behavior.

## 4. Iceberg storage layout

### Table: `traces.spans`

| Column           | Type             | Notes                                  |
|------------------|------------------|----------------------------------------|
| trace_id         | string           |                                        |
| run_id           | string           |                                        |
| span_id          | string           | PK                                     |
| parent_span_id   | string nullable  |                                        |
| agent_id         | string           |                                        |
| step_idx         | int              |                                        |
| span_kind        | string           |                                        |
| ts_start_ns      | long             |                                        |
| ts_end_ns        | long             |                                        |
| ts_date          | date             | partition column (derived)             |
| model            | string nullable  |                                        |
| tool_name        | string nullable  |                                        |
| input            | string (json)    | left as string for schema flexibility  |
| output           | string (json)    |                                        |
| tokens_in        | int nullable     |                                        |
| tokens_out       | int nullable     |                                        |
| cost_usd         | double nullable  |                                        |
| error            | string nullable  |                                        |
| attrs            | string (json)    | overflow for non-promoted attrs        |

**Partitioning:** `(ts_date, bucket(agent_id, 16))`.

- `ts_date` because almost every query has a time bound and we want to prune.
- `bucket(agent_id)` because cardinality is in the hundreds, not millions, and per-agent scans are common ("show me everything for support-bot last week"). Bucketing keeps partition count finite as agents are added.

**Why not Postgres?** Postgres works fine at GB scale. At the TB/day scale we are designing for (in line with LinkedIn Featurelog: 10+ TB/day), columnar parquet with partition pruning is ~50x cheaper to scan and is the only path to "query 30 days of agent X in seconds". Iceberg gives us schema evolution and atomic appends, both of which Parquet-on-S3 alone does not.

**Why not Postgres + ClickHouse?** Two systems = two sets of failure modes, two backup stories, two query languages for users. For an interview project, the simpler defensible answer is "one columnar store, one catalog."

**Local vs production catalog:** Locally we use `pyiceberg.catalog.sql.SqlCatalog` with sqlite. In a real deployment this is swapped for Glue / Polaris / Nessie. The catalog interface is the only thing that changes.

## 5. Deterministic replay

### Goal

Given a stored run R that produced output O on model M, re-run R on model M' and produce O' such that **the only source of variance between O and O' is the model swap**. Tool calls, system prompts, retrieval contents, and seeds are held constant.

### How

1. Load all spans for `run_id = R`.
2. Reconstruct the step DAG via `parent_span_id`.
3. Walk the agent loop in stored order. For each step:
   - If `span.kind == llm`: call M' with the same `llm.input` as recorded. Record M''s output.
   - If `span.kind == tool`: **do not** re-execute the tool. Return `tool.output` from the cache (the recorded run). This is what makes replay deterministic. A flag `--live-tools` exists for users who explicitly want fresh tool calls.
4. Persist the replay as a new run with `replay.of = R` attribute.

### Why this matters in an interview

This mirrors the "Model Validation Service" line on the Salesforce Einstein resume bullet: *"Offline experimentation service validating ML experiments against production data."* For agents, the equivalent is: re-run yesterday's production conversations against a candidate model and diff. You can talk about the parallel directly.

### Determinism limits (be honest about these)

- LLMs are non-deterministic even at temperature 0 across hardware. Replay measures distribution drift, not bit-exact reproducibility.
- Tools with hidden state that affects the conversation (e.g., a counter incrementing on each call) cannot be perfectly cached. The platform flags spans where `tool.args` was identical across runs but `tool.output` differed.

## 6. Eval rubrics

Composable scorers. Each takes `(original_run, replay_run)` and returns a numeric score plus per-step breakdown.

| Rubric          | Definition                                                                |
|-----------------|---------------------------------------------------------------------------|
| `exact`         | Final-answer string equality.                                             |
| `semantic`      | Embedding cosine between final answers (Voyage / OpenAI embeddings).      |
| `cost-delta`    | `(replay_cost - original_cost) / original_cost`.                          |
| `latency-delta` | Same, on wall-clock per step.                                             |
| `tool-seq`      | Normalized Levenshtein distance between tool-name sequences.              |
| `tool-args`     | Fraction of tool calls where args matched (after JSON normalization).     |

A rubric is a Python class with one method: `score(original, replay) -> RubricResult`. Adding a new one is < 30 lines.

## 7. SLO targets

Honest about what's tested vs. aspirational.

| SLO                                | Target            | Measured?                  |
|------------------------------------|-------------------|----------------------------|
| Ingest p99 (single span, single node)| < 50 ms         | Yes — see `benchmarks/`    |
| Ingest throughput (single node)    | > 5k spans/sec    | Yes                        |
| Query: spans for one run           | p95 < 200 ms      | Yes                        |
| Query: 7-day window for one agent  | p95 < 2 s         | Yes                        |
| Replay overhead vs raw model call  | < 15% over baseline| Yes                       |
| Catalog scale                      | "fits on one node up to ~1 TB; designed to scale via Glue/Polaris" | Designed, not load-tested |

## 8. What's explicitly out of scope

- Auth / multi-tenancy. Single-tenant local deploy. Documented path: gateway in front, tenant_id partition column.
- Streaming aggregations (Flink / Spark Structured Streaming). Iceberg + scheduled compaction is the v1 story.
- Distributed replay. Replay is single-node and sequential within a run; runs are independent and trivially parallelizable across machines.
- A polished web UI. CLI + JSON API only. A minimal Streamlit page may land later.

## 9. Comparison to existing systems

| System          | Ingest at scale | Iceberg/columnar | Deterministic replay | Multi-rubric eval | Open source |
|-----------------|:----:|:----:|:----:|:----:|:----:|
| LangSmith       | ✓ | partial | ✗ | partial | ✗ |
| Langfuse        | ✓ | ClickHouse | ✗ | partial | ✓ |
| Phoenix (Arize) | ✓ | ✗ | ✗ | ✓ | ✓ |
| Inspect (UK AISI)| ✗ | ✗ | ✓ (different model) | ✓ | ✓ |
| **this**        | ✓ | ✓ | ✓ | ✓ | ✓ |

The genuine differentiator is the combination of Iceberg-backed storage (for cheap long-window analytics over production traces) and deterministic replay against the *same* stored runs.

## 10. Open questions / decisions deferred

These are honest "I would decide this in a design review" items. Leaving them open is fine for a portfolio project — pretending they're solved would be worse.

- Compaction cadence. Current plan: hourly small-file compaction via a `pyiceberg` action; daily rewrite to optimize layout.
- Schema evolution policy. Iceberg supports it; the explicit policy (which fields are append-only vs. mutable) is TBD.
- Retention. No automatic TTL in v1. Production deploy needs a partition-drop job tied to a retention config.
- Backpressure on ingest. Current behavior is synchronous write to Iceberg; under sustained > 5k spans/sec a Kafka or pulsar buffer is the standard answer.
