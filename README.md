# agent-trace-platform

> Streaming trace + replay + eval platform for LLM agents. OTLP-shaped ingest, Iceberg-backed storage, deterministic replay, multi-rubric eval — all open source.

[![ci](https://github.com/nileshpatel7048/agent-trace-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/nileshpatel7048/agent-trace-platform/actions/workflows/ci.yml)

## Why this exists

Production agent systems emit per-step structured events (LLM calls, tool calls, tokens, latency, cost) that look a lot like ML feature logs. The same engineering problems apply: TB/day storage cost, long-window analytics, training-serving skew when evaluating a candidate model, deterministic replay for regression testing.

Existing tools split into two camps and neither covers both:

- **APM/observability tools** (LangSmith, Langfuse, Phoenix): great at live inspection. Weak at offline analytics over months of traces. No deterministic replay against stored runs.
- **Eval frameworks** (Inspect, promptfoo, ragas): great at scoring fixed eval sets. Don't ingest production traffic.

This project sits in between: ingest production agent traces at scale, store them durably in Iceberg, and run deterministic replays of stored runs against alternate models — so the eval delta you measure is the model swap, not tool variance.

The mapping below is the design rationale:

| Feature platform concept | Agent trace platform concept |
|---|---|
| Streaming feature logging (TB/day)        | OTLP-shaped span ingest              |
| Offline feature store (Iceberg, Arrow)    | Iceberg-backed trace warehouse       |
| Eliminating training-serving skew         | Deterministic replay w/ cached tools |
| Model Validation Service (offline eval)   | Replay + multi-rubric eval engine    |

## Quickstart

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. run the server
atp serve --warehouse ./warehouse
#   listening on http://127.0.0.1:8080

# 3. emit a demo agent run (LLM -> calc tool -> web_search tool -> LLM)
atp demo agent
#   run_id: <hex>

# 4. inspect
atp query runs

# 5. replay the run against a different model
atp replay <run_id> --model claude-sonnet-4-6

# 6. score the replay vs original
atp eval <run_id> --replay-run <replay_id> --rubric exact,semantic,cost-delta,tool-seq
```

If `ANTHROPIC_API_KEY` is set, the demo agent calls the real Anthropic API. If not, a deterministic offline stub is used so everything runs without network.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design — schema, partitioning, replay semantics, what's deliberately out of scope.

```
   Agents ───OTLP-shaped JSON───▶  Ingest API (FastAPI + batcher)
                                              │
                                              ▼
                                    Iceberg trace store
                                    (partitioned by date, agent_bucket)
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                          Query API       Replay API       Eval API
                                              │
                                              ▼
                                  Model adapters (Anthropic/stub)
                                  + tool output cache (deterministic)
```

## Measured performance

From [`benchmarks/bench_ingest.py`](benchmarks/bench_ingest.py), 10,000 spans, 100/batch, single-node, M2 MacBook Pro. Reproduce with:

```bash
python benchmarks/bench_ingest.py 10000 100
```

| Metric                     | Target (SLO)  | Measured    |
|----------------------------|---------------|-------------|
| Ingest throughput          | > 5k spans/s  | **23.5k spans/s** |
| Ingest POST p50            | —             | 0.99 ms     |
| Ingest POST p95            | —             | 7.4 ms      |
| Ingest POST p99            | < 50 ms       | **18.2 ms** |
| Query: spans for one run, p50 | —          | 17 ms       |
| Query: spans for one run, p99 | < 200 ms   | **41.5 ms** |
| Query: list 200 runs (DuckDB agg) | < 2 s  | **151 ms**  |

Honest caveats:
- These are single-node, single-process numbers. Distribution / horizontal scale is out of scope for the open-source build (see ARCHITECTURE.md §8).
- The batcher coalesces by 256-spans-or-50ms, so steady-state throughput beats single-burst latency.
- p99 ingest includes the synchronous flush to Iceberg when a batch fills.

## Deterministic replay — why it matters

When you ask "how does model X compare to model Y on my production traffic?", you need apples-to-apples: identical inputs, identical tool results, identical retrieval contents. Otherwise the eval signal is polluted by tool variance.

The replay engine walks the original run in order. For each step:

- **LLM span** → re-invoke the (possibly different) model with the same recorded input.
- **Tool span** → return the cached `tool.output` from the original run (unless `--live-tools`).

The new run is written back as a separate run with `replay_of = <original_run_id>` in attrs, so it's queryable and re-replayable.

Determinism limits (called out honestly in ARCHITECTURE.md §5): LLMs are non-deterministic even at temperature 0 across hardware. Replay measures distribution drift, not bit-exact reproducibility.

## Eval rubrics

| Rubric         | What it measures                                                   |
|----------------|--------------------------------------------------------------------|
| `exact`        | Final-answer string equality                                       |
| `semantic`     | Embedding cosine on final answers (BoW locally; pluggable)         |
| `cost-delta`   | `(replay_cost - original_cost) / original_cost`, score=0 if worse  |
| `latency-delta`| Same, on wall-clock                                                |
| `tool-seq`     | Normalized Levenshtein between tool-name sequences                 |
| `tool-args`    | Fraction of tool calls where args matched (normalized JSON)        |

Adding a new rubric is a class with one method: `score(original, replay) -> RubricResult`. See [`src/agent_trace_platform/eval.py`](src/agent_trace_platform/eval.py).

## Project layout

```
src/agent_trace_platform/
  schema.py        # Pydantic Span model + OTel-shaped attributes
  store.py         # Iceberg-backed trace store (PyIceberg + sqlite catalog)
  ingest.py        # Async batcher (size or time triggered)
  server.py        # FastAPI app composing ingest/query/replay/eval
  models.py        # Anthropic + offline-stub adapters with pricing
  replay.py        # Deterministic replay engine
  eval.py          # 6 rubrics + composable runner
  demo_agent.py    # Scripted ReAct-style agent that emits spans
  cli.py           # `atp` entry point
benchmarks/
  bench_ingest.py  # Throughput + p99 ingest + query latency
tests/             # 22 tests, < 1s
ARCHITECTURE.md    # Full design doc, including what's out of scope
```

## Comparison to existing systems

| System          | Production ingest | Columnar/Iceberg | Deterministic replay | Multi-rubric eval | Open source |
|-----------------|:-:|:-:|:-:|:-:|:-:|
| LangSmith       | ✓ | partial | ✗ | partial | ✗ |
| Langfuse        | ✓ | ClickHouse | ✗ | partial | ✓ |
| Phoenix (Arize) | ✓ | ✗ | ✗ | ✓ | ✓ |
| Inspect (AISI)  | ✗ | ✗ | ✓ (different model) | ✓ | ✓ |
| **this**        | ✓ | ✓ | ✓ | ✓ | ✓ |

The differentiator is the combination of Iceberg-backed storage for cheap long-window analytics over production traces *and* deterministic replay against those same stored runs.

## What's deliberately out of scope (v1)

- Auth / multi-tenancy
- Streaming aggregations via Flink / Spark
- Distributed replay
- A web UI (CLI + JSON API only)

See ARCHITECTURE.md §8 for the full list and the design path to add each.

## License

MIT. See [LICENSE](LICENSE).
