# agent-trace-platform

> Streaming trace + replay + eval platform for LLM agents. OTel-compatible ingest, Iceberg-backed storage, deterministic replay, multi-rubric eval.

**Status:** Active development. See [ARCHITECTURE.md](ARCHITECTURE.md) for design.

## Why this exists

Production agent systems emit per-step structured signals (LLM calls, tool calls, tokens, latency, cost) that look a lot like ML feature logs. The same problems apply: storage cost at TB/day scale, training-serving skew when you want to evaluate a new model, deterministic replay for regression testing, and SLO-aware ingest under bursty load.

This project ports the feature-platform pattern to agent traces:

| Feature platform concept | Agent trace platform concept |
|---|---|
| Streaming feature logging (TB/day)        | OTLP-compatible span ingest          |
| Offline feature store (Iceberg)           | Iceberg-backed trace warehouse       |
| Training-serving skew elimination         | Deterministic replay (cached tools)  |
| Model validation service                  | Replay + multi-rubric eval           |
| Per-feature lineage                       | Span DAG + run lineage               |

## Quickstart

```bash
uv sync --all-extras            # or: pip install -e ".[dev]"
atp serve                       # starts ingest + replay API on :8080
atp demo agent                  # runs the demo ReAct agent against :8080
atp query runs --agent demo     # list runs
atp replay <run_id> --model claude-haiku-4-5
atp eval <run_id> --rubric semantic,cost-delta
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## License

MIT. See [LICENSE](LICENSE).
