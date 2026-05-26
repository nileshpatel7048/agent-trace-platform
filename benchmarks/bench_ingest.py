"""Ingest + query benchmark.

Measures end-to-end p50/p95/p99 ingest latency through the HTTP layer using the
in-process TestClient (no network jitter), plus query latency for
"all spans for one run". Results pinned to README.md.
"""
from __future__ import annotations

import json
import statistics
import tempfile
import time
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from agent_trace_platform.schema import Span, SpanKind
from agent_trace_platform.server import create_app


def _mk_batch(run_id: str, n: int, agent_id: str = "bench") -> list[dict]:
    spans = []
    base_ns = time.time_ns()
    for i in range(n):
        s = Span(
            agent_id=agent_id,
            run_id=run_id,
            step_idx=i,
            span_kind=SpanKind.LLM if i % 2 == 0 else SpanKind.TOOL,
            ts_start_ns=base_ns + i * 1_000_000,
            ts_end_ns=base_ns + i * 1_000_000 + 500_000,
            model="claude-haiku-4-5" if i % 2 == 0 else None,
            tool_name="calc" if i % 2 == 1 else None,
            input={"i": i},
            output={"i": i, "ok": True},
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.0001,
        )
        spans.append(s.model_dump(mode="json"))
    return spans


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def run(total_spans: int = 10_000, batch_size: int = 100) -> dict:
    with tempfile.TemporaryDirectory() as wh:
        app = create_app(warehouse_path=wh)
        latencies_ms: list[float] = []
        run_ids = []
        with TestClient(app) as client:
            t0 = time.perf_counter()
            for _ in range(total_spans // batch_size):
                run_id = uuid.uuid4().hex
                run_ids.append(run_id)
                payload = {"spans": _mk_batch(run_id, batch_size)}
                start = time.perf_counter()
                r = client.post("/v1/traces", json=payload)
                latencies_ms.append((time.perf_counter() - start) * 1000)
                r.raise_for_status()
            wall = time.perf_counter() - t0

            # Query latency: spans for one run (after batcher has flushed)
            time.sleep(0.2)  # let final batch flush
            q_lat: list[float] = []
            for rid in run_ids[:50]:
                s = time.perf_counter()
                r = client.get(f"/v1/runs/{rid}")
                q_lat.append((time.perf_counter() - s) * 1000)
                r.raise_for_status()

            list_start = time.perf_counter()
            r = client.get("/v1/runs?limit=200")
            list_ms = (time.perf_counter() - list_start) * 1000
            r.raise_for_status()

        return {
            "total_spans": total_spans,
            "batch_size": batch_size,
            "wall_seconds": round(wall, 3),
            "throughput_spans_per_sec": round(total_spans / wall, 1),
            "ingest_post_ms": {
                "p50": round(statistics.median(latencies_ms), 2),
                "p95": round(pct(latencies_ms, 95), 2),
                "p99": round(pct(latencies_ms, 99), 2),
                "max": round(max(latencies_ms), 2),
            },
            "query_get_run_ms": {
                "p50": round(statistics.median(q_lat), 2),
                "p95": round(pct(q_lat, 95), 2),
                "p99": round(pct(q_lat, 99), 2),
            },
            "query_list_runs_ms": round(list_ms, 2),
        }


if __name__ == "__main__":
    import sys

    total = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    result = run(total_spans=total, batch_size=batch)
    print(json.dumps(result, indent=2))
    out = Path(__file__).parent / "latest.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote: {out}")
