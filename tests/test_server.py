"""HTTP integration test: spin up the app, ingest, query, replay, eval."""
from fastapi.testclient import TestClient

from agent_trace_platform.schema import Span, SpanKind
from agent_trace_platform.server import create_app


def test_http_end_to_end(tmp_path):
    app = create_app(warehouse_path=str(tmp_path))
    with TestClient(app) as client:
        # ingest
        spans = [
            Span(
                agent_id="ag",
                run_id="run1",
                step_idx=0,
                span_kind=SpanKind.LLM,
                model="claude-haiku-4-5",
                input={"q": "calc"},
                output={"content": [{"type": "text", "text": "use calc"}]},
                tokens_in=5,
                tokens_out=3,
                cost_usd=0.0001,
            ),
            Span(
                agent_id="ag",
                run_id="run1",
                step_idx=1,
                span_kind=SpanKind.TOOL,
                tool_name="calc",
                input={"expr": "1+1"},
                output="2",
            ),
            Span(
                agent_id="ag",
                run_id="run1",
                step_idx=2,
                span_kind=SpanKind.LLM,
                model="claude-haiku-4-5",
                input={"q": "synth"},
                output={"content": [{"type": "text", "text": "2"}]},
                tokens_in=4,
                tokens_out=1,
                cost_usd=0.00005,
            ),
        ]
        r = client.post(
            "/v1/traces", json={"spans": [s.model_dump(mode="json") for s in spans]}
        )
        assert r.status_code == 200
        assert r.json()["accepted"] == 3

        # batcher flushes asynchronously; poll briefly
        import time

        for _ in range(50):
            runs = client.get("/v1/runs").json()
            if runs:
                break
            time.sleep(0.05)
        assert any(run["run_id"] == "run1" for run in runs)

        # get one run
        r = client.get("/v1/runs/run1")
        assert r.status_code == 200
        assert len(r.json()["spans"]) == 3

        # replay
        r = client.post(
            "/v1/runs/run1/replay", json={"model_override": "claude-haiku-4-5"}
        )
        assert r.status_code == 200
        replay_id = r.json()["replay_run_id"]

        # eval
        r = client.post(
            "/v1/runs/run1/eval",
            json={"replay_run_id": replay_id, "rubrics": ["tool-seq", "tool-args"]},
        )
        assert r.status_code == 200
        scores = {x["name"]: x["score"] for x in r.json()["rubrics"]}
        assert scores["tool-seq"] == 1.0
        assert scores["tool-args"] == 1.0


def test_replay_unknown_run_returns_404(tmp_path):
    app = create_app(warehouse_path=str(tmp_path))
    with TestClient(app) as client:
        r = client.post("/v1/runs/nope/replay", json={})
        assert r.status_code == 404
