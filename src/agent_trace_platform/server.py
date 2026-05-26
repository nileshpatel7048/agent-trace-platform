"""FastAPI app composing ingest, query, replay, eval."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .eval import run_rubrics
from .ingest import IngestBatcher
from .replay import ReplayEngine
from .schema import IngestRequest, IngestResponse
from .store import TraceStore

log = logging.getLogger(__name__)


class ReplayRequest(BaseModel):
    model_override: str | None = None
    live_tools: bool = False


class EvalRequest(BaseModel):
    replay_run_id: str
    rubrics: list[str] = ["exact", "semantic", "cost-delta", "latency-delta", "tool-seq"]


def create_app(warehouse_path: str | None = None) -> FastAPI:
    store = TraceStore(warehouse_path=warehouse_path)
    batcher = IngestBatcher(store)
    replay_engine = ReplayEngine(store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await batcher.start()
        yield
        await batcher.stop()

    app = FastAPI(title="agent-trace-platform", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/traces", response_model=IngestResponse)
    async def ingest(req: IngestRequest) -> IngestResponse:
        await batcher.submit(req.spans)
        return IngestResponse(accepted=len(req.spans))

    @app.get("/v1/runs")
    def list_runs(agent: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return store.list_runs(agent_id=agent, limit=limit)

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        spans = store.get_run(run_id)
        if not spans:
            raise HTTPException(status_code=404, detail="run not found")
        return {"run_id": run_id, "spans": spans}

    @app.post("/v1/runs/{run_id}/replay")
    def replay(run_id: str, body: ReplayRequest) -> dict[str, Any]:
        try:
            result = replay_engine.replay(
                run_id, model_override=body.model_override, live_tools=body.live_tools
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {
            "replay_run_id": result.replay_run_id,
            "original_run_id": result.original_run_id,
            "cost_usd": {"original": result.cost_usd_original, "replay": result.cost_usd_replay},
            "latency_ms": {
                "original": result.latency_ms_original,
                "replay": result.latency_ms_replay,
            },
            "n_spans": len(result.spans),
        }

    @app.post("/v1/runs/{run_id}/eval")
    def evaluate(run_id: str, body: EvalRequest) -> dict[str, Any]:
        original = store.get_run(run_id)
        replay = store.get_run(body.replay_run_id)
        if not original or not replay:
            raise HTTPException(status_code=404, detail="run(s) not found")
        results = run_rubrics(body.rubrics, original, replay)
        return {
            "original_run_id": run_id,
            "replay_run_id": body.replay_run_id,
            "rubrics": [
                {"name": r.name, "score": r.score, "details": r.details} for r in results
            ],
        }

    return app
