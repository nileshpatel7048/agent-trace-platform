"""Deterministic replay engine.

For each span in the original run:
  - llm span: re-invoke the (possibly-different) model with the same input.
  - tool span: return the cached tool.output unless --live-tools is set.

See ARCHITECTURE.md section 5 for the design rationale.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from .models import ModelAdapter, get_adapter
from .schema import Span, SpanKind
from .store import TraceStore


@dataclass
class ReplayResult:
    replay_run_id: str
    original_run_id: str
    spans: list[Span]
    cost_usd_original: float
    cost_usd_replay: float
    latency_ms_original: float
    latency_ms_replay: float


class ReplayEngine:
    def __init__(self, store: TraceStore):
        self.store = store

    def replay(
        self,
        run_id: str,
        model_override: str | None = None,
        live_tools: bool = False,
        adapter: ModelAdapter | None = None,
    ) -> ReplayResult:
        original_spans = self.store.get_run(run_id)
        if not original_spans:
            raise ValueError(f"no spans found for run_id={run_id}")

        replay_run_id = uuid.uuid4().hex
        replay_spans: list[Span] = []

        cost_orig = sum(s.get("cost_usd") or 0.0 for s in original_spans)
        cost_replay = 0.0
        lat_orig = sum(
            (s["ts_end_ns"] - s["ts_start_ns"]) / 1e6 for s in original_spans
        )
        lat_replay = 0.0

        agent_id = original_spans[0]["agent_id"]

        for orig in original_spans:
            kind = SpanKind(orig["span_kind"])
            t0 = time.time_ns()

            if kind == SpanKind.LLM:
                model = model_override or orig["model"]
                ad = adapter or get_adapter(model)
                out, usage = ad.complete(model=model, request=orig["input"])
                t1 = time.time_ns()
                cost_replay += usage.cost_usd
                lat_replay += (t1 - t0) / 1e6
                replay_spans.append(
                    Span(
                        trace_id=orig["trace_id"],
                        span_id=uuid.uuid4().hex,
                        parent_span_id=orig.get("parent_span_id"),
                        agent_id=agent_id,
                        run_id=replay_run_id,
                        step_idx=orig["step_idx"],
                        span_kind=SpanKind.LLM,
                        ts_start_ns=t0,
                        ts_end_ns=t1,
                        model=model,
                        input=orig["input"],
                        output=out,
                        tokens_in=usage.tokens_in,
                        tokens_out=usage.tokens_out,
                        cost_usd=usage.cost_usd,
                        attrs={"replay_of": run_id, "original_span_id": orig["span_id"]},
                    )
                )

            elif kind == SpanKind.TOOL:
                if live_tools:
                    raise NotImplementedError(
                        "live_tools=True requires a registered tool runner; deterministic replay only for now"
                    )
                t1 = time.time_ns()
                lat_replay += (t1 - t0) / 1e6
                replay_spans.append(
                    Span(
                        trace_id=orig["trace_id"],
                        span_id=uuid.uuid4().hex,
                        parent_span_id=orig.get("parent_span_id"),
                        agent_id=agent_id,
                        run_id=replay_run_id,
                        step_idx=orig["step_idx"],
                        span_kind=SpanKind.TOOL,
                        ts_start_ns=t0,
                        ts_end_ns=t1,
                        tool_name=orig["tool_name"],
                        input=orig["input"],
                        output=orig["output"],  # cached, deterministic
                        attrs={
                            "replay_of": run_id,
                            "original_span_id": orig["span_id"],
                            "tool_cached": True,
                        },
                    )
                )

            else:
                # agent / internal spans are bookkeeping; preserve as-is
                replay_spans.append(
                    Span(
                        trace_id=orig["trace_id"],
                        span_id=uuid.uuid4().hex,
                        parent_span_id=orig.get("parent_span_id"),
                        agent_id=agent_id,
                        run_id=replay_run_id,
                        step_idx=orig["step_idx"],
                        span_kind=kind,
                        ts_start_ns=t0,
                        ts_end_ns=time.time_ns(),
                        attrs={"replay_of": run_id, "original_span_id": orig["span_id"]},
                    )
                )

        self.store.append(replay_spans)

        return ReplayResult(
            replay_run_id=replay_run_id,
            original_run_id=run_id,
            spans=replay_spans,
            cost_usd_original=cost_orig,
            cost_usd_replay=cost_replay,
            latency_ms_original=lat_orig,
            latency_ms_replay=lat_replay,
        )


def final_answer(spans: list[dict[str, Any]] | list[Span]) -> str | None:
    """Last LLM span's text output is the final answer."""
    def kind(s: Any) -> str:
        return s["span_kind"] if isinstance(s, dict) else s.span_kind.value

    def out(s: Any) -> Any:
        return s["output"] if isinstance(s, dict) else s.output

    llm = [s for s in spans if kind(s) == "llm"]
    if not llm:
        return None
    o = out(llm[-1])
    if o is None:
        return None
    if isinstance(o, dict):
        # Anthropic-ish shape
        if "content" in o:
            parts = o["content"]
            if isinstance(parts, list):
                texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
                return "\n".join(t for t in texts if t)
            return str(parts)
        return str(o)
    return str(o)
