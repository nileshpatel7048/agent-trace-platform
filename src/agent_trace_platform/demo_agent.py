"""A tiny ReAct-style agent that emits spans to the platform.

It's intentionally simple: enough to produce realistic-looking traces with
LLM + tool spans, not a serious agent framework. The point is to exercise
ingest/replay/eval end-to-end.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx

from .models import get_adapter
from .schema import Span, SpanKind

DEFAULT_MODEL = "claude-haiku-4-5"


def _calc(args: dict[str, Any]) -> str:
    """Eval-free tiny calculator. Only supports + - * / on numbers."""
    expr = str(args.get("expr", "")).strip()
    if not all(c in "0123456789.+-*/() " for c in expr):
        return "ERROR: only digits and + - * / ( ) allowed"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - chars restricted above
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def _web_search(args: dict[str, Any]) -> str:
    """Stub. Returns canned content keyed on the query so replay is deterministic."""
    q = str(args.get("query", "")).lower()
    canned = {
        "weather sf": "San Francisco: 62F, partly cloudy.",
        "price gold": "Gold spot: $2,610/oz.",
    }
    for k, v in canned.items():
        if k in q:
            return v
    return f"no results for: {q}"


TOOLS = {"calc": _calc, "web_search": _web_search}


class DemoAgent:
    """Runs a fixed scripted plan so traces are reproducible. The agent loop
    is deliberately simple — LLM → optional tool → LLM → final. In a real
    agent this would be a ReAct/tool-use loop driven by the model."""

    def __init__(
        self,
        agent_id: str = "demo",
        platform_url: str | None = "http://127.0.0.1:8080",
        model: str = DEFAULT_MODEL,
    ):
        self.agent_id = agent_id
        self.platform_url = platform_url
        self.model = model
        self.adapter = get_adapter(model)

    def run(self, prompt: str, plan: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Execute one run. `plan` is an optional list of {tool, args} entries.
        If omitted, the agent skips tool use and just LLM-answers."""
        run_id = uuid.uuid4().hex
        trace_id = uuid.uuid4().hex
        spans: list[Span] = []
        step = 0

        # Step 0: initial LLM call (planning)
        t0 = time.time_ns()
        plan_prompt = {
            "messages": [{"role": "user", "content": prompt}],
            "system": "You are a helpful agent. Use tools when needed.",
        }
        plan_out, plan_usage = self.adapter.complete(self.model, plan_prompt)
        spans.append(
            Span(
                trace_id=trace_id,
                run_id=run_id,
                agent_id=self.agent_id,
                step_idx=step,
                span_kind=SpanKind.LLM,
                ts_start_ns=t0,
                ts_end_ns=time.time_ns(),
                model=self.model,
                input=plan_prompt,
                output=plan_out,
                tokens_in=plan_usage.tokens_in,
                tokens_out=plan_usage.tokens_out,
                cost_usd=plan_usage.cost_usd,
            )
        )
        step += 1

        tool_results: list[tuple[str, Any, str]] = []
        for call in plan or []:
            tname = call["tool"]
            targs = call.get("args", {})
            t0 = time.time_ns()
            fn = TOOLS.get(tname)
            if fn is None:
                tres = f"ERROR: unknown tool {tname}"
                tool_results.append((tname, targs, tres))
                spans.append(
                    Span(
                        trace_id=trace_id,
                        run_id=run_id,
                        agent_id=self.agent_id,
                        step_idx=step,
                        span_kind=SpanKind.TOOL,
                        ts_start_ns=t0,
                        ts_end_ns=time.time_ns(),
                        tool_name=tname,
                        input=targs,
                        output=tres,
                        error="unknown tool",
                    )
                )
                step += 1
                continue
            tres = fn(targs)
            tool_results.append((tname, targs, tres))
            spans.append(
                Span(
                    trace_id=trace_id,
                    run_id=run_id,
                    agent_id=self.agent_id,
                    step_idx=step,
                    span_kind=SpanKind.TOOL,
                    ts_start_ns=t0,
                    ts_end_ns=time.time_ns(),
                    tool_name=tname,
                    input=targs,
                    output=tres,
                )
            )
            step += 1

        # Final LLM call (synthesis)
        t0 = time.time_ns()
        synth_prompt = {
            "messages": [
                {"role": "user", "content": prompt},
                {
                    "role": "user",
                    "content": "Tool results:\n"
                    + "\n".join(f"- {n}({json.dumps(a)}) -> {r}" for n, a, r in tool_results),
                },
            ],
            "system": "Synthesize a final answer using the tool results.",
        }
        synth_out, synth_usage = self.adapter.complete(self.model, synth_prompt)
        spans.append(
            Span(
                trace_id=trace_id,
                run_id=run_id,
                agent_id=self.agent_id,
                step_idx=step,
                span_kind=SpanKind.LLM,
                ts_start_ns=t0,
                ts_end_ns=time.time_ns(),
                model=self.model,
                input=synth_prompt,
                output=synth_out,
                tokens_in=synth_usage.tokens_in,
                tokens_out=synth_usage.tokens_out,
                cost_usd=synth_usage.cost_usd,
            )
        )

        if self.platform_url:
            self._emit(spans)

        return {"run_id": run_id, "spans": [s.model_dump(mode="json") for s in spans]}

    def _emit(self, spans: list[Span]) -> None:
        payload = {"spans": [s.model_dump(mode="json") for s in spans]}
        try:
            httpx.post(f"{self.platform_url}/v1/traces", json=payload, timeout=5.0)
        except httpx.RequestError as e:  # noqa: BLE001
            # Don't crash the demo if the server isn't running.
            print(f"[demo] warn: failed to emit spans: {e}")
