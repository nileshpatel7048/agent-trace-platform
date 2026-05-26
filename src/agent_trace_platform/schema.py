"""Span schema. Mirrors ARCHITECTURE.md section 3."""
from __future__ import annotations

import json
import time
import uuid
from datetime import UTC
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class SpanKind(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    AGENT = "agent"
    INTERNAL = "internal"


def _new_id() -> str:
    return uuid.uuid4().hex


def _now_ns() -> int:
    return time.time_ns()


class Span(BaseModel):
    """A single agent-step span. OTel-shaped but with promoted agent attributes."""

    trace_id: str = Field(default_factory=_new_id)
    span_id: str = Field(default_factory=_new_id)
    parent_span_id: str | None = None

    agent_id: str
    run_id: str
    step_idx: int = 0
    span_kind: SpanKind

    ts_start_ns: int = Field(default_factory=_now_ns)
    ts_end_ns: int = Field(default_factory=_now_ns)

    # llm
    model: str | None = None
    input: Any | None = None
    output: Any | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None

    # tool
    tool_name: str | None = None
    error: str | None = None

    # overflow
    attrs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_kind_required(self) -> Span:
        if self.span_kind == SpanKind.LLM and self.model is None:
            raise ValueError("llm spans require `model`")
        if self.span_kind == SpanKind.TOOL and self.tool_name is None:
            raise ValueError("tool spans require `tool_name`")
        if self.ts_end_ns < self.ts_start_ns:
            raise ValueError("ts_end_ns < ts_start_ns")
        return self

    @property
    def latency_ms(self) -> float:
        return (self.ts_end_ns - self.ts_start_ns) / 1e6

    def to_row(self) -> dict[str, Any]:
        """Flatten to a row matching the Iceberg schema."""
        from datetime import datetime

        ts_date = datetime.fromtimestamp(self.ts_start_ns / 1e9, tz=UTC).date()
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "agent_id": self.agent_id,
            "step_idx": self.step_idx,
            "span_kind": self.span_kind.value,
            "ts_start_ns": self.ts_start_ns,
            "ts_end_ns": self.ts_end_ns,
            "ts_date": ts_date,
            "model": self.model,
            "tool_name": self.tool_name,
            "input": json.dumps(self.input) if self.input is not None else None,
            "output": json.dumps(self.output) if self.output is not None else None,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "error": self.error,
            "attrs": json.dumps(self.attrs) if self.attrs else "{}",
        }


class IngestRequest(BaseModel):
    spans: list[Span]


class IngestResponse(BaseModel):
    accepted: int
    rejected: int = 0
    rejected_reasons: list[str] = Field(default_factory=list)
