import pytest
from pydantic import ValidationError

from agent_trace_platform.schema import Span, SpanKind


def test_llm_span_requires_model():
    with pytest.raises(ValidationError):
        Span(agent_id="a", run_id="r", span_kind=SpanKind.LLM)


def test_tool_span_requires_tool_name():
    with pytest.raises(ValidationError):
        Span(agent_id="a", run_id="r", span_kind=SpanKind.TOOL)


def test_end_before_start_rejected():
    with pytest.raises(ValidationError):
        Span(
            agent_id="a",
            run_id="r",
            span_kind=SpanKind.INTERNAL,
            ts_start_ns=1000,
            ts_end_ns=500,
        )


def test_to_row_round_trip():
    s = Span(
        agent_id="a",
        run_id="r",
        span_kind=SpanKind.LLM,
        model="claude-haiku-4-5",
        input={"messages": [{"role": "user", "content": "hi"}]},
        output={"content": [{"type": "text", "text": "hello"}]},
        tokens_in=2,
        tokens_out=1,
        cost_usd=0.001,
    )
    row = s.to_row()
    assert row["span_kind"] == "llm"
    assert row["model"] == "claude-haiku-4-5"
    assert "messages" in row["input"]  # serialized to JSON string
    assert row["ts_date"] is not None
