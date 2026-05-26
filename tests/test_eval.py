from agent_trace_platform.eval import RUBRICS, run_rubrics


def _span(kind, **kw):
    base = {
        "trace_id": "t",
        "run_id": kw.pop("run_id", "r"),
        "span_id": kw.pop("span_id", "s"),
        "parent_span_id": None,
        "agent_id": "a",
        "step_idx": kw.pop("step_idx", 0),
        "span_kind": kind,
        "ts_start_ns": kw.pop("ts_start_ns", 0),
        "ts_end_ns": kw.pop("ts_end_ns", 100),
        "model": None,
        "tool_name": None,
        "input": None,
        "output": None,
        "tokens_in": None,
        "tokens_out": None,
        "cost_usd": None,
        "error": None,
    }
    base.update(kw)
    return base


def test_exact_match():
    o = [_span("llm", output={"content": [{"type": "text", "text": "42"}]})]
    r = [_span("llm", output={"content": [{"type": "text", "text": "42"}]})]
    res = RUBRICS["exact"]().score(o, r)
    assert res.score == 1.0


def test_exact_mismatch():
    o = [_span("llm", output={"content": [{"type": "text", "text": "42"}]})]
    r = [_span("llm", output={"content": [{"type": "text", "text": "43"}]})]
    assert RUBRICS["exact"]().score(o, r).score == 0.0


def test_semantic_bag_overlap():
    o = [_span("llm", output={"content": [{"type": "text", "text": "the cat sat"}]})]
    r = [_span("llm", output={"content": [{"type": "text", "text": "the cat sat on mat"}]})]
    res = RUBRICS["semantic"]().score(o, r)
    assert 0.5 < res.score < 1.0


def test_cost_delta_replay_more_expensive():
    o = [_span("llm", cost_usd=0.001)]
    r = [_span("llm", cost_usd=0.005)]
    res = RUBRICS["cost-delta"]().score(o, r)
    assert res.score == 0.0
    assert res.details["delta"] == 4.0


def test_cost_delta_replay_cheaper():
    o = [_span("llm", cost_usd=0.005)]
    r = [_span("llm", cost_usd=0.001)]
    res = RUBRICS["cost-delta"]().score(o, r)
    assert res.score == 1.0


def test_tool_seq_exact():
    o = [
        _span("tool", tool_name="calc", step_idx=0),
        _span("tool", tool_name="web_search", step_idx=1),
    ]
    r = [
        _span("tool", tool_name="calc", step_idx=0),
        _span("tool", tool_name="web_search", step_idx=1),
    ]
    assert RUBRICS["tool-seq"]().score(o, r).score == 1.0


def test_tool_seq_drift():
    o = [_span("tool", tool_name="calc"), _span("tool", tool_name="web_search")]
    r = [_span("tool", tool_name="calc")]
    res = RUBRICS["tool-seq"]().score(o, r)
    assert 0 < res.score < 1


def test_tool_args_match():
    o = [_span("tool", tool_name="calc", input={"expr": "1+1"})]
    r = [_span("tool", tool_name="calc", input={"expr": "1+1"})]
    assert RUBRICS["tool-args"]().score(o, r).score == 1.0


def test_unknown_rubric_raises():
    import pytest

    with pytest.raises(ValueError):
        run_rubrics(["does-not-exist"], [], [])
