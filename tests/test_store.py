from agent_trace_platform.schema import Span, SpanKind


def _mkspan(run_id="r1", agent="a", step=0, kind=SpanKind.LLM):
    kwargs = dict(
        agent_id=agent,
        run_id=run_id,
        step_idx=step,
        span_kind=kind,
        cost_usd=0.001,
        tokens_in=10,
        tokens_out=5,
    )
    if kind == SpanKind.LLM:
        kwargs["model"] = "claude-haiku-4-5"
        kwargs["input"] = {"q": "hi"}
        kwargs["output"] = {"a": "ok"}
    elif kind == SpanKind.TOOL:
        kwargs["tool_name"] = "calc"
        kwargs["input"] = {"expr": "1+1"}
        kwargs["output"] = "2"
    return Span(**kwargs)


def test_append_and_get_run(store):
    spans = [_mkspan(step=0), _mkspan(step=1, kind=SpanKind.TOOL), _mkspan(step=2)]
    n = store.append(spans)
    assert n == 3
    rows = store.get_run("r1")
    assert len(rows) == 3
    assert [r["step_idx"] for r in rows] == [0, 1, 2]
    assert rows[1]["span_kind"] == "tool"
    # JSON round-trip happened
    assert rows[0]["input"] == {"q": "hi"}


def test_list_runs_aggregation(store):
    store.append([_mkspan(run_id="r1", step=0), _mkspan(run_id="r1", step=1)])
    store.append([_mkspan(run_id="r2", step=0)])
    runs = store.list_runs()
    by_id = {r["run_id"]: r for r in runs}
    assert by_id["r1"]["spans"] == 2
    assert by_id["r2"]["spans"] == 1
    assert by_id["r1"]["cost_usd"] == 0.002


def test_get_unknown_run_returns_empty(store):
    assert store.get_run("nope") == []
