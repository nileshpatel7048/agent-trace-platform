"""End-to-end: append a run via store, replay it, score with eval rubrics."""
from agent_trace_platform.eval import run_rubrics
from agent_trace_platform.replay import ReplayEngine
from agent_trace_platform.schema import Span, SpanKind


def _seed_run(store, run_id="r1"):
    spans = [
        Span(
            agent_id="a",
            run_id=run_id,
            step_idx=0,
            span_kind=SpanKind.LLM,
            model="claude-haiku-4-5",
            input={"messages": [{"role": "user", "content": "calc 2+2"}]},
            output={"content": [{"type": "text", "text": "use calc tool"}]},
            tokens_in=8,
            tokens_out=4,
            cost_usd=0.0001,
        ),
        Span(
            agent_id="a",
            run_id=run_id,
            step_idx=1,
            span_kind=SpanKind.TOOL,
            tool_name="calc",
            input={"expr": "2+2"},
            output="4",
        ),
        Span(
            agent_id="a",
            run_id=run_id,
            step_idx=2,
            span_kind=SpanKind.LLM,
            model="claude-haiku-4-5",
            input={"messages": [{"role": "user", "content": "synthesize"}]},
            output={"content": [{"type": "text", "text": "the answer is 4"}]},
            tokens_in=10,
            tokens_out=6,
            cost_usd=0.0002,
        ),
    ]
    store.append(spans)
    return run_id


def test_replay_preserves_tool_outputs(store):
    run_id = _seed_run(store)
    engine = ReplayEngine(store)
    result = engine.replay(run_id, model_override="claude-sonnet-4-6")
    assert len(result.spans) == 3
    # tool span keeps cached output -> deterministic
    tool_spans = [s for s in result.spans if s.span_kind == SpanKind.TOOL]
    assert len(tool_spans) == 1
    assert tool_spans[0].output == "4"
    assert tool_spans[0].input == {"expr": "2+2"}


def test_replay_costs_differ_when_model_changes(store):
    run_id = _seed_run(store)
    engine = ReplayEngine(store)
    result = engine.replay(run_id, model_override="claude-sonnet-4-6")
    # Sonnet is pricier than Haiku per token; replay cost should be > original
    assert result.cost_usd_replay > result.cost_usd_original


def test_eval_after_replay(store):
    run_id = _seed_run(store)
    engine = ReplayEngine(store)
    result = engine.replay(run_id, model_override="claude-haiku-4-5")
    original = store.get_run(run_id)
    replay = store.get_run(result.replay_run_id)
    scores = {r.name: r.score for r in run_rubrics(["tool-seq", "tool-args"], original, replay)}
    assert scores["tool-seq"] == 1.0
    assert scores["tool-args"] == 1.0


def test_replay_unknown_run_raises(store):
    engine = ReplayEngine(store)
    import pytest

    with pytest.raises(ValueError):
        engine.replay("does-not-exist")
