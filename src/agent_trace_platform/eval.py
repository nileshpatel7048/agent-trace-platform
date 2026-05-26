"""Eval rubrics. Each rubric: score(original_spans, replay_spans) -> RubricResult.

See ARCHITECTURE.md section 6.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from .replay import final_answer


@dataclass
class RubricResult:
    name: str
    score: float  # 0.0 = bad, 1.0 = perfect (for delta-style rubrics, see `details`)
    details: dict[str, Any] = field(default_factory=dict)


class Rubric(Protocol):
    name: str

    def score(
        self, original: list[dict[str, Any]], replay: list[dict[str, Any]]
    ) -> RubricResult: ...


class ExactMatch:
    name = "exact"

    def score(self, original, replay):
        o = (final_answer(original) or "").strip()
        r = (final_answer(replay) or "").strip()
        return RubricResult(self.name, 1.0 if o == r else 0.0, {"original": o, "replay": r})


class SemanticSimilarity:
    """Cosine similarity on cheap bag-of-tokens embeddings.

    Production: swap _embed for OpenAI/Voyage. The interface is the only thing
    that matters here — keeping zero-network for tests.
    """

    name = "semantic"

    def _embed(self, text: str) -> dict[str, float]:
        # bag-of-words term frequency; good enough for unit tests and the demo
        toks = [t for t in text.lower().split() if t]
        if not toks:
            return {}
        v: dict[str, float] = {}
        for t in toks:
            v[t] = v.get(t, 0.0) + 1.0
        return v

    @staticmethod
    def _cos(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[k] * b.get(k, 0.0) for k in a)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def score(self, original, replay):
        o = final_answer(original) or ""
        r = final_answer(replay) or ""
        sim = self._cos(self._embed(o), self._embed(r))
        return RubricResult(self.name, sim, {"original": o[:200], "replay": r[:200]})


class CostDelta:
    name = "cost-delta"

    def score(self, original, replay):
        oc = sum(s.get("cost_usd") or 0.0 for s in original)
        rc = sum(s.get("cost_usd") or 0.0 for s in replay)
        delta = (rc - oc) / oc if oc > 0 else 0.0
        # Lower cost is "better". Map to [0,1] where 1=replay was free relative to original.
        # Clamp to [-1, +inf]; score = max(0, 1 - max(delta, 0))
        score = max(0.0, 1.0 - max(delta, 0.0))
        return RubricResult(self.name, score, {"original_usd": oc, "replay_usd": rc, "delta": delta})


class LatencyDelta:
    name = "latency-delta"

    def score(self, original, replay):
        ol = sum((s["ts_end_ns"] - s["ts_start_ns"]) / 1e6 for s in original)
        rl = sum((s["ts_end_ns"] - s["ts_start_ns"]) / 1e6 for s in replay)
        delta = (rl - ol) / ol if ol > 0 else 0.0
        score = max(0.0, 1.0 - max(delta, 0.0))
        return RubricResult(self.name, score, {"original_ms": ol, "replay_ms": rl, "delta": delta})


class ToolSequence:
    """Normalized Levenshtein between tool-name sequences."""

    name = "tool-seq"

    @staticmethod
    def _lev(a: list[str], b: list[str]) -> int:
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                cur[j] = min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            prev = cur
        return prev[-1]

    def score(self, original, replay):
        os_ = [s["tool_name"] for s in original if s["span_kind"] == "tool"]
        rs = [s["tool_name"] for s in replay if s["span_kind"] == "tool"]
        if not os_ and not rs:
            return RubricResult(self.name, 1.0, {"original": [], "replay": []})
        dist = self._lev(os_, rs)
        denom = max(len(os_), len(rs)) or 1
        return RubricResult(
            self.name, 1.0 - dist / denom, {"original": os_, "replay": rs, "dist": dist}
        )


class ToolArgsMatch:
    name = "tool-args"

    def score(self, original, replay):
        os_ = [s for s in original if s["span_kind"] == "tool"]
        rs = [s for s in replay if s["span_kind"] == "tool"]
        n = min(len(os_), len(rs))
        if n == 0:
            return RubricResult(self.name, 1.0 if not os_ and not rs else 0.0, {"matched": 0, "n": 0})
        matched = 0
        for a, b in zip(os_, rs, strict=False):
            ai = json.dumps(a.get("input"), sort_keys=True, default=str)
            bi = json.dumps(b.get("input"), sort_keys=True, default=str)
            if ai == bi:
                matched += 1
        return RubricResult(self.name, matched / n, {"matched": matched, "n": n})


RUBRICS: dict[str, type[Rubric]] = {
    r.name: r  # type: ignore[misc]
    for r in [ExactMatch, SemanticSimilarity, CostDelta, LatencyDelta, ToolSequence, ToolArgsMatch]
}


def run_rubrics(
    names: list[str],
    original: list[dict[str, Any]],
    replay: list[dict[str, Any]],
) -> list[RubricResult]:
    out = []
    for n in names:
        cls = RUBRICS.get(n)
        if cls is None:
            raise ValueError(f"unknown rubric: {n}. available: {sorted(RUBRICS)}")
        out.append(cls().score(original, replay))
    return out
