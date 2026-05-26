"""Model adapters. Pluggable per provider.

The real Anthropic client is used when ANTHROPIC_API_KEY is set; otherwise a
deterministic stub adapter returns canned responses so the demo and tests run
offline. Cost numbers below are rough public list prices as of late 2025 —
in a real deployment this lives in a config file or pricing service.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Usage:
    tokens_in: int
    tokens_out: int
    cost_usd: float


PRICING_PER_MTOK = {
    # input, output USD per 1M tokens (approximate, public list)
    "claude-opus-4-7":    (15.0, 75.0),
    "claude-sonnet-4-6":  (3.0,  15.0),
    "claude-haiku-4-5":   (0.8,  4.0),
    "gpt-4o":             (2.5,  10.0),
    "gpt-4o-mini":        (0.15, 0.6),
}


def _price(model: str, tokens_in: int, tokens_out: int) -> float:
    for key, (pin, pout) in PRICING_PER_MTOK.items():
        if model.startswith(key):
            return (tokens_in / 1e6) * pin + (tokens_out / 1e6) * pout
    return 0.0


class ModelAdapter(Protocol):
    def complete(self, model: str, request: Any) -> tuple[Any, Usage]: ...


class AnthropicAdapter:
    def __init__(self) -> None:
        from anthropic import Anthropic

        self.client = Anthropic()

    def complete(self, model: str, request: Any) -> tuple[dict[str, Any], Usage]:
        # `request` is whatever was stored as `llm.input` — accept either a full
        # messages payload or a {"messages": [...], "system": "..."} dict.
        kwargs: dict[str, Any] = {"model": model, "max_tokens": 1024}
        if isinstance(request, dict):
            kwargs.update({k: v for k, v in request.items() if k != "model"})
        else:
            kwargs["messages"] = [{"role": "user", "content": str(request)}]
        kwargs.setdefault("max_tokens", 1024)

        resp = self.client.messages.create(**kwargs)
        out = {
            "content": [
                {"type": b.type, "text": getattr(b, "text", "")} for b in resp.content
            ],
            "stop_reason": resp.stop_reason,
        }
        usage = Usage(
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            cost_usd=_price(model, resp.usage.input_tokens, resp.usage.output_tokens),
        )
        return out, usage


class StubAdapter:
    """Deterministic offline adapter. Used when no API key is available."""

    def complete(self, model: str, request: Any) -> tuple[dict[str, Any], Usage]:
        payload = json.dumps(request, sort_keys=True, default=str)
        h = hashlib.sha256(payload.encode()).hexdigest()[:8]
        text = f"[stub:{model}:{h}] response to request of {len(payload)} chars"
        tokens_in = max(1, len(payload) // 4)
        tokens_out = max(1, len(text) // 4)
        usage = Usage(
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=_price(model, tokens_in, tokens_out),
        )
        return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}, usage


def get_adapter(model: str) -> ModelAdapter:
    if model.startswith("claude") and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicAdapter()
    return StubAdapter()
