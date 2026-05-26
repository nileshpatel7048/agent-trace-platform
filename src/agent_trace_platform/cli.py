"""CLI entry point: `atp <command>`."""
from __future__ import annotations

import json
import sys

import click
import httpx
import uvicorn

from .demo_agent import DemoAgent


@click.group()
def main() -> None:
    """agent-trace-platform CLI."""


@main.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8080, type=int)
@click.option("--warehouse", default="./warehouse")
def serve(host: str, port: int, warehouse: str) -> None:
    """Start the ingest + replay API."""
    import os

    os.environ["ATP_WAREHOUSE"] = warehouse
    uvicorn.run("agent_trace_platform.server:create_app", host=host, port=port, factory=True)


@main.group()
def demo() -> None:
    """Demo agent commands."""


@demo.command("agent")
@click.option("--url", default="http://127.0.0.1:8080")
@click.option("--model", default="claude-haiku-4-5")
@click.option("--prompt", default="What's 17 * 23 plus the SF weather?")
def demo_agent(url: str, model: str, prompt: str) -> None:
    """Run the demo agent once. Emits spans to the platform."""
    plan = [
        {"tool": "calc", "args": {"expr": "17 * 23"}},
        {"tool": "web_search", "args": {"query": "weather sf"}},
    ]
    agent = DemoAgent(platform_url=url, model=model)
    result = agent.run(prompt, plan=plan)
    click.echo(f"run_id: {result['run_id']}")
    click.echo(f"spans: {len(result['spans'])}")


@main.group()
def query() -> None:
    """Query the platform."""


@query.command("runs")
@click.option("--url", default="http://127.0.0.1:8080")
@click.option("--agent", default=None)
@click.option("--limit", default=20, type=int)
def query_runs(url: str, agent: str | None, limit: int) -> None:
    params = {"limit": limit}
    if agent:
        params["agent"] = agent
    r = httpx.get(f"{url}/v1/runs", params=params, timeout=10.0)
    r.raise_for_status()
    click.echo(json.dumps(r.json(), indent=2, default=str))


@main.command()
@click.argument("run_id")
@click.option("--url", default="http://127.0.0.1:8080")
@click.option("--model", default=None, help="Model override for replay")
def replay(run_id: str, url: str, model: str | None) -> None:
    """Replay a run with optional model override."""
    body = {"model_override": model, "live_tools": False}
    r = httpx.post(f"{url}/v1/runs/{run_id}/replay", json=body, timeout=60.0)
    r.raise_for_status()
    click.echo(json.dumps(r.json(), indent=2))


@main.command()
@click.argument("run_id")
@click.option("--replay-run", required=True)
@click.option("--url", default="http://127.0.0.1:8080")
@click.option(
    "--rubric",
    default="exact,semantic,cost-delta,latency-delta,tool-seq",
    help="Comma-separated rubric names.",
)
def eval(run_id: str, replay_run: str, url: str, rubric: str) -> None:
    """Score replay vs original."""
    rubrics = [r.strip() for r in rubric.split(",") if r.strip()]
    body = {"replay_run_id": replay_run, "rubrics": rubrics}
    r = httpx.post(f"{url}/v1/runs/{run_id}/eval", json=body, timeout=60.0)
    r.raise_for_status()
    click.echo(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    main()
