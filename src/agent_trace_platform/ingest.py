"""Ingest path. FastAPI route + a small in-memory batcher.

The batcher exists because Iceberg appends produce one Parquet file per call.
At 5k spans/sec with no batching we'd produce 5k tiny files per second; that's
a compaction nightmare. The batcher coalesces by time (50ms) or size (256 spans),
whichever hits first.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .schema import Span
from .store import TraceStore

log = logging.getLogger(__name__)


@dataclass
class BatcherConfig:
    max_batch_size: int = 256
    max_delay_ms: int = 50


@dataclass
class _State:
    buf: list[Span] = field(default_factory=list)
    flush_event: asyncio.Event = field(default_factory=asyncio.Event)


class IngestBatcher:
    def __init__(self, store: TraceStore, cfg: BatcherConfig | None = None):
        self.store = store
        self.cfg = cfg or BatcherConfig()
        self._state = _State()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        self._state.flush_event.set()
        if self._task:
            await self._task

    async def submit(self, spans: list[Span]) -> None:
        async with self._lock:
            self._state.buf.extend(spans)
            if len(self._state.buf) >= self.cfg.max_batch_size:
                self._state.flush_event.set()

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._state.flush_event.wait(), timeout=self.cfg.max_delay_ms / 1000
                )
            except TimeoutError:
                pass
            await self._flush()

        # final drain
        await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if not self._state.buf:
                self._state.flush_event.clear()
                return
            batch = self._state.buf
            self._state.buf = []
            self._state.flush_event.clear()
        # Iceberg writes are sync; offload to thread to avoid blocking the loop.
        await asyncio.to_thread(self.store.append, batch)
        log.debug("flushed %d spans", len(batch))


@asynccontextmanager
async def lifespan_batcher(store: TraceStore, cfg: BatcherConfig | None = None):
    batcher = IngestBatcher(store, cfg)
    await batcher.start()
    try:
        yield batcher
    finally:
        await batcher.stop()
