"""Iceberg-backed trace store.

Local mode: SQLite catalog + Parquet warehouse on the filesystem.
Prod path: swap catalog impl to Glue/Polaris/Nessie. Schema and code unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, TableAlreadyExistsError
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import BucketTransform, IdentityTransform
from pyiceberg.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    NestedField,
    StringType,
)

from .schema import Span

NAMESPACE = "traces"
TABLE = "spans"
FQN = f"{NAMESPACE}.{TABLE}"

ICEBERG_SCHEMA = Schema(
    NestedField(1, "trace_id", StringType(), required=True),
    NestedField(2, "run_id", StringType(), required=True),
    NestedField(3, "span_id", StringType(), required=True),
    NestedField(4, "parent_span_id", StringType(), required=False),
    NestedField(5, "agent_id", StringType(), required=True),
    NestedField(6, "step_idx", IntegerType(), required=True),
    NestedField(7, "span_kind", StringType(), required=True),
    NestedField(8, "ts_start_ns", LongType(), required=True),
    NestedField(9, "ts_end_ns", LongType(), required=True),
    NestedField(10, "ts_date", DateType(), required=True),
    NestedField(11, "model", StringType(), required=False),
    NestedField(12, "tool_name", StringType(), required=False),
    NestedField(13, "input", StringType(), required=False),
    NestedField(14, "output", StringType(), required=False),
    NestedField(15, "tokens_in", IntegerType(), required=False),
    NestedField(16, "tokens_out", IntegerType(), required=False),
    NestedField(17, "cost_usd", DoubleType(), required=False),
    NestedField(18, "error", StringType(), required=False),
    NestedField(19, "attrs", StringType(), required=False),
)

PARTITION_SPEC = PartitionSpec(
    PartitionField(source_id=10, field_id=1000, transform=IdentityTransform(), name="ts_date"),
    PartitionField(
        source_id=5, field_id=1001, transform=BucketTransform(num_buckets=16), name="agent_bucket"
    ),
)


class TraceStore:
    def __init__(self, warehouse_path: str | None = None):
        warehouse_path = warehouse_path or os.environ.get("ATP_WAREHOUSE", "./warehouse")
        self.warehouse_path = Path(warehouse_path).resolve()
        self.warehouse_path.mkdir(parents=True, exist_ok=True)
        self.catalog: Catalog = load_catalog(
            "default",
            **{
                "type": "sql",
                "uri": f"sqlite:///{self.warehouse_path}/catalog.db",
                "warehouse": f"file://{self.warehouse_path}",
            },
        )
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            self.catalog.create_namespace(NAMESPACE)
        except NamespaceAlreadyExistsError:
            pass
        try:
            self.catalog.create_table(
                identifier=FQN,
                schema=ICEBERG_SCHEMA,
                partition_spec=PARTITION_SPEC,
            )
        except TableAlreadyExistsError:
            pass

    def append(self, spans: list[Span]) -> int:
        if not spans:
            return 0
        rows = [s.to_row() for s in spans]
        arrow_table = _rows_to_arrow(rows)
        table = self.catalog.load_table(FQN)
        table.append(arrow_table)
        return len(rows)

    def get_run(self, run_id: str) -> list[dict[str, Any]]:
        """Return all spans for a run, ordered by step_idx."""
        table = self.catalog.load_table(FQN)
        scan = table.scan(row_filter=f"run_id = '{run_id}'")
        df = scan.to_arrow().to_pylist()
        df.sort(key=lambda r: (r["step_idx"], r["ts_start_ns"]))
        for r in df:
            if r.get("input"):
                r["input"] = json.loads(r["input"])
            if r.get("output"):
                r["output"] = json.loads(r["output"])
            if r.get("attrs"):
                r["attrs"] = json.loads(r["attrs"])
        return df

    def list_runs(
        self, agent_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Distinct (agent_id, run_id) with summary stats."""
        table = self.catalog.load_table(FQN)
        if agent_id:
            scan = table.scan(row_filter=f"agent_id = '{agent_id}'")
        else:
            scan = table.scan()
        arrow_table = scan.to_arrow()
        if arrow_table.num_rows == 0:
            return []
        # Push down via DuckDB for the aggregation — Iceberg scan + Arrow + DuckDB SQL.
        import duckdb

        con = duckdb.connect(":memory:")
        con.register("spans", arrow_table)
        res = con.execute(
            """
            SELECT
                run_id,
                agent_id,
                COUNT(*) AS spans,
                MIN(ts_start_ns) AS started_ns,
                MAX(ts_end_ns)  AS ended_ns,
                SUM(COALESCE(cost_usd, 0))    AS cost_usd,
                SUM(COALESCE(tokens_in, 0))   AS tokens_in,
                SUM(COALESCE(tokens_out, 0))  AS tokens_out
            FROM spans
            GROUP BY run_id, agent_id
            ORDER BY started_ns DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        cols = ["run_id", "agent_id", "spans", "started_ns", "ended_ns",
                "cost_usd", "tokens_in", "tokens_out"]
        return [dict(zip(cols, row, strict=True)) for row in res]


def _rows_to_arrow(rows: list[dict[str, Any]]) -> pa.Table:
    arrow_schema = pa.schema([
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("span_id", pa.string(), nullable=False),
        pa.field("parent_span_id", pa.string(), nullable=True),
        pa.field("agent_id", pa.string(), nullable=False),
        pa.field("step_idx", pa.int32(), nullable=False),
        pa.field("span_kind", pa.string(), nullable=False),
        pa.field("ts_start_ns", pa.int64(), nullable=False),
        pa.field("ts_end_ns", pa.int64(), nullable=False),
        pa.field("ts_date", pa.date32(), nullable=False),
        pa.field("model", pa.string(), nullable=True),
        pa.field("tool_name", pa.string(), nullable=True),
        pa.field("input", pa.string(), nullable=True),
        pa.field("output", pa.string(), nullable=True),
        pa.field("tokens_in", pa.int32(), nullable=True),
        pa.field("tokens_out", pa.int32(), nullable=True),
        pa.field("cost_usd", pa.float64(), nullable=True),
        pa.field("error", pa.string(), nullable=True),
        pa.field("attrs", pa.string(), nullable=True),
    ])
    cols: dict[str, list[Any]] = {f.name: [] for f in arrow_schema}
    for r in rows:
        for f in arrow_schema:
            cols[f.name].append(r.get(f.name))
    return pa.table(cols, schema=arrow_schema)
