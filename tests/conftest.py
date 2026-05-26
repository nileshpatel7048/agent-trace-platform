import tempfile

import pytest

from agent_trace_platform.store import TraceStore


@pytest.fixture
def store(tmp_path):
    return TraceStore(warehouse_path=str(tmp_path))


@pytest.fixture
def warehouse_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d
