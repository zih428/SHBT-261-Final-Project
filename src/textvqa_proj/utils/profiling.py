from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def time_block() -> Iterator[dict[str, float]]:
    state: dict[str, float] = {}
    start = time.perf_counter()
    yield state
    state["elapsed_seconds"] = time.perf_counter() - start
