from __future__ import annotations

from typing import Any


def metrics_table_row(run_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row = {"run_name": run_name}
    row.update(metrics)
    return row
