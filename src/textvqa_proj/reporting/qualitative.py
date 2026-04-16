from __future__ import annotations

from textvqa_proj.inference.run_store import PredictionRecord


def collect_matches(records: list[PredictionRecord], *, matched: bool) -> list[PredictionRecord]:
    if matched:
        return [record for record in records if record.any_match >= 1.0]
    return [record for record in records if record.any_match < 1.0]
