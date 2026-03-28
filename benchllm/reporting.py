from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from typing import Iterable

from benchllm.runner import BenchmarkResult


@dataclass(frozen=True)
class SummaryRow:
    profile_id: str
    workload_id: str
    samples: int
    median_ttft_ms: float
    median_decode_tokens_per_second: float
    validation_pass_rate: float


def load_results(path: str | Path) -> list[BenchmarkResult]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [BenchmarkResult(**row) for row in rows]


def summarize_results(results: Iterable[BenchmarkResult]) -> list[SummaryRow]:
    grouped: dict[tuple[str, str], list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[(result.profile_id, result.workload_id)].append(result)

    rows: list[SummaryRow] = []
    for (profile_id, workload_id), items in sorted(grouped.items()):
        rows.append(
            SummaryRow(
                profile_id=profile_id,
                workload_id=workload_id,
                samples=len(items),
                median_ttft_ms=round(median(item.ttft_ms for item in items), 3),
                median_decode_tokens_per_second=round(
                    median(item.decode_tokens_per_second for item in items),
                    3,
                ),
                validation_pass_rate=round(
                    sum(1 for item in items if item.validation_passed) / len(items),
                    3,
                ),
            )
        )
    return rows
