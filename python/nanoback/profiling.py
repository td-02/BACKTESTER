from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import time
from typing import Iterator


@dataclass(slots=True)
class LatencySample:
    stage: str
    elapsed_seconds: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class LatencySummary:
    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    max: float
    total: float


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


@dataclass(slots=True)
class LatencyLogBook:
    scenario: str
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    seed: int | None = None
    samples: list[LatencySample] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def record(self, stage: str, elapsed_seconds: float, **metadata: object) -> None:
        self.samples.append(
            LatencySample(
                stage=stage,
                elapsed_seconds=float(elapsed_seconds),
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )

    @contextmanager
    def timing(self, stage: str, **metadata: object) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, time.perf_counter() - started, **metadata)

    def stage_summaries(self) -> dict[str, LatencySummary]:
        grouped: dict[str, list[float]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.stage, []).append(sample.elapsed_seconds)

        summaries: dict[str, LatencySummary] = {}
        for stage, values in grouped.items():
            summaries[stage] = LatencySummary(
                count=len(values),
                mean=mean(values),
                p50=_percentile(values, 0.50),
                p95=_percentile(values, 0.95),
                p99=_percentile(values, 0.99),
                max=max(values),
                total=sum(values),
            )
        return summaries

    def total_seconds(self) -> float:
        return sum(sample.elapsed_seconds for sample in self.samples)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "seed": self.seed,
            "metadata": self.metadata,
            "samples": [
                {
                    "stage": sample.stage,
                    "elapsed_seconds": sample.elapsed_seconds,
                    "metadata": sample.metadata,
                }
                for sample in self.samples
            ],
            "stage_summaries": {
                stage: {
                    "count": summary.count,
                    "mean": summary.mean,
                    "p50": summary.p50,
                    "p95": summary.p95,
                    "p99": summary.p99,
                    "max": summary.max,
                    "total": summary.total,
                }
                for stage, summary in self.stage_summaries().items()
            },
            "total_seconds": self.total_seconds(),
        }

    def write_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(
                    json.dumps(
                        {
                            "run_id": self.run_id,
                            "scenario": self.scenario,
                            "seed": self.seed,
                            "stage": sample.stage,
                            "elapsed_seconds": sample.elapsed_seconds,
                            "metadata": sample.metadata,
                        }
                    )
                )
                handle.write("\n")

    def render_text(self) -> str:
        lines = [
            f"run_id={self.run_id} scenario={self.scenario} seed={self.seed}",
        ]
        if self.metadata:
            lines.append(" ".join(f"{key}={value}" for key, value in sorted(self.metadata.items())))
        lines.append("stage                     count      mean       p95       p99       max      total")
        for stage, summary in sorted(self.stage_summaries().items()):
            lines.append(
                f"{stage:<24} {summary.count:>5} "
                f"{summary.mean:>9.6f} {summary.p95:>9.6f} {summary.p99:>9.6f} {summary.max:>9.6f} {summary.total:>9.6f}"
            )
        lines.append(f"total_seconds={self.total_seconds():.6f}")
        return "\n".join(lines)
