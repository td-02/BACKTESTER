from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    return str(value)


def _series_summary(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


@dataclass(slots=True)
class PerformanceReport:
    metrics: dict[str, float | int]
    curve_summary: dict[str, dict[str, float | int]]
    audit_event_counts: dict[str, int]
    asset_execution: list[dict[str, object]]
    symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics,
            "curve_summary": self.curve_summary,
            "audit_event_counts": self.audit_event_counts,
            "asset_execution": self.asset_execution,
            "symbols": self.symbols,
        }

    def render_markdown(self) -> str:
        lines = ["# Backtest Performance Report", ""]
        lines.extend(
            [
                "## Key Metrics",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        for key, value in self.metrics.items():
            lines.append(f"| {key} | {value} |")
        lines.extend(["", "## Curve Summary", ""])
        for curve_name, summary in self.curve_summary.items():
            lines.append(f"### {curve_name}")
            lines.append("")
            lines.append("| Statistic | Value |")
            lines.append("| --- | ---: |")
            for key, value in summary.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")
        lines.extend(
            [
                "## Audit Events",
                "",
                "| Event | Count |",
                "| --- | ---: |",
            ]
        )
        for key, value in sorted(self.audit_event_counts.items()):
            lines.append(f"| {key} | {value} |")
        lines.extend(["", "## Asset Execution", "", "| Asset | Symbol | Fills | Abs Quantity | Fees |", "| --- | --- | ---: | ---: | ---: |"])
        for row in self.asset_execution:
            lines.append(
                f"| {row['asset']} | {row.get('symbol', '')} | {row['fills']} | {row['filled_quantity']} | {row['fees']} |"
            )
        lines.append("")
        return "\n".join(lines)


def summarize_backtest(result: object, symbols: Sequence[str] | None = None) -> PerformanceReport:
    fills = list(getattr(result, "fills", ()))
    audit_events = list(getattr(result, "audit_events", ()))
    equity_curve = np.asarray(getattr(result, "equity_curve", ()), dtype=np.float64)
    cash_curve = np.asarray(getattr(result, "cash_curve", ()), dtype=np.float64)

    ending_equity = float(getattr(result, "ending_equity"))
    pnl = float(getattr(result, "pnl"))
    starting_equity = ending_equity - pnl
    return_pct = (pnl / starting_equity) if starting_equity else 0.0

    metrics = {
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "pnl": pnl,
        "return_pct": return_pct,
        "turnover": float(getattr(result, "turnover", 0.0)),
        "total_fees": float(getattr(result, "total_fees", 0.0)),
        "total_borrow_cost": float(getattr(result, "total_borrow_cost", 0.0)),
        "total_cash_yield": float(getattr(result, "total_cash_yield", 0.0)),
        "peak_equity": float(getattr(result, "peak_equity", 0.0)),
        "max_drawdown": float(getattr(result, "max_drawdown", 0.0)),
        "submitted_orders": int(getattr(result, "submitted_orders", 0)),
        "filled_orders": int(getattr(result, "filled_orders", len(fills))),
        "rejected_orders": int(getattr(result, "rejected_orders", 0)),
        "fill_count": int(len(fills)),
        "audit_event_count": int(len(audit_events)),
        "fill_rate": (
            float(getattr(result, "filled_orders", len(fills))) / float(getattr(result, "submitted_orders", 0))
            if float(getattr(result, "submitted_orders", 0))
            else 0.0
        ),
    }

    curve_summary = {
        "equity_curve": _series_summary(equity_curve),
        "cash_curve": _series_summary(cash_curve),
    }

    audit_event_counts = Counter(_enum_name(event.type) for event in audit_events)

    symbol_list = list(symbols or getattr(result, "symbols", []))
    fill_counts: Counter[int] = Counter()
    fill_quantities: defaultdict[int, int] = defaultdict(int)
    fill_fees: defaultdict[int, float] = defaultdict(float)
    for fill in fills:
        asset = int(fill.asset)
        fill_counts[asset] += 1
        fill_quantities[asset] += abs(int(fill.quantity))
        fill_fees[asset] += float(fill.fee)

    asset_execution: list[dict[str, object]] = []
    asset_ids = sorted(set(fill_counts) | set(fill_quantities) | set(fill_fees))
    for asset in asset_ids:
        asset_execution.append(
            {
                "asset": asset,
                "symbol": symbol_list[asset] if asset < len(symbol_list) else "",
                "fills": int(fill_counts.get(asset, 0)),
                "filled_quantity": int(fill_quantities.get(asset, 0)),
                "fees": float(fill_fees.get(asset, 0.0)),
            }
        )

    return PerformanceReport(
        metrics=metrics,
        curve_summary=curve_summary,
        audit_event_counts=dict(audit_event_counts),
        asset_execution=asset_execution,
        symbols=symbol_list,
    )


def export_performance_report_json(report: PerformanceReport, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def export_performance_report_markdown(report: PerformanceReport, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.render_markdown(), encoding="utf-8")
