from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import nanoback as nb


def test_performance_report_summarizes_backtest(tmp_path: Path) -> None:
    result = nb.run_backtest(
        timestamps=np.array([1, 2, 3, 4], dtype=np.int64),
        prices=np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float64),
        signals=np.array([1, 1, 0, -1], dtype=np.int64),
        config=nb.BacktestConfig(
            max_position=2,
            child_order_size=1,
            child_slice_delay_steps=1,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )

    report = nb.summarize_backtest(result, symbols=["AAA"])

    assert report.metrics["fill_count"] == len(result.fills)
    assert report.metrics["submitted_orders"] == result.submitted_orders
    assert report.metrics["ending_equity"] == result.ending_equity
    assert "equity_curve" in report.curve_summary
    assert report.asset_execution

    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    nb.export_performance_report_json(report, json_path)
    nb.export_performance_report_markdown(report, md_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["metrics"]["fill_count"] == len(result.fills)
    assert "# Backtest Performance Report" in md_path.read_text(encoding="utf-8")


def test_performance_report_counts_audit_events() -> None:
    result = nb.run_backtest(
        timestamps=np.array([1, 2], dtype=np.int64),
        prices=np.array([100.0, 99.0], dtype=np.float64),
        signals=np.array([1, -1], dtype=np.int64),
        config=nb.BacktestConfig(
            max_position=1,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )

    report = nb.summarize_backtest(result)

    assert report.audit_event_counts
    assert sum(report.audit_event_counts.values()) == len(result.audit_events)
