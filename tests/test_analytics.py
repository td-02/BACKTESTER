from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import nanoback as nb


def _make_result(equity: list[float]) -> object:
    ledger = [
        SimpleNamespace(
            timestamp=int(idx + 1),
            equity_after=float(value),
            sequence=idx + 1,
        )
        for idx, value in enumerate(equity)
    ]
    return SimpleNamespace(
        ledger=ledger,
        fills=[],
        equity_curve=np.asarray(equity, dtype=np.float64),
        symbols=["asset_0"],
    )


def test_flat_equity_curve_has_zero_sharpe_and_drawdown() -> None:
    result = _make_result([100.0, 100.0, 100.0, 100.0])
    summary = nb.summarize_result(result)

    assert summary.sharpe == 0.0
    assert summary.max_drawdown == 0.0
    assert np.allclose(summary.drawdown_series, 0.0)


def test_linearly_rising_curve_has_infinite_sharpe() -> None:
    result = _make_result([100.0, 101.0, 102.01, 103.0301, 104.060401])
    summary = nb.summarize_result(result)

    assert np.isinf(summary.sharpe) or summary.sharpe > 1e6


def test_peak_to_trough_50pct_drawdown() -> None:
    result = _make_result([100.0, 120.0, 60.0, 70.0])
    summary = nb.summarize_result(result)

    assert summary.max_drawdown == 0.5
