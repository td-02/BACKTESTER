from __future__ import annotations

import numpy as np
import pytest

import nanoback as nb


def _result() -> object:
    return nb.run_backtest(
        timestamps=np.array([1, 2, 3, 4], dtype=np.int64),
        prices=np.array([100.0, 101.0, 99.0, 102.0], dtype=np.float64),
        signals=np.array([1, 1, 0, -1], dtype=np.int64),
        config=nb.BacktestConfig(max_position=2, slippage_bps=0.0, volume_share_impact=0.0),
    )


def test_result_plot_and_html_export(tmp_path) -> None:
    pytest.importorskip("plotly")
    res = _result()
    fig = res.plot()
    assert len(fig.data) >= 2

    output = res.to_html(tmp_path / "report.html")
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "nanoback Backtest Report" in text


def test_wfo_plot() -> None:
    pytest.importorskip("plotly")
    timestamps = np.arange(80, dtype=np.int64) + 1
    close = (100.0 + np.cumsum(np.ones(80) * 0.1)).reshape(-1, 1)
    data = nb.MarketData(
        timestamps=timestamps,
        close=close,
        high=close,
        low=close,
        volume=np.full_like(close, 1000.0),
        bid=close,
        ask=close,
        symbols=["asset_0"],
    )

    def strategy(md: nb.MarketData, lookback: int, max_position: int):
        return nb.run_compiled_policy_backtest(
            md,
            policy="momentum",
            lookback=lookback,
            max_position=max_position,
            config=nb.BacktestConfig(max_position=max_position, slippage_bps=0.0, volume_share_impact=0.0),
        )

    wfo = nb.WalkForward(n_splits=3, train_frac=0.7, anchored=False)
    result = wfo.run(data, strategy, {"lookback": [1, 2], "max_position": [1]}, n_jobs=1, compiled=True)
    fig = result.plot()
    assert len(fig.data) >= 3
