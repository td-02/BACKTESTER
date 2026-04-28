from __future__ import annotations

import numpy as np

import nanoback as nb


def _data(rows: int = 120) -> nb.MarketData:
    timestamps = np.arange(rows, dtype=np.int64) + 1
    close = (100.0 + np.cumsum(np.ones(rows, dtype=np.float64) * 0.2)).reshape(-1, 1)
    return nb.MarketData(
        timestamps=timestamps,
        close=close,
        high=close,
        low=close,
        volume=np.full_like(close, 10_000.0),
        bid=close,
        ask=close,
        symbols=["asset_0"],
    )


def _strategy(data: nb.MarketData, lookback: int, max_position: int):
    return nb.run_compiled_policy_backtest(
        data,
        policy="momentum",
        lookback=lookback,
        max_position=max_position,
        config=nb.BacktestConfig(max_position=max_position, slippage_bps=0.0, volume_share_impact=0.0),
    )


def test_walkforward_outputs_folds_and_efficiency() -> None:
    data = _data()
    wfo = nb.WalkForward(n_splits=4, train_frac=0.7, anchored=False)
    result = wfo.run(data, _strategy, {"lookback": [1, 2], "max_position": [1]}, n_jobs=1, compiled=True)

    assert result.folds
    assert result.oos_equity_curve.size > 0
    assert isinstance(result.efficiency_ratio, float)
