from __future__ import annotations

import numpy as np
import pytest

import nanoback as nb


def _data(rows: int = 80) -> nb.MarketData:
    timestamps = np.arange(rows, dtype=np.int64) + 1
    close = np.linspace(100.0, 120.0, rows, dtype=np.float64).reshape(-1, 1)
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


def test_param_grid_cartesian() -> None:
    grid = nb.ParamGrid({"a": [1, 2], "b": ["x", "y"]})
    combos = grid.combinations()
    assert len(combos) == 4


def test_sweep_runs_and_sorts() -> None:
    data = _data()
    sweep = nb.Sweep(data)
    result = sweep.run(_strategy, nb.ParamGrid({"lookback": [1, 2, 3], "max_position": [1]}), n_jobs=1)

    assert result.rows
    assert result.rows[0]["sharpe"] >= result.rows[-1]["sharpe"]


def test_sweep_oos_columns_present() -> None:
    data = _data()
    train = nb.MarketData(
        timestamps=data.timestamps[:60],
        close=data.close[:60],
        high=data.high[:60],
        low=data.low[:60],
        volume=data.volume[:60],
        bid=data.bid[:60],
        ask=data.ask[:60],
        symbols=data.symbols,
    )
    oos = nb.MarketData(
        timestamps=data.timestamps[60:],
        close=data.close[60:],
        high=data.high[60:],
        low=data.low[60:],
        volume=data.volume[60:],
        bid=data.bid[60:],
        ask=data.ask[60:],
        symbols=data.symbols,
    )
    sweep = nb.Sweep(train)
    result = sweep.run(_strategy, {"lookback": [1, 2], "max_position": [1]}, n_jobs=1, oos_data=oos)
    assert "oos_sharpe" in result.rows[0]
    assert "overfit_warning" in result.rows[0]


def test_sweep_parallel_fallback_on_executor_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _data()
    sweep = nb.Sweep(data)

    class _BrokenExecutor:
        def __init__(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, *args, **kwargs):
            raise RuntimeError("pickle failure")

    monkeypatch.setattr("nanoback.sweep.ProcessPoolExecutor", _BrokenExecutor)
    result = sweep.run(_strategy, {"lookback": [1, 2], "max_position": [1]}, n_jobs=2)
    assert len(result.rows) == 2
