from __future__ import annotations

import numpy as np
import pytest

import nanoback as nb


def test_run_backtest_ticks_executes() -> None:
    ts = np.array([1, 2, 3, 4], dtype=np.int64)
    asset = np.array([0, 0, 0, 0], dtype=np.int64)
    price = np.array([100.0, 100.5, 100.4, 100.8], dtype=np.float64)
    size = np.array([100.0, 120.0, 80.0, 200.0], dtype=np.float64)
    side = np.array([int(nb.TickSide.BID), int(nb.TickSide.ASK), int(nb.TickSide.TRADE), int(nb.TickSide.TRADE)], dtype=np.int8)
    targets = np.array([[0], [1], [1], [0]], dtype=np.int64)
    config = nb.BacktestConfig(max_position=1, slippage_bps=0.0, volume_share_impact=0.0)
    config.data_mode = nb.DataMode.TICK
    result = nb.run_backtest_ticks(
        timestamp_ns=ts,
        asset=asset,
        price=price,
        size=size,
        side=side,
        target_positions=targets,
        cols=1,
        config=config,
        symbols=["AAA"],
    )
    assert result.positions.shape == (4, 1)
    assert result.equity_curve.shape[0] == 4


def test_load_ticks_parquet_roundtrip(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame(
        {
            "timestamp": [2, 1],
            "symbol": ["AAA", "AAA"],
            "price": [100.2, 100.0],
            "size": [10.0, 11.0],
            "side": ["TRADE", "BID"],
        }
    )
    path = tmp_path / "ticks.parquet"
    frame.to_parquet(path)
    loaded = nb.load_ticks_parquet(path)
    assert loaded["timestamp_ns"][0] == 1
    assert loaded["asset"].shape[0] == 2
