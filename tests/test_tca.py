from __future__ import annotations

import numpy as np
import pytest

import nanoback as nb


pd = pytest.importorskip("pandas")


def _sample_result(*, with_spread: bool) -> object:
    timestamps = np.array([1, 2, 3, 4], dtype=np.int64)
    prices = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float64)
    if with_spread:
        bid = prices - 0.5
        ask = prices + 0.5
    else:
        bid = prices.copy()
        ask = prices.copy()
    return nb.run_backtest(
        timestamps=timestamps,
        prices=prices,
        bid=bid,
        ask=ask,
        volume=np.array([10_000.0, 10_000.0, 10_000.0, 10_000.0], dtype=np.float64),
        signals=np.array([1, 1, -1, -1], dtype=np.int64),
        config=nb.BacktestConfig(
            max_position=1,
            slippage_bps=0.0,
            volume_share_impact=0.0,
            use_bid_ask_execution=True,
        ),
    )


def test_tca_components_sum_to_total_fill_cost() -> None:
    result = _sample_result(with_spread=True)
    df = nb.tca_dataframe(result, volatility_window=2, impact_coefficient=0.7)

    assert not df.empty
    components = df["spread_cost"] + df["impact_cost"] + df["financing_cost"]
    assert np.allclose(components.to_numpy(), df["total_cost"].to_numpy())


def test_tca_zero_spread_zero_impact_produces_zero_cost() -> None:
    result = _sample_result(with_spread=False)
    df = nb.tca_dataframe(result, volatility_window=2, impact_coefficient=0.0)

    assert not df.empty
    assert np.allclose(df["spread_cost"].to_numpy(), 0.0)
    assert np.allclose(df["impact_cost"].to_numpy(), 0.0)
    assert np.allclose(df["total_cost"].to_numpy(), 0.0)
