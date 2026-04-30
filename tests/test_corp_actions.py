from __future__ import annotations

import numpy as np

import nanoback as nb


def _split_action(asset: int, ts: int, ratio: float) -> nb.CorporateAction:
    action = nb.CorporateAction()
    action.asset = asset
    action.ex_date_timestamp = ts
    action.action_type = nb.CorporateActionType.SPLIT
    action.ratio_or_amount = ratio
    return action


def test_split_doubles_position_and_preserves_equity_when_not_tradable() -> None:
    timestamps = np.array([1, 2, 3], dtype=np.int64)
    prices = np.array([100.0, 100.0, 50.0], dtype=np.float64)
    targets = np.array([1, 1, 1], dtype=np.int64)
    config = nb.BacktestConfig(max_position=2, slippage_bps=0.0, volume_share_impact=0.0)
    config.corporate_actions = [_split_action(asset=0, ts=3, ratio=2.0)]

    result = nb.run_backtest(
        timestamps=timestamps,
        prices=prices,
        signals=targets,
        tradable_mask=np.array([1, 1, 0], dtype=np.uint8),
        config=config,
    )

    # Equity should not jump from split mechanics alone.
    assert abs(float(result.equity_curve[2]) - float(result.equity_curve[1])) < 1e-9
    # Position should double on split date.
    assert int(result.positions[2, 0]) == 2
    assert float(result.adjustment_factors[2, 0]) == 0.5


def test_split_with_flat_position_has_zero_pnl_impact() -> None:
    timestamps = np.array([1, 2, 3], dtype=np.int64)
    prices = np.array([100.0, 50.0, 50.0], dtype=np.float64)
    config = nb.BacktestConfig(max_position=1, slippage_bps=0.0, volume_share_impact=0.0)
    config.corporate_actions = [_split_action(asset=0, ts=2, ratio=2.0)]

    result = nb.run_backtest(
        timestamps=timestamps,
        prices=prices,
        signals=np.array([0, 0, 0], dtype=np.int64),
        config=config,
    )

    assert float(result.pnl) == 0.0
