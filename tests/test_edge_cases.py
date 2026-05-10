from __future__ import annotations

import numpy as np
import pytest

import nanoback as nb


def test_zero_liquidity_bar_queues_without_fill() -> None:
    result = nb.run_backtest_matrix(
        timestamps=np.array([1], dtype=np.int64),
        close=np.array([[100.0]], dtype=np.float64),
        high=np.array([[100.0]], dtype=np.float64),
        low=np.array([[100.0]], dtype=np.float64),
        volume=np.array([[0.0]], dtype=np.float64),
        target_positions=np.array([[1]], dtype=np.int64),
        config=nb.BacktestConfig(max_position=5, slippage_bps=0.0, volume_share_impact=0.0),
    )
    assert len(result.fills) == 0
    assert any(event.type == nb.AuditEventType.ORDER_WAITING_QUEUE for event in result.audit_events)


def test_kill_switch_mid_bar_blocks_later_fills() -> None:
    result = nb.run_backtest_matrix(
        timestamps=np.array([1], dtype=np.int64),
        close=np.array([[100.0, 100.0]], dtype=np.float64),
        high=np.array([[100.0, 100.0]], dtype=np.float64),
        low=np.array([[100.0, 100.0]], dtype=np.float64),
        bid=np.array([[100.0, 100.0]], dtype=np.float64),
        ask=np.array([[300.0, 100.0]], dtype=np.float64),
        volume=np.array([[1000.0, 1000.0]], dtype=np.float64),
        target_positions=np.array([[1, 1]], dtype=np.int64),
        config=nb.BacktestConfig(
            starting_cash=1000.0,
            max_position=1,
            max_drawdown_pct=0.05,
            use_bid_ask_execution=True,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )
    filled_assets = [int(fill.asset) for fill in result.fills]
    assert 0 in filled_assets
    assert 1 not in filled_assets
    assert any(event.type == nb.AuditEventType.RISK_KILL_SWITCH for event in result.audit_events)


def test_snapshot_roundtrip_at_bar_500_matches_full_run() -> None:
    rows = 600
    timestamps = np.arange(rows, dtype=np.int64) + 1
    close = np.full((rows, 1), 100.0, dtype=np.float64)
    volume = np.full((rows, 1), 10000.0, dtype=np.float64)
    targets = np.ones((rows, 1), dtype=np.int64)
    cfg = nb.BacktestConfig(max_position=1, slippage_bps=0.0, volume_share_impact=0.0)

    full = nb.run_backtest_matrix(
        timestamps=timestamps,
        close=close,
        high=close,
        low=close,
        volume=volume,
        target_positions=targets,
        config=cfg,
    )
    partial = nb.run_backtest_matrix(
        timestamps=timestamps,
        close=close,
        high=close,
        low=close,
        volume=volume,
        target_positions=targets,
        config=cfg,
        end_row=500,
    )
    resumed = nb.run_backtest_matrix(
        timestamps=timestamps,
        close=close,
        high=close,
        low=close,
        volume=volume,
        target_positions=targets,
        config=cfg,
        snapshot=partial.snapshot,
        start_row=500,
        end_row=rows,
    )
    assert full.pnl == pytest.approx(resumed.pnl, rel=0.0, abs=1e-9)


def test_final_bar_with_long_slice_delay_completes() -> None:
    result = nb.run_backtest_matrix(
        timestamps=np.array([1, 2], dtype=np.int64),
        close=np.array([[100.0], [100.0]], dtype=np.float64),
        high=np.array([[100.0], [100.0]], dtype=np.float64),
        low=np.array([[100.0], [100.0]], dtype=np.float64),
        volume=np.array([[1.0], [1.0]], dtype=np.float64),
        target_positions=np.array([[0], [5]], dtype=np.int64),
        config=nb.BacktestConfig(
            max_position=5,
            child_order_size=1,
            child_slice_delay_steps=10,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )
    assert result.submitted_orders >= 1


def test_nan_price_raises_clear_value_error() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        nb.run_backtest_matrix(
            timestamps=np.array([1], dtype=np.int64),
            close=np.array([[np.nan]], dtype=np.float64),
            high=np.array([[np.nan]], dtype=np.float64),
            low=np.array([[np.nan]], dtype=np.float64),
            volume=np.array([[1000.0]], dtype=np.float64),
            target_positions=np.array([[1]], dtype=np.int64),
            config=nb.BacktestConfig(max_position=1),
        )


def test_100_asset_simultaneous_signal_leverage_rejects_before_fills() -> None:
    cols = 100
    close = np.full((1, cols), 100.0, dtype=np.float64)
    result = nb.run_backtest_matrix(
        timestamps=np.array([1], dtype=np.int64),
        close=close,
        high=close,
        low=close,
        volume=np.full((1, cols), 1000.0, dtype=np.float64),
        target_positions=np.ones((1, cols), dtype=np.int64),
        config=nb.BacktestConfig(
            starting_cash=1000.0,
            max_position=1,
            max_gross_leverage=0.1,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )
    assert len(result.fills) == 0
    assert result.rejected_orders >= cols


def test_leverage_and_drawdown_constraints_do_not_duplicate_kill_events() -> None:
    result = nb.run_backtest_matrix(
        timestamps=np.array([1], dtype=np.int64),
        close=np.array([[100.0, 100.0]], dtype=np.float64),
        high=np.array([[100.0, 100.0]], dtype=np.float64),
        low=np.array([[100.0, 100.0]], dtype=np.float64),
        bid=np.array([[100.0, 100.0]], dtype=np.float64),
        ask=np.array([[300.0, 100.0]], dtype=np.float64),
        volume=np.array([[1000.0, 1000.0]], dtype=np.float64),
        target_positions=np.array([[10, 1]], dtype=np.int64),
        config=nb.BacktestConfig(
            starting_cash=1000.0,
            max_position=10,
            max_gross_leverage=0.2,
            max_drawdown_pct=0.05,
            use_bid_ask_execution=True,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )
    kill_events = [e for e in result.audit_events if e.type == nb.AuditEventType.RISK_KILL_SWITCH]
    assert len(kill_events) <= 1
