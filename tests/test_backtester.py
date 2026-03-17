from __future__ import annotations

import numpy as np

import nanoback as nb


def test_single_asset_wrapper_smoke() -> None:
    result = nb.run_backtest(
        timestamps=np.array([1, 2, 3, 4], dtype=np.int64),
        prices=np.array([100.0, 101.0, 103.0, 102.0], dtype=np.float64),
        signals=np.array([1, 1, -1, 0], dtype=np.int64),
        config=nb.BacktestConfig(
            starting_cash=10_000.0,
            max_position=2,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )

    assert result.positions.shape == (4, 1)
    assert result.ending_equity > 0.0
    assert len(result.fills) >= 1


def test_multi_asset_limit_latency_and_partial_fills() -> None:
    timestamps = np.array([34_200, 34_260, 34_320, 34_380], dtype=np.int64)
    close = np.array([[10.0, 20.0], [9.7, 20.5], [9.6, 20.2], [10.2, 19.8]], dtype=np.float64)
    high = np.array([[10.2, 20.2], [10.0, 20.8], [9.8, 20.5], [10.3, 20.0]], dtype=np.float64)
    low = np.array([[9.9, 19.8], [9.6, 20.1], [9.4, 20.0], [10.0, 19.6]], dtype=np.float64)
    volume = np.array([[3.0, 5.0], [2.0, 5.0], [2.0, 5.0], [5.0, 5.0]], dtype=np.float64)
    targets = np.array([[4, 0], [4, -2], [0, -2], [0, 0]], dtype=np.int64)
    order_types = np.array(
        [
            [int(nb.OrderType.LIMIT), int(nb.OrderType.MARKET)],
            [int(nb.OrderType.LIMIT), int(nb.OrderType.MARKET)],
            [int(nb.OrderType.MARKET), int(nb.OrderType.MARKET)],
            [int(nb.OrderType.MARKET), int(nb.OrderType.MARKET)],
        ],
        dtype=np.int8,
    )
    limit_prices = np.array([[9.8, np.nan], [9.7, np.nan], [np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64)

    result = nb.run_backtest_matrix(
        timestamps=timestamps,
        close=close,
        high=high,
        low=low,
        volume=volume,
        target_positions=targets,
        order_types=order_types,
        limit_prices=limit_prices,
        config=nb.BacktestConfig(
            max_position=4,
            latency_steps=1,
            max_participation_rate=0.5,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
        symbols=["AAA", "BBB"],
    )

    assert result.positions.shape == (4, 2)
    assert len(result.fills) >= 3
    assert any(fill.remaining_quantity != 0 for fill in result.fills)
    assert result.submitted_orders >= 2


def test_leverage_limit_rejects_orders_and_audits() -> None:
    result = nb.run_backtest_matrix(
        timestamps=np.array([1, 2], dtype=np.int64),
        close=np.array([[100.0], [100.0]], dtype=np.float64),
        high=np.array([[100.0], [100.0]], dtype=np.float64),
        low=np.array([[100.0], [100.0]], dtype=np.float64),
        volume=np.array([[1_000.0], [1_000.0]], dtype=np.float64),
        target_positions=np.array([[5], [5]], dtype=np.int64),
        config=nb.BacktestConfig(
            starting_cash=100.0,
            max_position=5,
            max_gross_leverage=1.0,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )

    assert result.rejected_orders >= 1
    assert len(result.fills) == 0
    assert any(event.type == nb.AuditEventType.ORDER_REJECTED_LEVERAGE for event in result.audit_events)


def test_drawdown_kill_switch_halts_execution() -> None:
    result = nb.run_backtest(
        timestamps=np.array([1, 2, 3, 4], dtype=np.int64),
        prices=np.array([100.0, 50.0, 40.0, 30.0], dtype=np.float64),
        signals=np.array([1, 1, 1, 1], dtype=np.int64),
        config=nb.BacktestConfig(
            starting_cash=1_000.0,
            max_position=5,
            max_drawdown_pct=0.04,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )

    assert result.halted_by_risk is True
    assert result.max_drawdown > 0.04
    assert any(event.type == nb.AuditEventType.RISK_KILL_SWITCH for event in result.audit_events)


def test_borrow_and_cash_carry_are_accounted_for() -> None:
    result = nb.run_backtest(
        timestamps=np.array([1, 2, 3], dtype=np.int64),
        prices=np.array([100.0, 99.0, 98.0], dtype=np.float64),
        signals=np.array([-1, -1, -1], dtype=np.int64),
        config=nb.BacktestConfig(
            starting_cash=10_000.0,
            max_position=2,
            annual_borrow_bps=252.0,
            annual_cash_yield_bps=252.0,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )

    assert result.total_borrow_cost > 0.0
    assert result.total_cash_yield > 0.0


def test_bid_ask_execution_prices_market_buy_at_ask() -> None:
    result = nb.run_backtest(
        timestamps=np.array([1], dtype=np.int64),
        prices=np.array([100.0], dtype=np.float64),
        bid=np.array([99.5], dtype=np.float64),
        ask=np.array([100.5], dtype=np.float64),
        signals=np.array([1], dtype=np.int64),
        config=nb.BacktestConfig(
            max_position=1,
            use_bid_ask_execution=True,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )

    assert len(result.fills) == 1
    assert result.fills[0].price == 100.5


def test_queue_blocking_prevents_fill_and_audits() -> None:
    result = nb.run_backtest_matrix(
        timestamps=np.array([1, 2], dtype=np.int64),
        close=np.array([[100.0], [100.0]], dtype=np.float64),
        high=np.array([[100.0], [100.0]], dtype=np.float64),
        low=np.array([[100.0], [100.0]], dtype=np.float64),
        volume=np.array([[10.0], [10.0]], dtype=np.float64),
        bid=np.array([[99.9], [99.9]], dtype=np.float64),
        ask=np.array([[100.1], [100.1]], dtype=np.float64),
        target_positions=np.array([[5], [5]], dtype=np.int64),
        config=nb.BacktestConfig(
            max_position=5,
            queue_ahead_fraction=0.95,
            venue_volume_share_cap=0.5,
            max_participation_rate=0.5,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )

    assert len(result.fills) == 0
    assert any(event.type == nb.AuditEventType.ORDER_WAITING_QUEUE for event in result.audit_events)


def test_asset_level_notional_limit_rejects() -> None:
    result = nb.run_backtest_matrix(
        timestamps=np.array([1, 2], dtype=np.int64),
        close=np.array([[100.0, 50.0], [100.0, 50.0]], dtype=np.float64),
        high=np.array([[100.0, 50.0], [100.0, 50.0]], dtype=np.float64),
        low=np.array([[100.0, 50.0], [100.0, 50.0]], dtype=np.float64),
        volume=np.array([[1_000.0, 1_000.0], [1_000.0, 1_000.0]], dtype=np.float64),
        target_positions=np.array([[3, 1], [3, 1]], dtype=np.int64),
        asset_max_positions=np.array([5, 5], dtype=np.int64),
        asset_notional_limits=np.array([200.0, 1_000.0], dtype=np.float64),
        config=nb.BacktestConfig(max_position=5, slippage_bps=0.0, volume_share_impact=0.0),
    )

    assert result.rejected_orders >= 1
    assert any(event.type == nb.AuditEventType.ORDER_REJECTED_LEVERAGE for event in result.audit_events)
