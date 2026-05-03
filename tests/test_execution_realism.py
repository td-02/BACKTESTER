from __future__ import annotations

import numpy as np

import nanoback as nb


def test_multi_venue_splits_generate_venue_fills() -> None:
    venue_a = nb.Venue()
    venue_a.venue_id = 1
    venue_a.volume_share = 0.7
    venue_a.taker_fee_bps = 0.2
    venue_b = nb.Venue()
    venue_b.venue_id = 2
    venue_b.volume_share = 0.3
    venue_b.taker_fee_bps = 0.3

    cfg = nb.BacktestConfig(max_position=5, slippage_bps=0.0, volume_share_impact=0.0)
    cfg.venues = [venue_a, venue_b]

    result = nb.run_backtest(
        timestamps=np.array([1, 2, 3], dtype=np.int64),
        prices=np.array([100.0, 101.0, 102.0], dtype=np.float64),
        signals=np.array([5, 5, 0], dtype=np.int64),
        config=cfg,
    )
    venue_ids = {int(fill.venue_id) for fill in result.fills}
    assert venue_ids.issubset({1, 2})
    assert len(result.fills) >= 1


def test_option_expiry_cash_settlement() -> None:
    instr = nb.Instrument()
    instr.type = nb.InstrumentType.OPTION_CALL
    instr.expiry_timestamp = 3
    instr.strike = 100.0
    instr.underlying_asset = 0

    cfg = nb.BacktestConfig(max_position=1, slippage_bps=0.0, volume_share_impact=0.0)
    cfg.instruments = [instr]

    result = nb.run_backtest(
        timestamps=np.array([1, 2, 3], dtype=np.int64),
        prices=np.array([1.0, 2.0, 110.0], dtype=np.float64),
        signals=np.array([1, 1, 1], dtype=np.int64),
        tradable_mask=np.array([1, 1, 0], dtype=np.uint8),
        config=cfg,
    )
    assert int(result.positions[2, 0]) == 0
    assert any("OPTION_EXPIRY" in str(evt.type) for evt in result.audit_events)


def test_latency_penalty_changes_fill_price() -> None:
    cfg_no = nb.BacktestConfig(max_position=1, slippage_bps=0.0, volume_share_impact=0.0)
    cfg_yes = nb.BacktestConfig(max_position=1, slippage_bps=0.0, volume_share_impact=0.0)
    cfg_yes.signal_to_order_latency_us = 1000
    cfg_yes.order_to_fill_latency_us = 1000
    cfg_yes.latency_drift_model = nb.LatencyDriftModel.GBM
    cfg_yes.adverse_velocity_threshold = 0.0001
    cfg_yes.adverse_selection_penalty_bps = 5.0

    kwargs = dict(
        timestamps=np.array([1, 2], dtype=np.int64),
        prices=np.array([100.0, 100.2], dtype=np.float64),
        high=np.array([100.5, 100.7], dtype=np.float64),
        low=np.array([99.5, 99.7], dtype=np.float64),
        signals=np.array([1, 1], dtype=np.int64),
        config=None,
    )
    r0 = nb.run_backtest(**{**kwargs, "config": cfg_no})
    r1 = nb.run_backtest(**{**kwargs, "config": cfg_yes})
    if r0.fills and r1.fills:
        assert float(r0.fills[0].price) != float(r1.fills[0].price)
