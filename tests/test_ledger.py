from __future__ import annotations

from pathlib import Path
import json

import numpy as np

import nanoback as nb


def test_parent_child_order_ids_are_emitted() -> None:
    result = nb.run_backtest(
        timestamps=np.array([1, 2, 3], dtype=np.int64),
        prices=np.array([100.0, 100.0, 100.0], dtype=np.float64),
        volume=np.array([10.0, 10.0, 10.0], dtype=np.float64),
        signals=np.array([5, 5, 5], dtype=np.int64),
        config=nb.BacktestConfig(
            max_position=5,
            child_order_size=2,
            max_participation_rate=1.0,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )

    assert len(result.fills) >= 2
    parent_ids = {fill.parent_order_id for fill in result.fills}
    child_ids = [fill.order_id for fill in result.fills]
    assert len(parent_ids) == 1
    assert len(set(child_ids)) == len(child_ids)
    assert all(fill.parent_order_id != 0 for fill in result.fills)


def test_ledger_export_and_replay(tmp_path: Path) -> None:
    result = nb.run_backtest(
        timestamps=np.array([1, 2], dtype=np.int64),
        prices=np.array([100.0, 101.0], dtype=np.float64),
        signals=np.array([1, 0], dtype=np.int64),
        config=nb.BacktestConfig(
            max_position=1,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )

    target = tmp_path / "ledger.csv"
    nb.export_ledger_csv(result, target)
    rows = nb.load_ledger_csv(target)
    replay = nb.replay_ledger(rows)

    assert target.exists()
    assert replay.fill_count >= 1
    assert replay.final_cash == float(rows[-1]["cash_after"])


def test_snapshot_resume_matches_single_pass(tmp_path: Path) -> None:
    timestamps = np.array([1, 2, 3, 4], dtype=np.int64)
    prices = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float64)
    signals = np.array([1, 1, 0, -1], dtype=np.int64)
    config = nb.BacktestConfig(max_position=2, child_order_size=1, child_slice_delay_steps=1, slippage_bps=0.0, volume_share_impact=0.0)

    full = nb.run_backtest(timestamps=timestamps, prices=prices, signals=signals, config=config)
    first = nb.run_backtest(timestamps=timestamps, prices=prices, signals=signals, config=config, end_row=2)
    snapshot_path = tmp_path / "snapshot.json"
    nb.save_snapshot(first.snapshot, snapshot_path)
    resumed = nb.run_backtest(
        timestamps=timestamps,
        prices=prices,
        signals=signals,
        config=config,
        snapshot=nb.load_snapshot(snapshot_path),
        start_row=2,
    )

    assert full.ending_cash == resumed.ending_cash
    assert full.ending_equity == resumed.ending_equity
    assert first.snapshot.next_row == 2


def test_deterministic_ledger_fixture_matches() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "expected_simple_ledger.json"
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))
    result = nb.run_backtest(
        timestamps=np.array([1, 2], dtype=np.int64),
        prices=np.array([100.0, 101.0], dtype=np.float64),
        signals=np.array([1, 0], dtype=np.int64),
        config=nb.BacktestConfig(max_position=1, slippage_bps=0.0, volume_share_impact=0.0),
    )

    actual = [
        {
            "sequence": int(entry.sequence),
            "timestamp": int(entry.timestamp),
            "order_id": int(entry.order_id),
            "parent_order_id": int(entry.parent_order_id),
            "asset": int(entry.asset),
            "type": str(entry.type),
            "quantity": int(entry.quantity),
            "remaining_quantity": int(entry.remaining_quantity),
            "price": float(entry.price),
        }
        for entry in result.ledger
    ]

    assert actual == expected
