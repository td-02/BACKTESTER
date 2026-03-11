from __future__ import annotations

from pathlib import Path

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
