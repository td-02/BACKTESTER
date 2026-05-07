from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

import nanoback as nb
from nanoback.paper import PaperBroker, PaperTick


class MockFeed:
    def __init__(self, ticks):
        self._ticks = list(ticks)
        self.orders = []

    def next_tick(self, timeout_seconds=None):
        if not self._ticks:
            return None
        return self._ticks.pop(0)

    def fetch_positions(self):
        return {"AAA": 0.0}

    def submit_order(self, symbol, quantity_delta):
        self.orders.append((symbol, quantity_delta))


def test_paper_broker_runs_and_records_fills(tmp_path) -> None:
    ticks = [
        PaperTick(timestamp_ns=1, symbol="AAA", price=100.0, size=100.0, side="trade"),
        PaperTick(timestamp_ns=2, symbol="AAA", price=101.0, size=100.0, side="trade"),
    ]
    feed = MockFeed(ticks)

    def strategy(tick, state):
        return np.array([1], dtype=np.int64)

    ledger_path = tmp_path / "paper_ledger.jsonl"
    broker = PaperBroker(
        symbols=["AAA"],
        strategy=strategy,
        config=nb.BacktestConfig(max_position=5, slippage_bps=0.0, volume_share_impact=0.0),
        feed=feed,
        ledger_path=ledger_path,
    )
    broker.run_until(datetime.now(timezone.utc) + timedelta(milliseconds=100))

    assert len(broker.results) >= 1
    assert len(broker.fills) >= 1
    assert ledger_path.exists()
    assert len(ledger_path.read_text(encoding="utf-8").strip().splitlines()) >= 1


def test_paper_broker_reconcile_callback_on_fill() -> None:
    ticks = [PaperTick(timestamp_ns=1, symbol="AAA", price=100.0, size=100.0, side="trade")]
    feed = MockFeed(ticks)
    calls = []

    def strategy(tick, state):
        return np.array([1], dtype=np.int64)

    def reconcile(positions, ts):
        calls.append((positions, ts))

    broker = PaperBroker(
        symbols=["AAA"],
        strategy=strategy,
        config=nb.BacktestConfig(max_position=5, slippage_bps=0.0, volume_share_impact=0.0),
        feed=feed,
        reconcile_callback=reconcile,
    )
    broker.run_until(datetime.now(timezone.utc) + timedelta(milliseconds=50))

    assert len(calls) == 1
    assert calls[0][1] == 1
