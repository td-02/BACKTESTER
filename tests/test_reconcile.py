from __future__ import annotations

import json

from nanoback.reconcile import Reconciler


class MockAdapter:
    def __init__(self, positions):
        self.positions = dict(positions)
        self.orders = []

    def fetch_positions(self):
        return dict(self.positions)

    def submit_order(self, symbol, quantity_delta):
        self.orders.append((symbol, quantity_delta))


def test_reconciler_detects_mismatch_and_logs(tmp_path) -> None:
    adapter = MockAdapter({"AAA": 3.0})
    log_path = tmp_path / "reconcile.jsonl"
    reconciler = Reconciler(adapter=adapter, log_path=log_path, auto_reconcile=False)

    records = reconciler.reconcile({"AAA": 1.0}, timestamp_ns=123)

    assert len(records) == 1
    assert records[0].delta == 2.0
    assert records[0].action == "mismatch_detected"

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["timestamp_ns"] == 123
    assert rows[0]["delta"] == 2.0


def test_reconciler_auto_reconcile_submits_orders(tmp_path) -> None:
    adapter = MockAdapter({"AAA": 5.0, "BBB": -2.0})
    reconciler = Reconciler(adapter=adapter, log_path=tmp_path / "reconcile.jsonl", auto_reconcile=True)

    records = reconciler.reconcile({"AAA": 2.0, "BBB": -2.0}, timestamp_ns=999)

    assert len(records) == 2
    assert ("AAA", 3.0) in adapter.orders
    assert all(record.action in {"submitted_correction", "none"} for record in records)
