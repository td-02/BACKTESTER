from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


class BrokerPositionAdapter(Protocol):
    def fetch_positions(self) -> dict[str, float]:
        ...

    def submit_order(self, symbol: str, quantity_delta: float) -> None:
        ...


@dataclass(slots=True)
class ReconciliationRecord:
    timestamp: str
    symbol: str
    engine_position: float
    broker_position: float
    delta: float
    action: str


class Reconciler:
    def __init__(
        self,
        *,
        adapter: BrokerPositionAdapter,
        log_path: str | Path,
        auto_reconcile: bool = False,
        tolerance: float = 0.0,
    ) -> None:
        self.adapter = adapter
        self.log_path = Path(log_path)
        self.auto_reconcile = auto_reconcile
        self.tolerance = abs(float(tolerance))

    def reconcile(self, engine_positions: dict[str, float], timestamp_ns: int | None = None) -> list[ReconciliationRecord]:
        broker_positions = self.adapter.fetch_positions()
        symbols = sorted(set(engine_positions) | set(broker_positions))
        records: list[ReconciliationRecord] = []

        now = datetime.now(timezone.utc).isoformat()
        ts = timestamp_ns if timestamp_ns is not None else 0
        for symbol in symbols:
            engine = float(engine_positions.get(symbol, 0.0))
            broker = float(broker_positions.get(symbol, 0.0))
            delta = broker - engine
            action = "none"
            if abs(delta) > self.tolerance and self.auto_reconcile:
                self.adapter.submit_order(symbol, delta)
                action = "submitted_correction"
            elif abs(delta) > self.tolerance:
                action = "mismatch_detected"

            record = ReconciliationRecord(
                timestamp=now,
                symbol=symbol,
                engine_position=engine,
                broker_position=broker,
                delta=delta,
                action=action,
            )
            records.append(record)

        self._append_log(records, ts)
        return records

    def _append_log(self, records: list[ReconciliationRecord], timestamp_ns: int) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {
                            "timestamp": record.timestamp,
                            "timestamp_ns": int(timestamp_ns),
                            "symbol": record.symbol,
                            "engine_position": record.engine_position,
                            "broker_position": record.broker_position,
                            "delta": record.delta,
                            "action": record.action,
                        }
                    )
                    + "\n"
                )
