from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ._nanoback import EngineSnapshot


@dataclass(slots=True)
class ReplayState:
    order_ids: list[int]
    parent_order_ids: list[int]
    fill_count: int
    final_cash: float
    final_equity: float


def _ledger_rows(result: object) -> list[dict[str, object]]:
    return [
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
            "cash_after": float(entry.cash_after),
            "equity_after": float(entry.equity_after),
            "value": float(entry.value),
        }
        for entry in result.ledger
    ]


def export_ledger_csv(result: object, path: str | Path) -> None:
    rows = _ledger_rows(result)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()) if rows else [
                "sequence", "timestamp", "order_id", "parent_order_id", "asset", "type",
                "quantity", "remaining_quantity", "price", "cash_after", "equity_after", "value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def export_ledger_jsonl(result: object, path: str | Path) -> None:
    rows = _ledger_rows(result)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def load_ledger_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def replay_ledger(rows: Iterable[dict[str, object]]) -> ReplayState:
    order_ids: list[int] = []
    parent_order_ids: list[int] = []
    fill_count = 0
    final_cash = 0.0
    final_equity = 0.0

    for row in rows:
        order_id = int(row["order_id"])
        parent_order_id = int(row["parent_order_id"])
        if order_id not in order_ids and order_id != 0:
            order_ids.append(order_id)
        if parent_order_id not in parent_order_ids and parent_order_id != 0:
            parent_order_ids.append(parent_order_id)
        if "FILL_APPLIED" in str(row["type"]).upper():
            fill_count += 1
        final_cash = float(row["cash_after"])
        final_equity = float(row["equity_after"])

    return ReplayState(order_ids, parent_order_ids, fill_count, final_cash, final_equity)


def snapshot_to_dict(snapshot: EngineSnapshot) -> dict[str, object]:
    return {
        "next_row": int(snapshot.next_row),
        "cash": float(snapshot.cash),
        "peak_equity": float(snapshot.peak_equity),
        "total_fees": float(snapshot.total_fees),
        "total_borrow_cost": float(snapshot.total_borrow_cost),
        "total_cash_yield": float(snapshot.total_cash_yield),
        "turnover": float(snapshot.turnover),
        "submitted_orders": int(snapshot.submitted_orders),
        "filled_orders": int(snapshot.filled_orders),
        "rejected_orders": int(snapshot.rejected_orders),
        "next_parent_order_id": int(snapshot.next_parent_order_id),
        "next_child_order_id": int(snapshot.next_child_order_id),
        "next_ledger_sequence": int(snapshot.next_ledger_sequence),
        "halted_by_risk": bool(snapshot.halted_by_risk),
        "positions": list(snapshot.positions),
        "pending_parent_order_ids": list(snapshot.pending_parent_order_ids),
        "pending_target_positions": list(snapshot.pending_target_positions),
        "pending_remaining_quantities": list(snapshot.pending_remaining_quantities),
        "pending_limit_prices": list(snapshot.pending_limit_prices),
        "pending_order_types": list(snapshot.pending_order_types),
        "pending_ready_indices": list(snapshot.pending_ready_indices),
        "pending_active": list(snapshot.pending_active),
    }


def snapshot_from_dict(payload: dict[str, object]) -> EngineSnapshot:
    snapshot = EngineSnapshot()
    for key, value in payload.items():
        setattr(snapshot, key, value)
    return snapshot


def save_snapshot(snapshot: EngineSnapshot, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(snapshot_to_dict(snapshot), handle)


def load_snapshot(path: str | Path) -> EngineSnapshot:
    with Path(path).open("r", encoding="utf-8") as handle:
        return snapshot_from_dict(json.load(handle))
