from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class ReplayState:
    order_ids: list[int]
    parent_order_ids: list[int]
    fill_count: int
    final_cash: float
    final_equity: float


def _ledger_rows(result: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in result.ledger:
        rows.append(
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
        )
    return rows


def export_ledger_csv(result: object, path: str | Path) -> None:
    rows = _ledger_rows(result)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "sequence", "timestamp", "order_id", "parent_order_id", "asset", "type",
            "quantity", "remaining_quantity", "price", "cash_after", "equity_after", "value"
        ])
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

    return ReplayState(
        order_ids=order_ids,
        parent_order_ids=parent_order_ids,
        fill_count=fill_count,
        final_cash=final_cash,
        final_equity=final_equity,
    )
