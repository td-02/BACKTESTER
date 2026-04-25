from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _enum_name(value: object) -> str:
    return str(getattr(value, "name", value))


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("pandas is required for TCA features; install nanoback[io]") from exc
    return pd


def _rolling_volatility(close: np.ndarray, window: int) -> np.ndarray:
    rows, cols = close.shape
    out = np.zeros((rows, cols), dtype=np.float64)
    if rows <= 1:
        return out
    returns = np.zeros((rows, cols), dtype=np.float64)
    prev = close[:-1]
    curr = close[1:]
    valid = prev != 0.0
    returns[1:][valid] = (curr[valid] - prev[valid]) / prev[valid]
    for col in range(cols):
        series = returns[:, col]
        csum = np.cumsum(series)
        csum_sq = np.cumsum(series * series)
        for row in range(rows):
            start = max(0, row - window + 1)
            count = row - start + 1
            sum_v = csum[row] - (csum[start - 1] if start > 0 else 0.0)
            sum_sq = csum_sq[row] - (csum_sq[start - 1] if start > 0 else 0.0)
            mean = sum_v / count
            variance = max(sum_sq / count - mean * mean, 0.0)
            out[row, col] = np.sqrt(variance)
    return out


def _row_indices(fill_timestamps: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(timestamps, fill_timestamps, side="left")
    if idx.size == 0:
        return idx
    idx = np.clip(idx, 0, max(0, timestamps.size - 1))
    return idx


def _arrival_timestamps(result: object) -> dict[int, int]:
    arrival_by_parent: dict[int, int] = {}
    for entry in getattr(result, "ledger", ()):
        if int(getattr(entry, "parent_order_id", 0)) == 0:
            continue
        if _enum_name(getattr(entry, "type", "")).upper() != "ORDER_SUBMITTED":
            continue
        parent = int(entry.parent_order_id)
        ts = int(entry.timestamp)
        if parent not in arrival_by_parent:
            arrival_by_parent[parent] = ts
    return arrival_by_parent


def tca_dataframe(
    result: object,
    *,
    symbols: Sequence[str] | None = None,
    volatility_window: int = 20,
    impact_coefficient: float = 1.0,
    time_bucket_seconds: int = 3600,
):
    pd = _require_pandas()
    fills = list(getattr(result, "fills", ()))
    if not fills:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "asset",
                "parent_order_id",
                "order_id",
                "direction",
                "quantity",
                "abs_quantity",
                "arrival_price",
                "fill_price",
                "mid_price",
                "bid",
                "ask",
                "spread",
                "spread_cost",
                "participation",
                "volatility",
                "impact_cost",
                "financing_cost",
                "implementation_shortfall",
                "at_or_better_arrival",
                "time_bucket",
                "total_cost",
            ]
        )

    timestamps = np.asarray(getattr(result, "timestamps"), dtype=np.int64)
    close = np.asarray(getattr(result, "close"), dtype=np.float64)
    bid = np.asarray(getattr(result, "bid", close), dtype=np.float64)
    ask = np.asarray(getattr(result, "ask", close), dtype=np.float64)
    volume = np.asarray(getattr(result, "volume", np.ones_like(close)), dtype=np.float64)
    if close.ndim == 1:
        close = close.reshape(-1, 1)
        bid = bid.reshape(-1, 1)
        ask = ask.reshape(-1, 1)
        volume = volume.reshape(-1, 1)

    fill_timestamps = np.asarray([int(fill.timestamp) for fill in fills], dtype=np.int64)
    assets = np.asarray([int(fill.asset) for fill in fills], dtype=np.int64)
    quantities = np.asarray([int(fill.quantity) for fill in fills], dtype=np.int64)
    abs_qty = np.abs(quantities).astype(np.float64)
    fill_prices = np.asarray([float(fill.price) for fill in fills], dtype=np.float64)
    parent_order_ids = np.asarray([int(fill.parent_order_id) for fill in fills], dtype=np.int64)
    order_ids = np.asarray([int(fill.order_id) for fill in fills], dtype=np.int64)

    row_idx = _row_indices(fill_timestamps, timestamps)
    clamped_assets = np.clip(assets, 0, close.shape[1] - 1)
    fill_bid = bid[row_idx, clamped_assets]
    fill_ask = ask[row_idx, clamped_assets]
    fill_mid = 0.5 * (fill_bid + fill_ask)
    spread = np.maximum(fill_ask - fill_bid, 0.0)
    spread_cost = 0.5 * spread * abs_qty

    vol_matrix = _rolling_volatility(close, max(2, volatility_window))
    volatility = vol_matrix[row_idx, clamped_assets]
    denom_volume = np.maximum(volume[row_idx, clamped_assets], 1.0)
    participation = np.clip(abs_qty / denom_volume, 0.0, 1.0)
    impact_cost = np.maximum(
        0.0,
        impact_coefficient * abs_qty * fill_prices * volatility * np.sqrt(participation),
    )

    arrival_lookup = _arrival_timestamps(result)
    arrival_ts = np.asarray([arrival_lookup.get(int(parent), int(ts)) for parent, ts in zip(parent_order_ids, fill_timestamps)], dtype=np.int64)
    arrival_idx = _row_indices(arrival_ts, timestamps)
    arrival_price = close[arrival_idx, clamped_assets]

    side = np.sign(quantities).astype(np.float64)
    implementation_shortfall = np.maximum(0.0, side * (fill_prices - arrival_price) * abs_qty)
    at_or_better_arrival = side * (fill_prices - arrival_price) <= 0.0

    notional = abs_qty * fill_prices
    total_notional = float(np.sum(notional))
    total_borrow = float(getattr(result, "total_borrow_cost", 0.0))
    financing_cost = np.zeros_like(notional)
    if total_notional > 0.0 and total_borrow != 0.0:
        financing_cost = total_borrow * (notional / total_notional)

    total_cost = spread_cost + impact_cost + financing_cost

    symbol_list = list(symbols or getattr(result, "symbols", []))
    labels = [symbol_list[asset] if 0 <= asset < len(symbol_list) else str(asset) for asset in assets]
    direction = np.where(quantities >= 0, "buy", "sell")
    day_seconds = np.mod(fill_timestamps, 86_400)
    bucket = (day_seconds // max(1, int(time_bucket_seconds))) * int(time_bucket_seconds)

    return pd.DataFrame(
        {
            "timestamp": fill_timestamps,
            "symbol": labels,
            "asset": assets,
            "parent_order_id": parent_order_ids,
            "order_id": order_ids,
            "direction": direction,
            "quantity": quantities,
            "abs_quantity": abs_qty,
            "arrival_price": arrival_price,
            "fill_price": fill_prices,
            "mid_price": fill_mid,
            "bid": fill_bid,
            "ask": fill_ask,
            "spread": spread,
            "spread_cost": spread_cost,
            "participation": participation,
            "volatility": volatility,
            "impact_cost": impact_cost,
            "financing_cost": financing_cost,
            "implementation_shortfall": implementation_shortfall,
            "at_or_better_arrival": at_or_better_arrival,
            "time_bucket": bucket.astype(np.int64),
            "total_cost": total_cost,
        }
    )


def aggregate_tca(df, by: Sequence[str] = ("symbol", "direction", "time_bucket")):
    if df.empty:
        return df
    return (
        df.groupby(list(by), as_index=False)
        .agg(
            fills=("order_id", "count"),
            abs_quantity=("abs_quantity", "sum"),
            spread_cost=("spread_cost", "sum"),
            impact_cost=("impact_cost", "sum"),
            financing_cost=("financing_cost", "sum"),
            implementation_shortfall=("implementation_shortfall", "sum"),
            total_cost=("total_cost", "sum"),
            fill_quality=("at_or_better_arrival", "mean"),
        )
    )


def fill_quality_score(df) -> float:
    if df.empty:
        return 0.0
    return float(df["at_or_better_arrival"].mean())


def export_tca_jsonl(df, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in df.to_dict(orient="records"):
            handle.write(json.dumps(row, default=float))
            handle.write("\n")
