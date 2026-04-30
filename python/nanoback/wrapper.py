from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ._nanoback import (
    BacktestConfig,
    EngineSnapshot,
    OrderType,
    run_backtest_matrix as _run_backtest_matrix,
    run_backtest_ticks as _run_backtest_ticks,
)
from .analytics import BacktestSummary, summarize_result


@dataclass(slots=True)
class PythonBacktestResult:
    raw: object
    symbols: list[str]
    timestamps: np.ndarray
    close: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    volume: np.ndarray
    adjustment_factors: np.ndarray
    positions: np.ndarray
    equity_curve: np.ndarray
    cash_curve: np.ndarray

    def summary(self) -> BacktestSummary:
        return summarize_result(self, symbols=self.symbols)

    def to_dict(self) -> dict[str, object]:
        summary = self.summary()
        return {
            "symbols": list(self.symbols),
            "timestamps": self.timestamps.tolist(),
            "equity_curve": self.equity_curve.tolist(),
            "cash_curve": self.cash_curve.tolist(),
            "positions": self.positions.tolist(),
            "metrics": summary.to_dict(),
            "fills": [
                {
                    "timestamp": int(fill.timestamp),
                    "order_id": int(fill.order_id),
                    "parent_order_id": int(fill.parent_order_id),
                    "asset": int(fill.asset),
                    "price": float(fill.price),
                    "quantity": int(fill.quantity),
                    "remaining_quantity": int(fill.remaining_quantity),
                    "fee": float(fill.fee),
                    "order_type": str(fill.order_type),
                }
                for fill in self.fills
            ],
            "adjustment_factors": self.adjustment_factors.tolist(),
        }

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("pandas is required for to_dataframe(); install nanoback[io]") from exc

        df = pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "equity": self.equity_curve,
                "cash": self.cash_curve,
            }
        )
        if self.equity_curve.size > 0:
            start_equity = float(self.equity_curve[0])
            pnl = self.equity_curve - start_equity
            running_peak = np.maximum.accumulate(self.equity_curve)
            safe_peak = np.where(running_peak == 0.0, 1.0, running_peak)
            drawdown = (running_peak - self.equity_curve) / safe_peak
            drawdown[running_peak == 0.0] = 0.0
            df["pnl"] = pnl
            df["drawdown"] = drawdown
        else:
            df["pnl"] = np.asarray([], dtype=np.float64)
            df["drawdown"] = np.asarray([], dtype=np.float64)
        return df

    def __repr__(self) -> str:
        metrics = self.summary()
        rows = [
            ("Sharpe", metrics.sharpe),
            ("Max Drawdown", metrics.max_drawdown),
            ("CAGR", metrics.cagr),
            ("Turnover/Yr", metrics.turnover_per_year),
            ("Fill Count", float(metrics.fill_count)),
        ]
        lines = ["BacktestResult Summary", "Metric        Value", "------------  -------------"]
        for label, value in rows:
            if np.isinf(value):
                text = "inf" if value > 0 else "-inf"
            else:
                text = f"{value:.6f}"
            lines.append(f"{label:<12}  {text:>13}")
        return "\n".join(lines)

    @property
    def ending_cash(self) -> float:
        return float(self.raw.ending_cash)

    @property
    def ending_equity(self) -> float:
        return float(self.raw.ending_equity)

    @property
    def pnl(self) -> float:
        return float(self.raw.pnl)

    @property
    def turnover(self) -> float:
        return float(self.raw.turnover)

    @property
    def total_fees(self) -> float:
        return float(self.raw.total_fees)

    @property
    def total_borrow_cost(self) -> float:
        return float(self.raw.total_borrow_cost)

    @property
    def total_cash_yield(self) -> float:
        return float(self.raw.total_cash_yield)

    @property
    def peak_equity(self) -> float:
        return float(self.raw.peak_equity)

    @property
    def max_drawdown(self) -> float:
        return float(self.raw.max_drawdown)

    @property
    def halted_by_risk(self) -> bool:
        return bool(self.raw.halted_by_risk)

    @property
    def fills(self):
        return self.raw.fills

    @property
    def audit_events(self):
        return self.raw.audit_events

    @property
    def ledger(self):
        return self.raw.ledger

    @property
    def snapshot(self):
        return self.raw.snapshot

    @property
    def submitted_orders(self) -> int:
        return int(self.raw.submitted_orders)

    @property
    def filled_orders(self) -> int:
        return int(self.raw.filled_orders)

    @property
    def rejected_orders(self) -> int:
        return int(self.raw.rejected_orders)


def _as_matrix(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        return np.ascontiguousarray(array.reshape(-1, 1))
    if array.ndim != 2:
        raise ValueError("expected a 1D or 2D array")
    return np.ascontiguousarray(array)


def _as_vector(values: np.ndarray | Sequence[float], dtype: np.dtype, length: int, fill_value: float | int) -> np.ndarray:
    if values is None:
        return np.full(length, fill_value, dtype=dtype)
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError("expected a 1D vector with matching length")
    return np.ascontiguousarray(array)


def run_backtest_matrix(
    *,
    timestamps: np.ndarray,
    close: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    bid: np.ndarray | None = None,
    ask: np.ndarray | None = None,
    target_positions: np.ndarray,
    order_types: np.ndarray | None = None,
    limit_prices: np.ndarray | None = None,
    tradable_mask: np.ndarray | None = None,
    asset_max_positions: np.ndarray | None = None,
    asset_notional_limits: np.ndarray | None = None,
    config: BacktestConfig | None = None,
    symbols: Sequence[str] | None = None,
    snapshot: EngineSnapshot | None = None,
    start_row: int = 0,
    end_row: int | None = None,
) -> PythonBacktestResult:
    timestamps = np.asarray(timestamps, dtype=np.int64)
    close = _as_matrix(close, np.float64)
    rows, cols = close.shape
    high = close if high is None else _as_matrix(high, np.float64)
    low = close if low is None else _as_matrix(low, np.float64)
    volume = np.full((rows, cols), 1_000_000.0, dtype=np.float64) if volume is None else _as_matrix(volume, np.float64)
    bid = close if bid is None else _as_matrix(bid, np.float64)
    ask = close if ask is None else _as_matrix(ask, np.float64)
    target_positions = _as_matrix(target_positions, np.int64)
    order_types = (
        np.full((rows, cols), int(OrderType.MARKET), dtype=np.int8)
        if order_types is None else _as_matrix(order_types, np.int8)
    )
    limit_prices = (
        np.full((rows, cols), np.nan, dtype=np.float64)
        if limit_prices is None else _as_matrix(limit_prices, np.float64)
    )
    tradable_mask = np.ones(rows, dtype=np.uint8) if tradable_mask is None else np.asarray(tradable_mask, dtype=np.uint8)
    asset_max_positions = _as_vector(asset_max_positions, np.int64, cols, 0)
    asset_notional_limits = _as_vector(asset_notional_limits, np.float64, cols, 0.0)
    config = config or BacktestConfig()
    symbols = list(symbols or [f"asset_{idx}" for idx in range(cols)])
    effective_end_row = rows if end_row is None else end_row

    raw = _run_backtest_matrix(
        timestamps=timestamps,
        close=close,
        high=high,
        low=low,
        volume=volume,
        bid=bid,
        ask=ask,
        target_positions=target_positions,
        order_types=order_types,
        limit_prices=limit_prices,
        tradable_mask=tradable_mask,
        asset_max_positions=asset_max_positions,
        asset_notional_limits=asset_notional_limits,
        config=config,
        snapshot=snapshot,
        start_row=start_row,
        end_row=effective_end_row,
    )
    return PythonBacktestResult(
        raw=raw,
        symbols=symbols,
        timestamps=timestamps.copy(),
        close=close.copy(),
        bid=bid.copy(),
        ask=ask.copy(),
        volume=volume.copy(),
        positions=np.asarray(raw.positions, dtype=np.int64).reshape(rows, cols),
        equity_curve=np.asarray(raw.equity_curve, dtype=np.float64),
        cash_curve=np.asarray(raw.cash_curve, dtype=np.float64),
        adjustment_factors=np.asarray(raw.adjustment_factors, dtype=np.float64).reshape(rows, cols),
    )


def run_backtest_ticks(
    *,
    timestamp_ns: np.ndarray,
    asset: np.ndarray,
    price: np.ndarray,
    size: np.ndarray,
    side: np.ndarray,
    target_positions: np.ndarray,
    cols: int,
    config: BacktestConfig | None = None,
    symbols: Sequence[str] | None = None,
) -> PythonBacktestResult:
    timestamp_ns = np.asarray(timestamp_ns, dtype=np.int64)
    asset = np.asarray(asset, dtype=np.int64)
    price = np.asarray(price, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    side = np.asarray(side, dtype=np.int8)
    target_positions = _as_matrix(target_positions, np.int64)
    rows = timestamp_ns.shape[0]
    if target_positions.shape != (rows, cols):
        raise ValueError("target_positions must have shape (len(timestamp_ns), cols)")
    config = config or BacktestConfig()
    symbols = list(symbols or [f"asset_{idx}" for idx in range(cols)])
    raw = _run_backtest_ticks(
        timestamp_ns=timestamp_ns,
        asset=asset,
        price=price,
        size=size,
        side=side,
        target_positions=target_positions.reshape(-1),
        cols=cols,
        config=config,
    )
    # Reconstruct minimal matrices for downstream analytics.
    close = np.tile(price.reshape(-1, 1), (1, cols))
    bid = close.copy()
    ask = close.copy()
    volume = np.tile(np.maximum(size, 1.0).reshape(-1, 1), (1, cols))
    return PythonBacktestResult(
        raw=raw,
        symbols=symbols,
        timestamps=timestamp_ns.copy(),
        close=close,
        bid=bid,
        ask=ask,
        volume=volume,
        positions=np.asarray(raw.positions, dtype=np.int64).reshape(rows, cols),
        equity_curve=np.asarray(raw.equity_curve, dtype=np.float64),
        cash_curve=np.asarray(raw.cash_curve, dtype=np.float64),
        adjustment_factors=np.asarray(raw.adjustment_factors, dtype=np.float64).reshape(rows, cols),
    )


def run_backtest(
    *,
    timestamps: np.ndarray,
    prices: np.ndarray,
    signals: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    bid: np.ndarray | None = None,
    ask: np.ndarray | None = None,
    order_type: OrderType = OrderType.MARKET,
    limit_prices: np.ndarray | None = None,
    tradable_mask: np.ndarray | None = None,
    asset_max_positions: np.ndarray | None = None,
    asset_notional_limits: np.ndarray | None = None,
    config: BacktestConfig | None = None,
    snapshot: EngineSnapshot | None = None,
    start_row: int = 0,
    end_row: int | None = None,
) -> PythonBacktestResult:
    prices = np.asarray(prices, dtype=np.float64)
    signals = np.asarray(signals, dtype=np.int64)
    targets = signals.reshape(-1, 1)
    order_types = np.full_like(targets, int(order_type), dtype=np.int8)
    return run_backtest_matrix(
        timestamps=np.asarray(timestamps, dtype=np.int64),
        close=prices.reshape(-1, 1),
        high=None if high is None else np.asarray(high, dtype=np.float64).reshape(-1, 1),
        low=None if low is None else np.asarray(low, dtype=np.float64).reshape(-1, 1),
        volume=None if volume is None else np.asarray(volume, dtype=np.float64).reshape(-1, 1),
        bid=None if bid is None else np.asarray(bid, dtype=np.float64).reshape(-1, 1),
        ask=None if ask is None else np.asarray(ask, dtype=np.float64).reshape(-1, 1),
        target_positions=targets,
        order_types=order_types,
        limit_prices=None if limit_prices is None else np.asarray(limit_prices, dtype=np.float64).reshape(-1, 1),
        tradable_mask=tradable_mask,
        asset_max_positions=asset_max_positions,
        asset_notional_limits=asset_notional_limits,
        config=config,
        symbols=["asset_0"],
        snapshot=snapshot,
        start_row=start_row,
        end_row=end_row,
    )
