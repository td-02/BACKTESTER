from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ._nanoback import (
    BacktestConfig,
    Backtester,
    OrderType,
    run_backtest_matrix as _run_backtest_matrix,
)


@dataclass(slots=True)
class PythonBacktestResult:
    raw: object
    symbols: list[str]
    positions: np.ndarray
    equity_curve: np.ndarray
    cash_curve: np.ndarray

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
    config: BacktestConfig | None = None,
    symbols: Sequence[str] | None = None,
) -> PythonBacktestResult:
    timestamps = np.asarray(timestamps, dtype=np.int64)
    close = _as_matrix(close, np.float64)
    rows, cols = close.shape
    high = close if high is None else _as_matrix(high, np.float64)
    low = close if low is None else _as_matrix(low, np.float64)
    if volume is None:
        volume = np.full((rows, cols), 1_000_000.0, dtype=np.float64)
    else:
        volume = _as_matrix(volume, np.float64)
    bid = close if bid is None else _as_matrix(bid, np.float64)
    ask = close if ask is None else _as_matrix(ask, np.float64)
    target_positions = _as_matrix(target_positions, np.int64)
    if order_types is None:
        order_types = np.full((rows, cols), int(OrderType.MARKET), dtype=np.int8)
    else:
        order_types = _as_matrix(order_types, np.int8)
    if limit_prices is None:
        limit_prices = np.full((rows, cols), np.nan, dtype=np.float64)
    else:
        limit_prices = _as_matrix(limit_prices, np.float64)
    if tradable_mask is None:
        tradable_mask = np.ones(rows, dtype=np.uint8)
    else:
        tradable_mask = np.asarray(tradable_mask, dtype=np.uint8)
    config = config or BacktestConfig()
    symbols = list(symbols or [f"asset_{idx}" for idx in range(cols)])

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
        config=config,
    )
    return PythonBacktestResult(
        raw=raw,
        symbols=symbols,
        positions=np.asarray(raw.positions, dtype=np.int64).reshape(rows, cols),
        equity_curve=np.asarray(raw.equity_curve, dtype=np.float64),
        cash_curve=np.asarray(raw.cash_curve, dtype=np.float64),
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
    config: BacktestConfig | None = None,
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
        config=config,
        symbols=["asset_0"],
    )
