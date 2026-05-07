from __future__ import annotations

import json
import math
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from ._nanoback import BacktestConfig, EngineSnapshot, OrderType
from .wrapper import PythonBacktestResult, run_backtest_matrix


@dataclass(slots=True)
class PaperTick:
    timestamp_ns: int
    symbol: str
    price: float
    size: float = 1.0
    side: str = "trade"


class FeedAdapter(Protocol):
    def next_tick(self, timeout_seconds: float | None = None) -> PaperTick | None:
        ...

    def fetch_positions(self) -> dict[str, float]:
        ...

    def submit_order(self, symbol: str, quantity_delta: float) -> None:
        ...


class PaperBroker:
    def __init__(
        self,
        *,
        symbols: Sequence[str],
        strategy: Callable[[PaperTick, dict[str, object]], Sequence[int] | dict[str, int]],
        config: BacktestConfig | None = None,
        feed: FeedAdapter,
        ledger_path: str | Path | None = None,
        reconcile_callback: Callable[[dict[str, int], int], object] | None = None,
        async_ledger_flush: bool = True,
        ledger_queue_size: int = 1_000_000,
    ) -> None:
        self.symbols = list(symbols)
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        self._symbol_to_asset = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.feed = feed
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.reconcile_callback = reconcile_callback
        self.async_ledger_flush = bool(async_ledger_flush)
        self.ledger_queue_size = max(1, int(ledger_queue_size))

        self.snapshot: EngineSnapshot | None = None
        self._latest_price = np.full(len(self.symbols), np.nan, dtype=np.float64)
        self._latest_bid = np.full(len(self.symbols), np.nan, dtype=np.float64)
        self._latest_ask = np.full(len(self.symbols), np.nan, dtype=np.float64)
        self._positions = np.zeros(len(self.symbols), dtype=np.int64)

        self.results: list[PythonBacktestResult] = []
        self.fills = []
        self.ledger = []
        self._ledger_writer = (
            _AsyncLedgerWriter(self.ledger_path, queue_size=self.ledger_queue_size)
            if self.ledger_path is not None and self.async_ledger_flush
            else None
        )

    @property
    def positions(self) -> dict[str, int]:
        return {symbol: int(self._positions[idx]) for idx, symbol in enumerate(self.symbols)}

    def run_until(self, end_time: datetime) -> None:
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        while datetime.now(timezone.utc) < end_time:
            timeout = max(0.0, min(1.0, (end_time - datetime.now(timezone.utc)).total_seconds()))
            tick = self.feed.next_tick(timeout_seconds=timeout)
            if tick is None:
                if timeout <= 1e-6:
                    break
                continue
            if tick.symbol not in self._symbol_to_asset:
                continue
            result = self._ingest_tick(tick)
            if len(result.fills) > 0 and self.reconcile_callback is not None:
                self.reconcile_callback(self.positions, int(tick.timestamp_ns))
            if datetime.now(timezone.utc) >= end_time:
                break
        self.flush_ledger()

    def _ingest_tick(self, tick: PaperTick) -> PythonBacktestResult:
        idx = self._symbol_to_asset[tick.symbol]
        px = float(tick.price)
        size = max(float(tick.size), 1.0)

        if tick.side == "bid":
            self._latest_bid[idx] = px
        elif tick.side == "ask":
            self._latest_ask[idx] = px
        else:
            self._latest_price[idx] = px

        if not np.isfinite(self._latest_price[idx]):
            # Use quote midpoint or quote itself until first trade prints.
            bid = self._latest_bid[idx]
            ask = self._latest_ask[idx]
            if np.isfinite(bid) and np.isfinite(ask):
                self._latest_price[idx] = 0.5 * (bid + ask)
            elif np.isfinite(bid):
                self._latest_price[idx] = bid
            elif np.isfinite(ask):
                self._latest_price[idx] = ask
            else:
                self._latest_price[idx] = px

        close = np.where(np.isfinite(self._latest_price), self._latest_price, px).reshape(1, -1)
        bid = np.where(np.isfinite(self._latest_bid), self._latest_bid, close[0]).reshape(1, -1)
        ask = np.where(np.isfinite(self._latest_ask), self._latest_ask, close[0]).reshape(1, -1)
        high = close.copy()
        low = close.copy()
        volume = np.full((1, len(self.symbols)), size, dtype=np.float64)

        targets = self._resolve_targets(tick, close[0]).reshape(1, -1)
        order_types = np.full((1, len(self.symbols)), int(OrderType.MARKET), dtype=np.int8)
        limit_prices = np.full((1, len(self.symbols)), math.nan, dtype=np.float64)

        result = run_backtest_matrix(
            timestamps=np.array([int(tick.timestamp_ns)], dtype=np.int64),
            close=close,
            high=high,
            low=low,
            volume=volume,
            bid=bid,
            ask=ask,
            target_positions=targets,
            order_types=order_types,
            limit_prices=limit_prices,
            config=self.config,
            symbols=self.symbols,
            snapshot=self.snapshot,
            start_row=0,
            end_row=1,
        )
        self.snapshot = result.snapshot
        self._positions = np.asarray(result.snapshot.positions, dtype=np.int64)
        self.results.append(result)
        self.fills.extend(result.fills)
        self.ledger.extend(result.ledger)
        if self._ledger_writer is not None:
            for entry in result.ledger:
                self._ledger_writer.push(entry)
        return result

    def _resolve_targets(self, tick: PaperTick, latest_close: np.ndarray) -> np.ndarray:
        state = {
            "symbols": self.symbols,
            "positions": self.positions,
            "latest_close": latest_close.copy(),
            "snapshot": self.snapshot,
        }
        out = self.strategy(tick, state)
        if isinstance(out, dict):
            targets = self._positions.copy()
            for symbol, position in out.items():
                if symbol in self._symbol_to_asset:
                    targets[self._symbol_to_asset[symbol]] = int(position)
            return targets
        arr = np.asarray(out, dtype=np.int64)
        if arr.ndim != 1 or arr.shape[0] != len(self.symbols):
            raise ValueError("strategy must return dict[symbol, target] or 1D targets with len(symbols)")
        return arr

    def flush_ledger(self) -> None:
        if self._ledger_writer is not None:
            self._ledger_writer.close()
            return
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("w", encoding="utf-8") as handle:
            for entry in self.ledger:
                handle.write(
                    json.dumps(
                        {
                            "sequence": int(entry.sequence),
                            "timestamp": int(entry.timestamp),
                            "order_id": int(entry.order_id),
                            "parent_order_id": int(entry.parent_order_id),
                            "asset": int(entry.asset),
                            "type": int(entry.type),
                            "quantity": int(entry.quantity),
                            "remaining_quantity": int(entry.remaining_quantity),
                            "price": float(entry.price),
                            "cash_after": float(entry.cash_after),
                            "equity_after": float(entry.equity_after),
                            "value": float(entry.value),
                        }
                    )
                    + "\n"
                )


class _AsyncLedgerWriter:
    def __init__(self, path: Path, queue_size: int) -> None:
        self.path = path
        self._queue: queue.Queue[object | None] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, entry: object) -> None:
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            self._queue.put(entry)

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                entry = item
                handle.write(
                    json.dumps(
                        {
                            "sequence": int(entry.sequence),
                            "timestamp": int(entry.timestamp),
                            "order_id": int(entry.order_id),
                            "parent_order_id": int(entry.parent_order_id),
                            "asset": int(entry.asset),
                            "type": int(entry.type),
                            "quantity": int(entry.quantity),
                            "remaining_quantity": int(entry.remaining_quantity),
                            "price": float(entry.price),
                            "cash_after": float(entry.cash_after),
                            "equity_after": float(entry.equity_after),
                            "value": float(entry.value),
                        }
                    )
                    + "\n"
                )


class AlpacaFeedAdapter:
    def __init__(self) -> None:
        raise NotImplementedError("Alpaca adapter requires runtime credentials and websocket integration")


class YFinanceFeedAdapter:
    def __init__(self) -> None:
        raise NotImplementedError("yfinance polling adapter requires runtime network integration")


class BinanceFeedAdapter:
    def __init__(self) -> None:
        raise NotImplementedError("Binance adapter requires runtime credentials and websocket integration")
