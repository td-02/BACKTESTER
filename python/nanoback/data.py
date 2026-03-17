from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class AssetConfig:
    symbol: str
    max_position: int = 0
    notional_limit: float = 0.0
    lot_size: int = 1
    tick_size: float = 0.0


@dataclass(slots=True)
class MarketData:
    timestamps: np.ndarray
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    symbols: list[str]
    bid: np.ndarray | None = None
    ask: np.ndarray | None = None
    asset_configs: list[AssetConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.int64)
        self.close = _as_2d(self.close, np.float64)
        self.high = _as_2d(self.high, np.float64)
        self.low = _as_2d(self.low, np.float64)
        self.volume = _as_2d(self.volume, np.float64)
        self.bid = self.close.copy() if self.bid is None else _as_2d(self.bid, np.float64)
        self.ask = self.close.copy() if self.ask is None else _as_2d(self.ask, np.float64)
        if not (
            self.close.shape == self.high.shape == self.low.shape == self.volume.shape == self.bid.shape == self.ask.shape
        ):
            raise ValueError("close, high, low, volume, bid, and ask must share the same shape")
        if self.close.shape[0] != self.timestamps.shape[0]:
            raise ValueError("timestamps length must match the first matrix dimension")
        if self.close.shape[1] != len(self.symbols):
            raise ValueError("symbol count must match the second matrix dimension")
        if not self.asset_configs:
            self.asset_configs = [AssetConfig(symbol=symbol) for symbol in self.symbols]
        if len(self.asset_configs) != len(self.symbols):
            raise ValueError("asset_configs length must match symbol count")

    @property
    def shape(self) -> tuple[int, int]:
        return self.close.shape

    @property
    def asset_count(self) -> int:
        return self.close.shape[1]

    @property
    def row_count(self) -> int:
        return self.close.shape[0]

    @property
    def asset_max_positions(self) -> np.ndarray:
        return np.asarray([config.max_position for config in self.asset_configs], dtype=np.int64)

    @property
    def asset_notional_limits(self) -> np.ndarray:
        return np.asarray([config.notional_limit for config in self.asset_configs], dtype=np.float64)


def _as_2d(values: np.ndarray | Iterable[float], dtype: np.dtype) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        return np.ascontiguousarray(array.reshape(-1, 1))
    if array.ndim != 2:
        raise ValueError("expected a 1D or 2D array")
    return np.ascontiguousarray(array)


def _load_symbol_csv(path: Path) -> tuple[str, dict[int, tuple[float, float, float, float, float, float]]]:
    rows: dict[int, tuple[float, float, float, float, float, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "close", "high", "low", "volume"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            timestamp = int(row["timestamp"])
            close = float(row["close"])
            bid = float(row["bid"]) if "bid" in row and row["bid"] else close
            ask = float(row["ask"]) if "ask" in row and row["ask"] else close
            rows[timestamp] = (
                close,
                float(row["high"]),
                float(row["low"]),
                float(row["volume"]),
                bid,
                ask,
            )
    return path.stem, rows


def load_csv(path: str | Path) -> MarketData:
    source = Path(path)
    if source.is_dir():
        symbol_rows = [_load_symbol_csv(candidate) for candidate in sorted(source.glob("*.csv"))]
        if not symbol_rows:
            raise ValueError(f"no CSV files found in {source}")

        all_timestamps = sorted({timestamp for _, rows in symbol_rows for timestamp in rows})
        symbols = [symbol for symbol, _ in symbol_rows]
        row_count = len(all_timestamps)
        col_count = len(symbols)
        close = np.full((row_count, col_count), np.nan, dtype=np.float64)
        high = np.full((row_count, col_count), np.nan, dtype=np.float64)
        low = np.full((row_count, col_count), np.nan, dtype=np.float64)
        volume = np.zeros((row_count, col_count), dtype=np.float64)
        bid = np.full((row_count, col_count), np.nan, dtype=np.float64)
        ask = np.full((row_count, col_count), np.nan, dtype=np.float64)
        ts_index = {timestamp: idx for idx, timestamp in enumerate(all_timestamps)}

        for col, (_, rows) in enumerate(symbol_rows):
            for timestamp, values in rows.items():
                row = ts_index[timestamp]
                close[row, col], high[row, col], low[row, col], volume[row, col], bid[row, col], ask[row, col] = values

        return MarketData(
            timestamps=np.asarray(all_timestamps, dtype=np.int64),
            close=close,
            high=high,
            low=low,
            volume=volume,
            bid=bid,
            ask=ask,
            symbols=symbols,
        )

    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "symbol", "close", "high", "low", "volume"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source} is missing columns: {sorted(missing)}")

        records: dict[tuple[int, str], tuple[float, float, float, float, float, float]] = {}
        timestamps: set[int] = set()
        symbols: set[str] = set()
        for row in reader:
            timestamp = int(row["timestamp"])
            symbol = row["symbol"]
            close = float(row["close"])
            bid = float(row["bid"]) if "bid" in row and row["bid"] else close
            ask = float(row["ask"]) if "ask" in row and row["ask"] else close
            timestamps.add(timestamp)
            symbols.add(symbol)
            records[(timestamp, symbol)] = (
                close,
                float(row["high"]),
                float(row["low"]),
                float(row["volume"]),
                bid,
                ask,
            )

    ordered_ts = sorted(timestamps)
    ordered_symbols = sorted(symbols)
    row_count = len(ordered_ts)
    col_count = len(ordered_symbols)
    ts_index = {timestamp: idx for idx, timestamp in enumerate(ordered_ts)}
    sym_index = {symbol: idx for idx, symbol in enumerate(ordered_symbols)}
    close = np.full((row_count, col_count), np.nan, dtype=np.float64)
    high = np.full((row_count, col_count), np.nan, dtype=np.float64)
    low = np.full((row_count, col_count), np.nan, dtype=np.float64)
    volume = np.zeros((row_count, col_count), dtype=np.float64)
    bid = np.full((row_count, col_count), np.nan, dtype=np.float64)
    ask = np.full((row_count, col_count), np.nan, dtype=np.float64)

    for (timestamp, symbol), values in records.items():
        row = ts_index[timestamp]
        col = sym_index[symbol]
        close[row, col], high[row, col], low[row, col], volume[row, col], bid[row, col], ask[row, col] = values

    return MarketData(
        timestamps=np.asarray(ordered_ts, dtype=np.int64),
        close=close,
        high=high,
        low=low,
        volume=volume,
        bid=bid,
        ask=ask,
        symbols=ordered_symbols,
    )


def load_parquet(path: str | Path) -> MarketData:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("load_parquet requires pandas and a parquet backend such as pyarrow") from exc

    frame = pd.read_parquet(path)
    missing = {"timestamp", "symbol", "close", "high", "low", "volume"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    timestamps = np.sort(frame["timestamp"].unique().astype(np.int64))
    symbols = sorted(frame["symbol"].astype(str).unique().tolist())
    row_count = len(timestamps)
    col_count = len(symbols)
    ts_index = {int(timestamp): idx for idx, timestamp in enumerate(timestamps)}
    sym_index = {symbol: idx for idx, symbol in enumerate(symbols)}
    close = np.full((row_count, col_count), np.nan, dtype=np.float64)
    high = np.full((row_count, col_count), np.nan, dtype=np.float64)
    low = np.full((row_count, col_count), np.nan, dtype=np.float64)
    volume = np.zeros((row_count, col_count), dtype=np.float64)
    bid = np.full((row_count, col_count), np.nan, dtype=np.float64)
    ask = np.full((row_count, col_count), np.nan, dtype=np.float64)

    for record in frame.itertuples(index=False):
        row = ts_index[int(record.timestamp)]
        col = sym_index[str(record.symbol)]
        close[row, col] = float(record.close)
        high[row, col] = float(record.high)
        low[row, col] = float(record.low)
        volume[row, col] = float(record.volume)
        bid[row, col] = float(getattr(record, "bid", record.close))
        ask[row, col] = float(getattr(record, "ask", record.close))

    return MarketData(
        timestamps=timestamps,
        close=close,
        high=high,
        low=low,
        volume=volume,
        bid=bid,
        ask=ask,
        symbols=symbols,
    )
