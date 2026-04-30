from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

from ._nanoback import CorporateAction, CorporateActionType, TickSide


def load_corporate_actions_csv(path: str | Path, symbol_to_asset: dict[str, int]) -> list[CorporateAction]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_corporate_actions_csv requires pandas") from exc

    frame = pd.read_csv(path)
    required = {"symbol", "ex_date", "action_type", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out: list[CorporateAction] = []
    for row in frame.itertuples(index=False):
        symbol = str(row.symbol)
        if symbol not in symbol_to_asset:
            continue
        action_name = str(row.action_type).strip().upper()
        action_type = {
            "SPLIT": CorporateActionType.SPLIT,
            "DIVIDEND": CorporateActionType.DIVIDEND,
            "SPINOFF": CorporateActionType.SPINOFF,
            "DELISTING": CorporateActionType.DELISTING,
        }.get(action_name)
        if action_type is None:
            continue
        ex_ts = int(pd.Timestamp(row.ex_date).value // 1_000_000_000)
        action = CorporateAction()
        action.asset = int(symbol_to_asset[symbol])
        action.ex_date_timestamp = ex_ts
        action.action_type = action_type
        action.ratio_or_amount = float(row.value)
        out.append(action)
    out.sort(key=lambda x: (x.ex_date_timestamp, x.asset))
    return out


def load_ticks_parquet(path: str | Path):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_ticks_parquet requires pandas and pyarrow") from exc

    frame = pd.read_parquet(path)
    required = {"timestamp", "symbol", "price", "size", "side"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    symbols = sorted(frame["symbol"].astype(str).unique().tolist())
    symbol_to_asset = {symbol: idx for idx, symbol in enumerate(symbols)}
    ts = frame["timestamp"].astype("int64").to_numpy()
    price = frame["price"].astype(float).to_numpy()
    size = frame["size"].astype(float).to_numpy()
    asset = frame["symbol"].astype(str).map(symbol_to_asset).astype("int64").to_numpy()
    side = (
        frame["side"].astype(str).str.upper().map({"BID": int(TickSide.BID), "ASK": int(TickSide.ASK), "TRADE": int(TickSide.TRADE)})
        .fillna(int(TickSide.TRADE))
        .astype("int8")
        .to_numpy()
    )
    order = np.argsort(ts, kind="stable")
    return {
        "timestamp_ns": ts[order],
        "asset": asset[order],
        "price": price[order],
        "size": size[order],
        "side": side[order],
        "symbols": symbols,
        "symbol_to_asset": symbol_to_asset,
    }


def load_yahoo_adjusted(
    tickers: Sequence[str],
    start: str,
    end: str,
):
    try:
        import pandas as pd
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_yahoo_adjusted requires pandas and yfinance") from exc

    frame = yf.download(list(tickers), start=start, end=end, auto_adjust=False, progress=False, actions=True)
    if frame.empty:
        raise ValueError("no data returned from yfinance")
    close = frame["Close"].copy()
    adj_close = frame["Adj Close"].copy()
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
        adj_close = adj_close.to_frame(tickers[0])
    adjustment = (adj_close / close).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    adjusted_prices = close * adjustment

    returns = adjusted_prices.pct_change().fillna(0.0)
    actions = frame.get("Stock Splits")
    if actions is None:
        actions = 0.0
    if isinstance(actions, pd.Series):
        actions = actions.to_frame(tickers[0])
    divs = frame.get("Dividends")
    if divs is None:
        divs = 0.0
    if isinstance(divs, pd.Series):
        divs = divs.to_frame(tickers[0])

    for symbol in adjusted_prices.columns:
        spike_idx = returns.index[np.abs(returns[symbol]) > 0.30]
        for dt in spike_idx:
            split_value = float(actions.loc[dt, symbol]) if symbol in actions.columns else 0.0
            div_value = float(divs.loc[dt, symbol]) if symbol in divs.columns else 0.0
            if split_value == 0.0 and div_value == 0.0:
                warnings.warn(
                    f"suspicious jump without corporate action: symbol={symbol} date={dt.date()} return={returns.loc[dt, symbol]:.2%}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    timestamps = (adjusted_prices.index.view("int64") // 1_000_000_000).astype(np.int64)
    return {
        "timestamps": timestamps,
        "prices": adjusted_prices.to_numpy(dtype=np.float64),
        "symbols": list(adjusted_prices.columns),
        "adjustment_factors": adjustment.to_numpy(dtype=np.float64),
    }

