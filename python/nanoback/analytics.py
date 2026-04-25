from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY


def _timestamp_scale_divisor(timestamps: np.ndarray) -> float:
    if timestamps.size == 0:
        return 1.0
    max_abs = float(np.max(np.abs(timestamps)))
    if max_abs > 1e17:
        return 1e9
    if max_abs > 1e14:
        return 1e6
    if max_abs > 1e11:
        return 1e3
    return 1.0


def _infer_years(timestamps: np.ndarray, observations: int) -> float:
    if timestamps.size >= 2:
        max_abs = float(np.max(np.abs(timestamps.astype(np.float64, copy=False))))
        if max_abs < 10_000_000.0:
            elapsed_steps = float(timestamps[-1] - timestamps[0])
            if elapsed_steps > 0.0:
                return max(elapsed_steps / 252.0, 1.0 / 252.0)
        div = _timestamp_scale_divisor(timestamps.astype(np.float64, copy=False))
        elapsed_seconds = (float(timestamps[-1]) - float(timestamps[0])) / div
        if elapsed_seconds > 0:
            return max(elapsed_seconds / SECONDS_PER_YEAR, 1.0 / 252.0)
    if observations > 1:
        return (observations - 1) / 252.0
    return 1.0 / 252.0


def _daily_equity(timestamps: np.ndarray, equity: np.ndarray) -> np.ndarray:
    if equity.size <= 1:
        return equity
    if timestamps.size != equity.size:
        return equity
    max_abs = float(np.max(np.abs(timestamps.astype(np.float64, copy=False))))
    if max_abs < 10_000_000.0:
        return equity
    scale = _timestamp_scale_divisor(timestamps.astype(np.float64, copy=False))
    day_index = np.floor((timestamps.astype(np.float64) / scale) / SECONDS_PER_DAY).astype(np.int64)
    if np.unique(day_index).size == equity.size:
        return equity
    _, first_idx = np.unique(day_index, return_index=True)
    end_idx = np.r_[first_idx[1:] - 1, equity.size - 1]
    return equity[end_idx]


def _returns(equity: np.ndarray) -> np.ndarray:
    if equity.size <= 1:
        return np.empty(0, dtype=np.float64)
    prev = equity[:-1]
    curr = equity[1:]
    valid = prev != 0.0
    if not np.any(valid):
        return np.empty(0, dtype=np.float64)
    out = np.zeros(prev.shape[0], dtype=np.float64)
    out[valid] = (curr[valid] - prev[valid]) / prev[valid]
    return out


def _sharpe_ratio(daily_returns: np.ndarray) -> float:
    if daily_returns.size == 0:
        return 0.0
    mean = float(np.mean(daily_returns))
    std = float(np.std(daily_returns, ddof=0))
    if std <= 0.0:
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return 0.0
    return mean / std * np.sqrt(252.0)


def _sortino_ratio(daily_returns: np.ndarray) -> float:
    if daily_returns.size == 0:
        return 0.0
    mean = float(np.mean(daily_returns))
    downside = np.minimum(daily_returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    if downside_dev <= 0.0:
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return 0.0
    return mean / downside_dev * np.sqrt(252.0)


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    if equity.size == 0:
        return np.empty(0, dtype=np.float64)
    running_peak = np.maximum.accumulate(equity)
    safe_peak = np.where(running_peak == 0.0, 1.0, running_peak)
    drawdown = (running_peak - equity) / safe_peak
    drawdown[running_peak == 0.0] = 0.0
    return drawdown


def _pnl_attribution(
    assets: np.ndarray,
    quantities: np.ndarray,
    prices: np.ndarray,
    fees: np.ndarray,
    symbols: Sequence[str] | None,
) -> dict[str, float]:
    if assets.size == 0:
        return {}
    cashflow = -quantities.astype(np.float64) * prices - fees
    unique_assets = np.unique(assets)
    labels = list(symbols or [])
    out: dict[str, float] = {}
    for asset in unique_assets:
        mask = assets == asset
        key = labels[int(asset)] if int(asset) < len(labels) else str(int(asset))
        out[key] = float(np.sum(cashflow[mask]))
    return out


def _fill_trade_stats(
    assets: np.ndarray,
    quantities: np.ndarray,
    prices: np.ndarray,
    fees: np.ndarray,
) -> tuple[float, float, float, float]:
    if quantities.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    realized: list[float] = []
    for asset in np.unique(assets):
        mask = assets == asset
        q = quantities[mask]
        p = prices[mask]
        f = fees[mask]
        position = 0.0
        avg_cost = 0.0
        for qty, px, fee in zip(q, p, f):
            signed = float(qty)
            if position == 0.0:
                position = signed
                avg_cost = px
                continue
            same_side = np.sign(position) == np.sign(signed)
            if same_side:
                total = abs(position) + abs(signed)
                if total > 0.0:
                    avg_cost = (abs(position) * avg_cost + abs(signed) * px) / total
                position += signed
                continue

            close_qty = min(abs(position), abs(signed))
            pnl = close_qty * (px - avg_cost) * np.sign(position) - float(fee)
            realized.append(float(pnl))
            position += signed
            if position == 0.0:
                avg_cost = 0.0
            elif np.sign(position) == np.sign(signed):
                avg_cost = px

    if not realized:
        return 0.0, 0.0, 0.0, 0.0
    realized_arr = np.asarray(realized, dtype=np.float64)
    wins = realized_arr[realized_arr > 0.0]
    losses = realized_arr[realized_arr < 0.0]
    win_rate = float(wins.size / realized_arr.size)
    avg_win = float(np.mean(wins)) if wins.size else 0.0
    avg_loss = float(np.mean(losses)) if losses.size else 0.0
    gross_wins = float(np.sum(wins)) if wins.size else 0.0
    gross_losses = float(-np.sum(losses)) if losses.size else 0.0
    if gross_losses == 0.0:
        profit_factor = float("inf") if gross_wins > 0.0 else 0.0
    else:
        profit_factor = gross_wins / gross_losses
    return win_rate, avg_win, avg_loss, profit_factor


@dataclass(slots=True)
class EquityCurve:
    timestamps: np.ndarray
    equity: np.ndarray
    pnl: np.ndarray


@dataclass(slots=True)
class BacktestSummary:
    sharpe: float
    sortino: float
    cagr: float
    max_drawdown: float
    calmar: float
    turnover_per_year: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    fill_count: int
    pnl: float
    ending_equity: float
    drawdown_series: np.ndarray
    pnl_attribution: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["drawdown_series"] = self.drawdown_series.tolist()
        return payload


def equity_curve_from_ledger(ledger: Sequence[object]) -> EquityCurve:
    if not ledger:
        return EquityCurve(
            timestamps=np.empty(0, dtype=np.int64),
            equity=np.empty(0, dtype=np.float64),
            pnl=np.empty(0, dtype=np.float64),
        )
    timestamps = np.asarray([int(entry.timestamp) for entry in ledger], dtype=np.int64)
    equity_after = np.asarray([float(entry.equity_after) for entry in ledger], dtype=np.float64)
    uniq_ts, first_idx = np.unique(timestamps, return_index=True)
    last_idx = np.r_[first_idx[1:] - 1, timestamps.size - 1]
    equity = equity_after[last_idx]
    pnl = equity - equity[0] if equity.size else np.empty(0, dtype=np.float64)
    return EquityCurve(timestamps=uniq_ts, equity=equity, pnl=pnl)


def summarize_result(result: object, symbols: Sequence[str] | None = None) -> BacktestSummary:
    ledger = list(getattr(result, "ledger", ()))
    fills = list(getattr(result, "fills", ()))
    curve = equity_curve_from_ledger(ledger)

    equity = curve.equity
    if equity.size == 0:
        equity = np.asarray(getattr(result, "equity_curve", ()), dtype=np.float64)
        timestamps = np.asarray(getattr(result, "timestamps", np.arange(equity.size)), dtype=np.int64)
    else:
        timestamps = curve.timestamps

    if equity.size == 0:
        equity = np.zeros(1, dtype=np.float64)
        timestamps = np.zeros(1, dtype=np.int64)

    drawdown = _drawdown_series(equity)
    max_drawdown = float(np.max(drawdown)) if drawdown.size else 0.0
    daily_eq = _daily_equity(timestamps, equity)
    daily_returns = _returns(daily_eq)
    sharpe = _sharpe_ratio(daily_returns)
    sortino = _sortino_ratio(daily_returns)

    years = _infer_years(timestamps, equity.size)
    start_equity = float(equity[0]) if equity.size else 0.0
    end_equity = float(equity[-1]) if equity.size else 0.0
    if start_equity <= 0.0 or end_equity <= 0.0:
        cagr = 0.0
    else:
        cagr = float((end_equity / start_equity) ** (1.0 / years) - 1.0)
    if max_drawdown == 0.0:
        calmar = float("inf") if cagr > 0.0 else 0.0
    else:
        calmar = cagr / max_drawdown

    fill_assets = np.asarray([int(fill.asset) for fill in fills], dtype=np.int64) if fills else np.empty(0, dtype=np.int64)
    fill_qty = np.asarray([int(fill.quantity) for fill in fills], dtype=np.int64) if fills else np.empty(0, dtype=np.int64)
    fill_prices = np.asarray([float(fill.price) for fill in fills], dtype=np.float64) if fills else np.empty(0, dtype=np.float64)
    fill_fees = np.asarray([float(fill.fee) for fill in fills], dtype=np.float64) if fills else np.empty(0, dtype=np.float64)

    gross_position_change = float(np.sum(np.abs(fill_qty.astype(np.float64) * fill_prices)))
    mean_aum = float(np.mean(np.abs(equity))) if equity.size else 0.0
    turnover = gross_position_change / mean_aum if mean_aum > 0.0 else 0.0
    turnover_per_year = turnover / years if years > 0.0 else 0.0

    pnl_attr = _pnl_attribution(fill_assets, fill_qty, fill_prices, fill_fees, symbols or getattr(result, "symbols", None))
    win_rate, avg_win, avg_loss, profit_factor = _fill_trade_stats(fill_assets, fill_qty, fill_prices, fill_fees)

    return BacktestSummary(
        sharpe=float(sharpe),
        sortino=float(sortino),
        cagr=float(cagr),
        max_drawdown=max_drawdown,
        calmar=float(calmar),
        turnover_per_year=float(turnover_per_year),
        win_rate=float(win_rate),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        profit_factor=float(profit_factor),
        fill_count=int(fill_qty.size),
        pnl=float(end_equity - start_equity),
        ending_equity=end_equity,
        drawdown_series=drawdown.astype(np.float64, copy=False),
        pnl_attribution=pnl_attr,
    )
