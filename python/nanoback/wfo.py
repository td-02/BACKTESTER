from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .analytics import summarize_result
from .data import MarketData
from .sweep import ParamGrid, Sweep


def _slice_market_data(data: MarketData, start: int, end: int) -> MarketData:
    return MarketData(
        timestamps=data.timestamps[start:end],
        close=data.close[start:end],
        high=data.high[start:end],
        low=data.low[start:end],
        volume=data.volume[start:end],
        bid=data.bid[start:end],
        ask=data.ask[start:end],
        symbols=list(data.symbols),
        asset_configs=list(data.asset_configs),
    )


@dataclass(slots=True)
class WFOFold:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    best_params: dict[str, Any]
    is_sharpe: float
    is_max_drawdown: float
    oos_sharpe: float
    oos_max_drawdown: float


@dataclass(slots=True)
class WFOResult:
    folds: list[WFOFold]
    oos_equity_curve: np.ndarray
    efficiency_ratio: float

    def to_dict(self) -> dict[str, Any]:
        def finite_mean(values: list[float]) -> float:
            arr = np.asarray(values, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return 0.0
            return float(np.mean(finite))

        return {
            "folds": [asdict(fold) for fold in self.folds],
            "oos_equity_curve": self.oos_equity_curve.tolist(),
            "efficiency_ratio": self.efficiency_ratio,
            "mean_is_sharpe": finite_mean([fold.is_sharpe for fold in self.folds]) if self.folds else 0.0,
            "mean_oos_sharpe": finite_mean([fold.oos_sharpe for fold in self.folds]) if self.folds else 0.0,
        }


class WalkForward:
    def __init__(self, *, n_splits: int, train_frac: float = 0.7, anchored: bool = False):
        if n_splits <= 0:
            raise ValueError("n_splits must be > 0")
        if not (0.0 < train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1)")
        self.n_splits = int(n_splits)
        self.train_frac = float(train_frac)
        self.anchored = bool(anchored)

    def _windows(self, n_rows: int) -> list[tuple[int, int, int, int]]:
        split_points = np.linspace(0, n_rows, self.n_splits + 1, dtype=int)
        windows: list[tuple[int, int, int, int]] = []
        for idx in range(self.n_splits):
            win_start = int(split_points[idx])
            win_end = int(split_points[idx + 1])
            if win_end - win_start < 3:
                continue
            train_len = max(1, int((win_end - win_start) * self.train_frac))
            if train_len >= (win_end - win_start):
                train_len = (win_end - win_start) - 1
            if self.anchored:
                train_start = 0
                train_end = win_start + train_len
                test_start = train_end
                test_end = win_end
            else:
                train_start = win_start
                train_end = win_start + train_len
                test_start = train_end
                test_end = win_end
            if test_end - test_start <= 0 or train_end - train_start <= 0:
                continue
            windows.append((train_start, train_end, test_start, test_end))
        return windows

    def run(
        self,
        data: MarketData,
        strategy: Callable[..., object],
        param_grid: ParamGrid | dict[str, Sequence[Any]],
        *,
        n_jobs: int = -1,
        compiled: bool = False,
    ) -> WFOResult:
        windows = self._windows(data.row_count)
        folds: list[WFOFold] = []
        oos_equity_parts: list[np.ndarray] = []

        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            train_data = _slice_market_data(data, train_start, train_end)
            test_data = _slice_market_data(data, test_start, test_end)

            sweep = Sweep(train_data)
            sweep_result = sweep.run(strategy, param_grid, n_jobs=n_jobs, compiled=compiled)
            best = sweep_result.best()
            params = {key: value for key, value in best.items() if key not in {
                "sharpe", "sortino", "cagr", "max_drawdown", "calmar", "turnover_per_year", "fill_count", "pnl",
                "is_sharpe", "is_max_drawdown", "oos_sharpe", "oos_max_drawdown", "overfit_warning"
            }}

            test_result = strategy(test_data, **params)
            oos_summary = summarize_result(test_result, symbols=test_data.symbols)
            oos_equity = np.asarray(getattr(test_result, "equity_curve", ()), dtype=np.float64)
            if oos_equity.size:
                oos_equity_parts.append(oos_equity)

            is_sharpe = float(best["sharpe"])
            oos_sharpe = float(oos_summary.sharpe)
            fold = WFOFold(
                fold=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=params,
                is_sharpe=is_sharpe,
                is_max_drawdown=float(best["max_drawdown"]),
                oos_sharpe=oos_sharpe,
                oos_max_drawdown=float(oos_summary.max_drawdown),
            )
            folds.append(fold)

        def _finite_mean(values: list[float]) -> float:
            arr = np.asarray(values, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return 0.0
            return float(np.mean(finite))

        mean_is = _finite_mean([fold.is_sharpe for fold in folds]) if folds else 0.0
        mean_oos = _finite_mean([fold.oos_sharpe for fold in folds]) if folds else 0.0
        efficiency = mean_oos / mean_is if mean_is != 0.0 else 0.0
        oos_curve = np.concatenate(oos_equity_parts) if oos_equity_parts else np.empty(0, dtype=np.float64)
        return WFOResult(folds=folds, oos_equity_curve=oos_curve, efficiency_ratio=float(efficiency))
