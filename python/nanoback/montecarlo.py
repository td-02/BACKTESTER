from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _returns_from_equity(equity: np.ndarray) -> np.ndarray:
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size <= 1:
        return np.empty(0, dtype=np.float64)
    prev = eq[:-1]
    curr = eq[1:]
    valid = prev != 0.0
    out = np.zeros(eq.size - 1, dtype=np.float64)
    out[valid] = (curr[valid] - prev[valid]) / prev[valid]
    return out


def _drawdown(curves: np.ndarray) -> np.ndarray:
    peaks = np.maximum.accumulate(curves, axis=1)
    safe_peaks = np.where(peaks == 0.0, 1.0, peaks)
    dd = (peaks - curves) / safe_peaks
    dd[peaks == 0.0] = 0.0
    return np.max(dd, axis=1)


def _sharpe(returns: np.ndarray) -> np.ndarray:
    mean = np.mean(returns, axis=1)
    std = np.std(returns, axis=1)
    out = np.zeros(mean.shape[0], dtype=np.float64)
    valid = std > 0.0
    out[valid] = mean[valid] / std[valid] * np.sqrt(252.0)
    out[(~valid) & (mean > 0)] = np.inf
    out[(~valid) & (mean < 0)] = -np.inf
    return out


def _cagr(curves: np.ndarray, years: float) -> np.ndarray:
    start = curves[:, 0]
    end = curves[:, -1]
    out = np.zeros(curves.shape[0], dtype=np.float64)
    valid = (start > 0.0) & (end > 0.0) & (years > 0.0)
    out[valid] = np.power(end[valid] / start[valid], 1.0 / years) - 1.0
    return out


@dataclass(slots=True)
class MonteCarloResult:
    sharpe: np.ndarray
    max_drawdown: np.ndarray
    cagr: np.ndarray
    sharpe_percentiles: dict[str, float]
    max_drawdown_percentiles: dict[str, float]
    cagr_percentiles: dict[str, float]
    p_value_proxy: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sharpe_percentiles": self.sharpe_percentiles,
            "max_drawdown_percentiles": self.max_drawdown_percentiles,
            "cagr_percentiles": self.cagr_percentiles,
            "p_value_proxy": self.p_value_proxy,
        }


class MonteCarlo:
    def __init__(self, returns: np.ndarray, *, seed: int = 42):
        self.returns = np.asarray(returns, dtype=np.float64).reshape(-1)
        self.seed = int(seed)

    @classmethod
    def from_backtest(cls, result: object, *, seed: int = 42) -> "MonteCarlo":
        equity = np.asarray(getattr(result, "equity_curve", ()), dtype=np.float64)
        return cls(_returns_from_equity(equity), seed=seed)

    def _simulated_returns(
        self,
        *,
        n_sims: int,
        method: Literal["shuffle", "block_bootstrap"],
        block_size: int = 20,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if self.returns.size == 0:
            return np.zeros((n_sims, 0), dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        n = self.returns.size
        if batch_size is None or batch_size <= 0:
            batch_size = n_sims
        if method == "shuffle":
            out = np.empty((n_sims, n), dtype=np.float64)
            base_idx = np.arange(n, dtype=np.int64)
            cursor = 0
            while cursor < n_sims:
                take = min(batch_size, n_sims - cursor)
                idx = np.tile(base_idx, (take, 1))
                idx = rng.permuted(idx, axis=1)
                out[cursor:cursor + take] = self.returns[idx]
                cursor += take
            return out
        if method == "block_bootstrap":
            b = max(1, int(block_size))
            n_blocks = int(np.ceil(n / b))
            out = np.empty((n_sims, n), dtype=np.float64)
            cursor = 0
            while cursor < n_sims:
                take = min(batch_size, n_sims - cursor)
                starts = rng.integers(0, max(1, n - b + 1), size=(take, n_blocks))
                block_idx = starts[..., None] + np.arange(b)[None, None, :]
                block_idx = np.clip(block_idx, 0, n - 1).reshape(take, n_blocks * b)[:, :n]
                out[cursor:cursor + take] = self.returns[block_idx]
                cursor += take
            return out
        raise ValueError(f"unknown method: {method}")

    def run(
        self,
        *,
        n_sims: int = 1_000,
        method: Literal["shuffle", "block_bootstrap"] = "shuffle",
        block_size: int = 20,
        batch_size: int | None = 1_000,
    ) -> MonteCarloResult:
        sims = self._simulated_returns(n_sims=n_sims, method=method, block_size=block_size, batch_size=batch_size)
        if sims.shape[1] == 0:
            zeros = np.zeros(n_sims, dtype=np.float64)
            percentiles = {"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
            return MonteCarloResult(zeros, zeros, zeros, percentiles, percentiles, percentiles, 1.0)

        equity = np.cumprod(1.0 + sims, axis=1)
        equity = np.concatenate([np.ones((n_sims, 1), dtype=np.float64), equity], axis=1)
        sharpe = _sharpe(sims)
        mdd = _drawdown(equity)
        years = sims.shape[1] / 252.0
        cagr = _cagr(equity, years)

        base_sharpe = float(_sharpe(self.returns.reshape(1, -1))[0]) if self.returns.size else 0.0
        p_proxy = float(np.mean(sharpe > base_sharpe))

        def pct(values: np.ndarray) -> dict[str, float]:
            q = np.percentile(values, [5, 25, 50, 75, 95])
            return {"p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p95": float(q[4])}

        return MonteCarloResult(
            sharpe=sharpe,
            max_drawdown=mdd,
            cagr=cagr,
            sharpe_percentiles=pct(sharpe),
            max_drawdown_percentiles=pct(mdd),
            cagr_percentiles=pct(cagr),
            p_value_proxy=p_proxy,
        )
