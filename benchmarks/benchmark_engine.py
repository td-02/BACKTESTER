from __future__ import annotations

import time

import numpy as np

import nanoback as nb


def main() -> None:
    rows = 50_000
    cols = 8
    rng = np.random.default_rng(42)
    timestamps = np.arange(rows, dtype=np.int64) * 60 + 34_200
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.1, size=(rows, cols)), axis=0)
    spread = np.abs(rng.normal(0.15, 0.03, size=(rows, cols)))
    high = close + spread
    low = close - spread
    volume = rng.integers(5_000, 20_000, size=(rows, cols)).astype(np.float64)
    started = time.perf_counter()
    data = nb.MarketData(
        timestamps=timestamps,
        close=close,
        high=high,
        low=low,
        volume=volume,
        symbols=[f"asset_{idx}" for idx in range(cols)],
    )
    result = nb.run_compiled_policy_backtest(
        data=data,
        policy="moving_average_crossover",
        fast_window=8,
        slow_window=32,
        max_position=5,
        config=nb.BacktestConfig(
            starting_cash=10_000_000.0,
            commission_bps=0.1,
            slippage_bps=0.2,
            max_position=5,
            max_participation_rate=0.2,
            latency_steps=1,
        ),
    )
    elapsed = time.perf_counter() - started

    print(f"elapsed_seconds={elapsed:.4f}")
    print(f"rows={rows} cols={cols}")
    print(f"fills={len(result.fills)} pnl={result.pnl:.2f}")


if __name__ == "__main__":
    main()
