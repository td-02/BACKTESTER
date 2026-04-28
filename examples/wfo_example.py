from __future__ import annotations

import numpy as np

import nanoback as nb


def momentum_strategy(data: nb.MarketData, lookback: int, max_position: int):
    return nb.run_compiled_policy_backtest(
        data,
        policy="momentum",
        lookback=lookback,
        max_position=max_position,
        config=nb.BacktestConfig(
            max_position=max_position,
            slippage_bps=0.0,
            volume_share_impact=0.0,
        ),
    )


def main() -> None:
    rows = 800
    timestamps = np.arange(rows, dtype=np.int64) + 1
    close = (100.0 + np.cumsum(np.random.default_rng(42).normal(0.0, 0.25, size=rows))).reshape(-1, 1)
    data = nb.MarketData(
        timestamps=timestamps,
        close=close,
        high=close,
        low=close,
        volume=np.full_like(close, 25_000.0),
        bid=close,
        ask=close,
        symbols=["asset_0"],
    )

    wfo = nb.WalkForward(n_splits=6, train_frac=0.7, anchored=True)
    result = wfo.run(
        data,
        momentum_strategy,
        nb.ParamGrid({"lookback": [5, 10, 20, 30], "max_position": [1, 2]}),
        n_jobs=1,
        compiled=True,
    )

    print(f"folds={len(result.folds)}")
    print(f"efficiency_ratio={result.efficiency_ratio:.4f}")
    print(f"oos_points={result.oos_equity_curve.size}")
    if result.efficiency_ratio > 0.5:
        print("edge_signal=pass")
    else:
        print("edge_signal=weak")


if __name__ == "__main__":
    main()
