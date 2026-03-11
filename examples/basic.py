import numpy as np

import nanoback as nb


def main() -> None:
    timestamps = np.array([34_200, 34_260, 34_320, 34_380, 34_440], dtype=np.int64)
    close = np.array(
        [
            [100.0, 50.0],
            [101.0, 49.5],
            [102.0, 49.0],
            [101.5, 50.5],
            [103.0, 51.0],
        ],
        dtype=np.float64,
    )
    high = close + 0.25
    low = close - 0.25
    volume = np.full_like(close, 1_000.0)
    data = nb.MarketData(
        timestamps=timestamps,
        close=close,
        high=high,
        low=low,
        volume=volume,
        symbols=["AAPL", "MSFT"],
    )

    result = nb.run_compiled_policy_backtest(
        data=data,
        policy="momentum",
        lookback=1,
        max_position=3,
        config=nb.BacktestConfig(
            starting_cash=500_000.0,
            commission_bps=0.5,
            slippage_bps=0.25,
            max_position=3,
            latency_steps=1,
        ),
    )

    print(f"PnL: {result.pnl:.2f}")
    print(f"Orders submitted: {result.submitted_orders}")
    print(f"Fills: {len(result.fills)}")
    print(f"Ending equity: {result.ending_equity:.2f}")


if __name__ == "__main__":
    main()
