from __future__ import annotations

import argparse

import numpy as np

import nanoback as nb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seconds", type=float, default=0.25)
    parser.add_argument("--min-fills", type=int, default=1_000)
    parser.add_argument("--log-book", type=str, default="outputs/benchmark_engine_latency.jsonl")
    args = parser.parse_args()

    log_book = nb.LatencyLogBook(scenario="benchmark_engine", seed=42)
    rows = 50_000
    cols = 8
    rng = np.random.default_rng(log_book.seed)
    with log_book.timing("data_generation", rows=rows, cols=cols):
        timestamps = np.arange(rows, dtype=np.int64) * 60 + 34_200
        close = 100.0 + np.cumsum(rng.normal(0.0, 0.1, size=(rows, cols)), axis=0)
        spread = np.abs(rng.normal(0.15, 0.03, size=(rows, cols)))
        high = close + spread
        low = close - spread
        volume = rng.integers(5_000, 20_000, size=(rows, cols)).astype(np.float64)
        bid = close - spread * 0.5
        ask = close + spread * 0.5
        data = nb.MarketData(
            timestamps=timestamps,
            close=close,
            high=high,
            low=low,
            volume=volume,
            bid=bid,
            ask=ask,
            symbols=[f"asset_{idx}" for idx in range(cols)],
            asset_configs=[nb.AssetConfig(symbol=f"asset_{idx}", max_position=5, notional_limit=5_000_000.0) for idx in range(cols)],
        )

    config = nb.BacktestConfig(
        starting_cash=10_000_000.0,
        commission_bps=0.1,
        slippage_bps=0.2,
        max_position=5,
        max_participation_rate=0.2,
        latency_steps=1,
        child_order_size=2,
        child_slice_delay_steps=1,
        use_bid_ask_execution=True,
    )

    with log_book.timing("policy_generation", policy="moving_average_crossover"):
        targets = nb.compiled_moving_average_crossover_targets(
            data.close,
            fast_window=8,
            slow_window=32,
            max_position=5,
        )

    with log_book.timing("engine_run"):
        result = nb.run_backtest_matrix(
            timestamps=data.timestamps,
            close=data.close,
            high=data.high,
            low=data.low,
            volume=data.volume,
            bid=data.bid,
            ask=data.ask,
            target_positions=targets,
            order_types=np.full_like(targets, int(nb.OrderType.MARKET), dtype=np.int8),
            limit_prices=np.full_like(targets, np.nan, dtype=np.float64),
            tradable_mask=None,
            asset_max_positions=data.asset_max_positions,
            asset_notional_limits=data.asset_notional_limits,
            config=config,
            symbols=data.symbols,
        )

    log_book.metadata.update({"fills": int(len(result.fills)), "pnl": float(result.pnl)})
    elapsed = log_book.total_seconds()

    print(f"elapsed_seconds={elapsed:.4f}")
    print(f"rows={rows} cols={cols}")
    print(f"fills={len(result.fills)} pnl={result.pnl:.2f}")
    print(log_book.render_text())
    log_book.write_jsonl(args.log_book)

    if elapsed > args.max_seconds:
        raise SystemExit(f"benchmark exceeded threshold: {elapsed:.4f}s > {args.max_seconds:.4f}s")
    if len(result.fills) < args.min_fills:
        raise SystemExit(f"benchmark produced too few fills: {len(result.fills)} < {args.min_fills}")


if __name__ == "__main__":
    main()
