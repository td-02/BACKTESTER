from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import numpy as np

import nanoback as nb


MODE_CONFIG = {
    "latency": {"rows": 50_000, "cols": 8, "max_seconds": 0.50, "min_fills": 1_000},
    "stress": {"rows": 200_000, "cols": 16, "max_seconds": 2.50, "min_fills": 5_000},
}


def _resolve_version(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        return package_version("nanoback")
    except PackageNotFoundError:
        return "unknown"


def _append_history(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODE_CONFIG.keys()), default="latency")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--min-fills", type=int, default=None)
    parser.add_argument("--log-book", type=str, default=None)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--regression-factor", type=float, default=1.25)
    parser.add_argument("--history-file", type=str, default="benchmarks/benchmark_results_history.jsonl")
    parser.add_argument("--record-history", action="store_true")
    parser.add_argument("--version", type=str, default=None)
    args = parser.parse_args()

    mode_defaults = MODE_CONFIG[args.mode]
    rows = int(args.rows if args.rows is not None else mode_defaults["rows"])
    cols = int(args.cols if args.cols is not None else mode_defaults["cols"])
    max_seconds = float(args.max_seconds if args.max_seconds is not None else mode_defaults["max_seconds"])
    min_fills = int(args.min_fills if args.min_fills is not None else mode_defaults["min_fills"])
    resolved_version = _resolve_version(args.version)

    log_book_path = args.log_book or f"outputs/benchmark_engine_{args.mode}.jsonl"
    baseline_default = (
        "benchmarks/benchmark_engine_baseline.json"
        if args.mode == "latency"
        else "benchmarks/benchmark_engine_stress_baseline.json"
    )
    baseline_path = Path(args.baseline or baseline_default)

    log_book = nb.LatencyLogBook(scenario=f"benchmark_engine_{args.mode}", seed=42)
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
    log_book.write_jsonl(log_book_path)
    current_metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "version": resolved_version,
        "mode": args.mode,
        "rows": rows,
        "cols": cols,
        "elapsed_seconds": elapsed,
        "fills": int(len(result.fills)),
        "pnl": float(result.pnl),
        "stages": {stage: summary.total for stage, summary in log_book.stage_summaries().items()},
    }
    if args.update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(current_metrics, indent=2), encoding="utf-8")
        print(f"updated_baseline={baseline_path}")
    elif baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_elapsed = float(baseline["elapsed_seconds"])
        baseline_fills = int(baseline["fills"])
        baseline_pnl = float(baseline["pnl"])
        if elapsed > baseline_elapsed * args.regression_factor:
            raise SystemExit(
                f"elapsed_seconds regression: current={elapsed:.6f} baseline={baseline_elapsed:.6f} "
                f"factor={args.regression_factor:.2f}"
            )
        if len(result.fills) != baseline_fills:
            raise SystemExit(f"fill-count regression: current={len(result.fills)} baseline={baseline_fills}")
        if abs(float(result.pnl) - baseline_pnl) > 1e-9:
            raise SystemExit(f"pnl regression: current={float(result.pnl):.12f} baseline={baseline_pnl:.12f}")
        baseline_stages = {name: float(value) for name, value in baseline.get("stages", {}).items()}
        for stage, current in current_metrics["stages"].items():
            baseline_stage = baseline_stages.get(stage)
            if baseline_stage is None:
                continue
            if current > baseline_stage * args.regression_factor:
                raise SystemExit(
                    f"stage regression[{stage}]: current={current:.6f} baseline={baseline_stage:.6f} "
                    f"factor={args.regression_factor:.2f}"
                )
        print(f"baseline_check=passed path={baseline_path}")

    if elapsed > max_seconds:
        raise SystemExit(f"benchmark exceeded threshold: {elapsed:.4f}s > {max_seconds:.4f}s")
    if len(result.fills) < min_fills:
        raise SystemExit(f"benchmark produced too few fills: {len(result.fills)} < {min_fills}")
    if args.record_history:
        history_payload = dict(current_metrics)
        history_payload["log_book"] = str(log_book_path)
        _append_history(Path(args.history_file), history_payload)
        print(f"history_recorded={args.history_file}")


if __name__ == "__main__":
    main()
