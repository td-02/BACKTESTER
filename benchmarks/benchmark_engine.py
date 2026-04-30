from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import numpy as np

import nanoback as nb


MODE_CONFIG = {
    "latency": {"rows": 50_000, "cols": 8, "max_seconds": 0.50, "min_fills": 1_000, "regression_factor": 2.0},
    "stress": {"rows": 200_000, "cols": 16, "max_seconds": 2.50, "min_fills": 5_000, "regression_factor": 1.25},
}
PROFILE_THRESHOLDS = {
    "default": {"latency": 1.0, "stress": 1.0},
    "large": {"latency": 2.5, "stress": 3.0},
    "xlarge": {"latency": 5.0, "stress": 6.0},
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


def _estimate_matrix_memory_bytes(rows: int, cols: int) -> int:
    # Benchmark builds close/high/low/volume/bid/ask + targets/order/limit intermediates.
    float64_arrays = 9
    int64_arrays = 2
    int8_arrays = 1
    base = rows * cols
    raw = base * (float64_arrays * 8 + int64_arrays * 8 + int8_arrays * 1)
    # Account for transient allocations/copies during numpy transforms and C++ boundary handoff.
    return int(raw * 2.25)


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
    parser.add_argument("--regression-factor", type=float, default=None)
    parser.add_argument("--pnl-tolerance", type=float, default=1e-2)
    parser.add_argument("--history-file", type=str, default="benchmarks/benchmark_results_history.jsonl")
    parser.add_argument("--record-history", action="store_true")
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--profile", choices=sorted(PROFILE_THRESHOLDS.keys()), default="default")
    parser.add_argument("--memory-guard-gb", type=float, default=3.5)
    args = parser.parse_args()

    mode_defaults = MODE_CONFIG[args.mode]
    rows = int(args.rows if args.rows is not None else mode_defaults["rows"])
    cols = int(args.cols if args.cols is not None else mode_defaults["cols"])
    profile_mult = PROFILE_THRESHOLDS[args.profile][args.mode]
    default_max_seconds = mode_defaults["max_seconds"] * profile_mult
    max_seconds = float(args.max_seconds if args.max_seconds is not None else default_max_seconds)
    min_fills = int(args.min_fills if args.min_fills is not None else mode_defaults["min_fills"])
    regression_factor = float(
        args.regression_factor if args.regression_factor is not None else mode_defaults.get("regression_factor", 1.25)
    )
    resolved_version = _resolve_version(args.version)
    est_gb = _estimate_matrix_memory_bytes(rows, cols) / (1024**3)
    if est_gb > float(args.memory_guard_gb):
        raise SystemExit(
            f"benchmark shape too large for configured memory guard: "
            f"rows={rows} cols={cols} estimated={est_gb:.2f}GiB guard={args.memory_guard_gb:.2f}GiB"
        )

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
        # float32 generation reduces peak allocation pressure for large stress runs.
        close = 100.0 + np.cumsum(rng.normal(0.0, 0.1, size=(rows, cols)).astype(np.float32), axis=0)
        spread = np.abs(rng.normal(0.15, 0.03, size=(rows, cols)).astype(np.float32))
        high = close + spread
        low = close - spread
        volume = rng.integers(5_000, 20_000, size=(rows, cols)).astype(np.float32)
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
        "profile": args.profile,
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
        baseline_rows = int(baseline.get("rows", rows))
        baseline_cols = int(baseline.get("cols", cols))
        baseline_elapsed = float(baseline["elapsed_seconds"])
        baseline_fills = int(baseline["fills"])
        baseline_pnl = float(baseline["pnl"])
        if elapsed > baseline_elapsed * regression_factor:
            raise SystemExit(
                f"elapsed_seconds regression: current={elapsed:.6f} baseline={baseline_elapsed:.6f} "
                f"factor={regression_factor:.2f}"
            )
        comparable_shape = baseline_rows == rows and baseline_cols == cols
        if comparable_shape and len(result.fills) != baseline_fills:
            raise SystemExit(f"fill-count regression: current={len(result.fills)} baseline={baseline_fills}")
        if comparable_shape and abs(float(result.pnl) - baseline_pnl) > float(args.pnl_tolerance):
            raise SystemExit(
                f"pnl regression: current={float(result.pnl):.12f} baseline={baseline_pnl:.12f} "
                f"tolerance={args.pnl_tolerance:.12f}"
            )
        baseline_stages = {name: float(value) for name, value in baseline.get("stages", {}).items()}
        for stage, current in current_metrics["stages"].items():
            baseline_stage = baseline_stages.get(stage)
            if baseline_stage is None:
                continue
            if current > baseline_stage * regression_factor:
                raise SystemExit(
                    f"stage regression[{stage}]: current={current:.6f} baseline={baseline_stage:.6f} "
                    f"factor={regression_factor:.2f}"
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
