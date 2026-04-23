# BACKTESTER

`BACKTESTER` is a low-latency, event-driven trading backtesting repo built around a C++20 execution engine and exposed through a Python package named `nanoback`.

## What It Does

- Multi-asset backtests over contiguous matrix inputs
- Market and limit order simulation
- Partial fills, queue blocking, venue caps, and bid/ask-aware execution
- Parent/child order IDs with deterministic audit ledger output
- Snapshot/resume support for long-running simulations
- Risk controls for leverage, drawdown, cash, and per-asset limits
- Financing and borrow-cost accrual
- Compiled research policies and analytics in C++
- Python strategy/plugin fallback
- CSV and Parquet loaders
- Asserted benchmarks and pytest coverage

## Repo Layout

- `include/nanoback`: C++ headers
- `cpp`: C++ engine, policies, and Python bindings
- `python/nanoback`: Python API, loaders, ledger utilities, and strategy helpers
- `examples`: runnable examples
- `benchmarks`: performance checks
- `tests`: regression and functional coverage

## Quickstart

```powershell
cd C:\Users\TAPESH\documents\BACKTESTEER
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
python -m pytest
```

## Core Capabilities

### Execution Engine

- Event-driven C++20 core
- Market and limit orders
- Child-order slicing with delay steps
- Cancel/replace semantics
- Session-aware order cancellation
- Snapshot/resume from engine state

### Market Realism

- Bid/ask execution path
- Queue-ahead fraction and venue volume share caps
- Slippage and participation-based impact
- Cost calibration helpers from empirical fill data

### Risk and Audit

- Gross leverage checks
- Drawdown kill switch
- Cash and borrow accounting
- Per-asset max position and notional limits
- Deterministic ledger export to CSV/JSONL

### Research Layer

- Compiled momentum, mean reversion, and moving-average crossover policies
- Compiled rolling volatility, cross-sectional ranking, and minimum-variance weights
- Python strategy hooks when custom event logic is needed

## Minimal Example

```python
import numpy as np
import nanoback as nb

result = nb.run_backtest(
    timestamps=np.array([1, 2, 3, 4], dtype=np.int64),
    prices=np.array([100.0, 101.0, 99.0, 102.0], dtype=np.float64),
    signals=np.array([1, 1, 0, -1], dtype=np.int64),
    config=nb.BacktestConfig(
        max_position=2,
        child_order_size=1,
        child_slice_delay_steps=1,
    ),
)

print(result.pnl)
print(len(result.ledger))
```

## Benchmark

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_engine.py --max-seconds 0.50 --min-fills 1000
```

The benchmark now writes a latency log book to `outputs/benchmark_engine_latency.jsonl` and checks
its results against `benchmarks/benchmark_engine_baseline.json`.

Useful flags:

- `--log-book` to change the JSONL output path
- `--baseline` to compare against a different baseline file
- `--update-baseline` to refresh the baseline after an intentional change
- `--regression-factor` to control how much slower a stage may get before failing

## Performance Notes

Measured on this repo with the current `0.2.0` code:

- Standard benchmark, `50_000 x 8`
  - `elapsed_seconds=0.0609`
  - `data_generation=0.0324s`
  - `policy_generation=0.0040s`
  - `engine_run=0.0246s`
  - `fills=61033`
- Heavier simulation, `200_000 x 16`
  - Full pipeline: `elapsed_seconds=0.5472`
  - `data_generation=0.2933s`
  - `policy_generation=0.0762s`
  - `engine_run=0.1777s`
- Isolated engine path for the same heavy simulation
  - `elapsed_seconds=0.2233`
  - `policy_generation=0.0733s`
  - `engine_run=0.1499s`

The isolated run shows the core backtest engine is fast and scales well, while the full pipeline is
more constrained by Python-side data setup.

## Institutional Reporting

Use the reporting helpers to generate a performance package for a completed run:

```python
import nanoback as nb

result = nb.run_backtest(...)
report = nb.summarize_backtest(result, symbols=["AAA"])
nb.export_performance_report_json(report, "outputs/performance_report.json")
nb.export_performance_report_markdown(report, "outputs/performance_report.md")
```

The report includes:

- core PnL, return, drawdown, turnover, and fill-rate metrics
- equity and cash curve summaries
- audit-event counts
- asset-level execution summaries

This is the layer you would use for institutional-style post-trade review and fast root-cause analysis.

## Status

The repo is strong as a research and simulation engine. It is not a full OMS/EMS, exchange adapter stack, or compliance platform.
