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

## Status

The repo is strong as a research and simulation engine. It is not a full OMS/EMS, exchange adapter stack, or compliance platform.
