# BACKTESTER (`nanoback`)

`BACKTESTER` is a C++20 event-driven multi-asset backtesting engine with Python bindings, packaged as `nanoback`.
It focuses on realistic execution simulation, data correctness, statistical validity, and backtest-to-paper continuity.

## Highlights (v0.6.0)

- Fast C++ core with Python APIs
- Bar-mode and tick-mode simulation paths
- Corporate actions support: split, dividend, spinoff, delisting
- Smart execution realism primitives:
  - multi-venue routing model (fees, volume share, fill curves)
  - signal/order/fill latency modeling + adverse selection penalties
- Derivatives primitives:
  - instrument model for equity, options, futures, FX forwards
  - option expiry settlement, futures roll events, margin liquidation path
- Research validity stack:
  - analytics (Sharpe, Sortino, CAGR, drawdown, attribution)
  - parameter sweeps and heatmaps
  - walk-forward optimization
  - Monte Carlo shuffle/block-bootstrap stress tests
- Live bridge (new in v0.6):
  - `PaperBroker` streams ticks and runs the same engine/risk/ledger path in realtime
  - feed adapter protocol for Alpaca/yfinance/Binance integrations
  - reconciliation hooks that can run after fill events
- Position reconciliation (new in v0.6):
  - `Reconciler` diffs engine vs broker positions
  - optional auto-reconcile corrective orders
  - JSONL reconciliation log for auditability

## Repository Layout

- `include/nanoback`: C++ public headers
- `cpp`: core engine and Python bindings
- `python/nanoback`: Python API, loaders, analytics, research modules
- `benchmarks`: performance and regression checks
- `examples`: runnable usage examples
- `tests`: regression and functional test suite

## Quickstart

```powershell
cd C:\Users\TAPESH\Documents\BACKTESTEER
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
python -m pytest
```

## Core Usage

### Basic Bar Backtest

```python
import numpy as np
import nanoback as nb

result = nb.run_backtest(
    timestamps=np.array([1, 2, 3, 4], dtype=np.int64),
    prices=np.array([100.0, 101.0, 99.0, 102.0], dtype=np.float64),
    signals=np.array([1, 1, 0, -1], dtype=np.int64),
    config=nb.BacktestConfig(max_position=2),
)

print(result.summary())
```

### Paper Trading Bridge

```python
from datetime import datetime, timedelta, timezone
import nanoback as nb
from nanoback.paper import PaperBroker, PaperTick

class DemoFeed:
    def __init__(self, ticks):
        self._ticks = list(ticks)
    def next_tick(self, timeout_seconds=None):
        return self._ticks.pop(0) if self._ticks else None
    def fetch_positions(self):
        return {"AAA": 0.0}
    def submit_order(self, symbol, quantity_delta):
        pass

def strategy(tick, state):
    return {"AAA": 1}

broker = PaperBroker(
    symbols=["AAA"],
    strategy=strategy,
    feed=DemoFeed([PaperTick(timestamp_ns=1, symbol="AAA", price=100.0)]),
)
broker.run_until(datetime.now(timezone.utc) + timedelta(seconds=1))
```

### Position Reconciliation

```python
from nanoback.reconcile import Reconciler

reconciler = Reconciler(adapter=broker.feed, log_path="outputs/reconcile.jsonl", auto_reconcile=False)
records = reconciler.reconcile(broker.positions)
```

## Data Loaders

- `load_csv`, `load_parquet` for bar data
- `load_ticks_parquet` for tick event replay
- `load_corporate_actions_csv` for corporate action ingestion
- `load_yahoo_adjusted` for adjusted prices + suspicious jump warning checks

## Benchmarks

Latency benchmark (CI-style):

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_engine.py --max-seconds 0.50 --min-fills 1000 --ci-mode
```

Stress benchmark (large shapes):

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_engine.py --mode stress --profile xlarge
```
