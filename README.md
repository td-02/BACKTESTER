# BACKTESTER (`nanoback`)

`nanoback` is a C++20 event-driven backtesting engine with Python APIs for research, paper trading, and production-grade validation workflows.

## What's New (v1.0)

- Live bridge (`v0.6`): `PaperBroker` streams real-time ticks into the same engine path as backtests
- Safety reconciliation: `Reconciler` detects and optionally auto-corrects engine-vs-broker position drift
- Interactive reporting:
  - `result.plot()` equity + underwater drawdown
  - `result.to_html("report.html")` self-contained shareable report
  - `result.dashboard()` Streamlit analytics panel
  - `sweep_result.heatmap(...)` Plotly parameter robustness diagnostics
  - `wfo_result.plot()` IS vs OOS Sharpe + OOS equity
- Release quality:
  - Multi-platform wheel publishing through `cibuildwheel`
  - Benchmark regression guard on CI merges to `main`
  - Expanded edge-case regression suite

## Performance Snapshot

Max stress run measured in this workspace:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_engine.py --mode stress --rows 1080000 --cols 16 --max-seconds 180 --min-fills 10000 --baseline benchmarks\benchmark_engine_stress_baseline.json --log-book outputs\max_stress_latency_1080000.jsonl
```

- `elapsed_seconds=25.2673`
- `data_generation=1.2495s`
- `policy_generation=0.4433s`
- `engine_run=23.5745s`
- `fills=2670270`

This run is the closest practical ceiling I measured here. The core engine remains the dominant cost at this size, which is the part that matters for throughput tuning.

## Hot Path Profile

The same `200000 x 16` stress shape was rerun with targeted engine variants to isolate the most expensive execution features:

```powershell
.\.venv\Scripts\python.exe benchmarks\benchmark_engine.py --mode stress --rows 200000 --cols 16 --hot-path-profile --max-seconds 120 --min-fills 5000 --log-book outputs\hot_path_profile.jsonl
```

- `elapsed_seconds=3.0829`
- `data_generation=0.2816s`
- `policy_generation=0.0782s`
- `engine_run[baseline]=0.5596s`
- `engine_run[no_latency]=0.5653s` `speedup_vs_baseline=0.99x`
- `engine_run[no_bid_ask]=0.5845s` `speedup_vs_baseline=0.96x`
- `engine_run[no_child_slice]=0.4668s` `speedup_vs_baseline=1.20x`
- `engine_run[fast_path]=0.5470s` `speedup_vs_baseline=1.02x`
- `fills=493502`

Interpretation:
- The child-slicing path is the clearest optimization target in this run.
- Latency-step logic and bid/ask execution are not the dominant cost at this workload.
- The engine is still the bottleneck, but the profile now points to a narrower hot path instead of a broad slowdown.

## Install

```bash
pip install nanoback
```

Optional extras:

```bash
pip install "nanoback[io,viz]"
```

## Core Example

```python
import numpy as np
import nanoback as nb

result = nb.run_backtest(
    timestamps=np.array([1, 2, 3, 4], dtype=np.int64),
    prices=np.array([100.0, 101.0, 99.0, 102.0], dtype=np.float64),
    signals=np.array([1, 1, 0, -1], dtype=np.int64),
    config=nb.BacktestConfig(max_position=2),
)

print(result)
fig = result.plot()
result.to_html("report.html")
```

## Paper Trading Bridge (`v0.6`)

```python
from datetime import datetime, timedelta, timezone
from nanoback.paper import PaperBroker, PaperTick
import nanoback as nb

class Feed:
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
    feed=Feed([PaperTick(timestamp_ns=1, symbol="AAA", price=100.0)]),
    config=nb.BacktestConfig(max_position=2),
)
broker.run_until(datetime.now(timezone.utc) + timedelta(seconds=1))
```

## Repo Layout

- `cpp/`, `include/nanoback/`: core engine and bindings
- `python/nanoback/`: analytics, live bridge, strategy, sweep/WFO/MC modules
- `benchmarks/`: latency/stress harnesses and baselines
- `tests/`: regression suite

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
python -m pytest
```
