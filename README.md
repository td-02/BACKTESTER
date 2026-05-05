# BACKTESTER (`nanoback`)

`BACKTESTER` is a C++20 event-driven multi-asset backtesting engine with Python bindings, packaged as `nanoback`.
It focuses on realistic execution simulation, data correctness, and statistically valid research workflows.

## Highlights (v0.5.x)

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
- Deterministic audit ledger and snapshot/resume support

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

### Tick Replay

```python
import nanoback as nb

config = nb.BacktestConfig()
config.data_mode = nb.DataMode.TICK

result = nb.run_backtest_ticks(
    timestamp_ns=timestamp_ns,
    asset=asset_idx,
    price=price,
    size=size,
    side=side,  # TickSide.BID / ASK / TRADE
    target_positions=targets,
    cols=n_assets,
    config=config,
    symbols=symbols,
)
```

### Corporate Actions

```python
import nanoback as nb

config = nb.BacktestConfig()
config.corporate_actions = nb.load_corporate_actions_csv("corp_actions.csv", symbol_to_asset)
```

### Statistical Validation

```python
import nanoback as nb

sweep = nb.Sweep(data)
grid = nb.ParamGrid({"lookback": [5, 10, 20], "max_position": [1, 2]})
res = sweep.run(strategy, grid, n_jobs=1, compiled=True)

wfo = nb.WalkForward(n_splits=6, train_frac=0.7, anchored=True)
wfo_res = wfo.run(data, strategy, grid, compiled=True)

mc = nb.MonteCarlo.from_backtest(result)
mc_res = mc.run(n_sims=5000, method="block_bootstrap", block_size=20)
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

Useful benchmark flags:

- `--baseline` / `--update-baseline`
- `--regression-factor`
- `--stage-regression-factor`
- `--pnl-tolerance`
- `--fill-count-tolerance`
- `--memory-guard-gb`
- `--record-history --history-file ...`

## Notes

- This is a research and simulation platform, not a production OMS/EMS.
- Performance and determinism are prioritized, but you should still validate strategy-specific assumptions before live deployment.
