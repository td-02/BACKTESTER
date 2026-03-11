# nanoback

`nanoback` is an event-driven trading backtester with a C++20 execution core and a Python package wrapper.

## Implemented

- Multi-asset matrix backtests
- Market and limit order simulation
- Partial fills using a max participation rate
- Latency steps and configurable slippage/commission
- Session gating through a tradable mask or `SessionCalendar`
- Python strategy/plugin interface
- CSV and Parquet loaders
- Benchmark script and pytest coverage

## Layout

- `include/nanoback`: C++ headers
- `cpp`: C++ engine and bindings
- `python/nanoback`: Python API, loaders, strategies, calendars
- `examples`: runnable examples
- `benchmarks`: micro-benchmark scripts
- `tests`: functional coverage

## Quickstart

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .[dev]
pytest
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
pytest
```

## Single-asset wrapper

```python
import numpy as np
import nanoback as nb

result = nb.run_backtest(
    timestamps=np.array([1, 2, 3, 4], dtype=np.int64),
    prices=np.array([100.0, 101.0, 103.0, 102.0], dtype=np.float64),
    signals=np.array([1, 1, -1, 0], dtype=np.int64),
    config=nb.BacktestConfig(max_position=2),
)

print(result.pnl)
```

## Strategy API

```python
import nanoback as nb

class BuyFirstAsset(nb.Strategy):
    def on_event(self, event):
        if event.index == 0:
            return [nb.OrderIntent(asset=0, target_position=2)]
        return ()
```

Use `nb.run_strategy_backtest(...)` to transform strategy output into target matrices and execute in the C++ engine.
