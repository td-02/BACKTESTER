from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from os import cpu_count
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .analytics import summarize_result


def _run_one(task: tuple[Callable[..., object], object, object | None, dict[str, Any]]) -> dict[str, Any]:
    strategy, data, oos_data, params = task
    result = strategy(data, **params)
    summary = summarize_result(result, symbols=getattr(data, "symbols", None))
    row: dict[str, Any] = dict(params)
    row.update(
        {
            "sharpe": float(summary.sharpe),
            "sortino": float(summary.sortino),
            "cagr": float(summary.cagr),
            "max_drawdown": float(summary.max_drawdown),
            "calmar": float(summary.calmar),
            "turnover_per_year": float(summary.turnover_per_year),
            "fill_count": int(summary.fill_count),
            "pnl": float(summary.pnl),
            "is_sharpe": float(summary.sharpe),
            "is_max_drawdown": float(summary.max_drawdown),
        }
    )
    if oos_data is not None:
        oos_result = strategy(oos_data, **params)
        oos_summary = summarize_result(oos_result, symbols=getattr(oos_data, "symbols", None))
        row["oos_sharpe"] = float(oos_summary.sharpe)
        row["oos_max_drawdown"] = float(oos_summary.max_drawdown)
        oos = float(oos_summary.sharpe)
        row["overfit_warning"] = bool(np.isfinite(oos) and oos > 0.0 and float(summary.sharpe) > 2.0 * oos)
    else:
        row["oos_sharpe"] = np.nan
        row["oos_max_drawdown"] = np.nan
        row["overfit_warning"] = False
    return row


@dataclass(slots=True)
class ParamGrid:
    grid: dict[str, Sequence[Any]]

    def combinations(self) -> list[dict[str, Any]]:
        if not self.grid:
            return [{}]
        keys = list(self.grid.keys())
        values = [list(self.grid[key]) for key in keys]
        return [dict(zip(keys, combo)) for combo in product(*values)]


@dataclass(slots=True)
class SweepResult:
    rows: list[dict[str, Any]]

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for SweepResult.to_dataframe(); install nanoback[io]") from exc
        return pd.DataFrame(self.rows)

    def sorted(self):
        if not self.rows:
            return SweepResult([])
        return SweepResult(sorted(self.rows, key=lambda row: float(row.get("sharpe", float("-inf"))), reverse=True))

    def best(self) -> dict[str, Any]:
        if not self.rows:
            raise ValueError("sweep result is empty")
        return self.sorted().rows[0]

    def heatmap(self, x_param: str, y_param: str, metric: str = "sharpe"):
        try:
            import plotly.express as px
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("plotly is required for heatmap(); install plotly") from exc

        frame = self.to_dataframe()
        pivot = frame.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc="mean")
        fig = px.imshow(
            pivot.values,
            x=[str(v) for v in pivot.columns.tolist()],
            y=[str(v) for v in pivot.index.tolist()],
            labels={"x": x_param, "y": y_param, "color": metric},
            title=f"{metric} heatmap ({x_param} x {y_param})",
            aspect="auto",
        )
        return fig


class Sweep:
    def __init__(self, data: object):
        self.data = data

    def run(
        self,
        strategy: Callable[..., object],
        param_grid: ParamGrid | dict[str, Sequence[Any]],
        *,
        n_jobs: int = -1,
        compiled: bool = False,
        oos_data: object | None = None,
    ) -> SweepResult:
        grid = param_grid if isinstance(param_grid, ParamGrid) else ParamGrid(dict(param_grid))
        combos = grid.combinations()
        if not combos:
            return SweepResult([])

        # Compiled policies already spend runtime in C++; keep single-process to avoid spawn overhead.
        effective_jobs = 1 if compiled else (cpu_count() or 1 if n_jobs == -1 else max(1, int(n_jobs)))
        tasks = [(strategy, self.data, oos_data, combo) for combo in combos]
        if effective_jobs == 1:
            rows = [_run_one(task) for task in tasks]
        else:
            try:
                with ProcessPoolExecutor(max_workers=effective_jobs) as pool:
                    rows = list(pool.map(_run_one, tasks))
            except Exception:
                # Fallback for non-picklable callables/closures in research notebooks.
                rows = [_run_one(task) for task in tasks]
        return SweepResult(rows).sorted()
