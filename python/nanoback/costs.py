from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ._nanoback import BacktestConfig


@dataclass(slots=True)
class CostCalibration:
    fixed_slippage_bps: float
    volume_share_impact: float
    spread_bps: float
    samples: int

    def to_config(self, base: BacktestConfig | None = None) -> BacktestConfig:
        config = base or BacktestConfig()
        config.slippage_bps = self.fixed_slippage_bps
        config.volume_share_impact = self.volume_share_impact
        return config


def _fit_impact(impact_bps: np.ndarray, participation: np.ndarray) -> tuple[float, float]:
    x = np.square(np.asarray(participation, dtype=np.float64))
    y = np.abs(np.asarray(impact_bps, dtype=np.float64))
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = max(0.0, float(beta[0]))
    slope = max(0.0, float(beta[1]) / 10_000.0)
    return intercept, slope


def calibrate_cost_model(
    *,
    arrival_mid: Iterable[float],
    fill_price: Iterable[float],
    side: Iterable[int],
    participation: Iterable[float],
    quoted_spread_bps: Iterable[float] | None = None,
) -> CostCalibration:
    arrival_mid = np.asarray(list(arrival_mid), dtype=np.float64)
    fill_price = np.asarray(list(fill_price), dtype=np.float64)
    side = np.asarray(list(side), dtype=np.float64)
    participation = np.asarray(list(participation), dtype=np.float64)

    if not (arrival_mid.size == fill_price.size == side.size == participation.size):
        raise ValueError("all calibration vectors must have the same length")

    impact_bps = side * ((fill_price / arrival_mid) - 1.0) * 10_000.0
    fixed_slippage_bps, volume_share_impact = _fit_impact(impact_bps, participation)
    if quoted_spread_bps is None:
        spread_bps = float(np.median(np.abs(impact_bps)) * 2.0)
    else:
        spread_bps = float(np.median(np.asarray(list(quoted_spread_bps), dtype=np.float64)))

    return CostCalibration(
        fixed_slippage_bps=fixed_slippage_bps,
        volume_share_impact=volume_share_impact,
        spread_bps=spread_bps,
        samples=int(arrival_mid.size),
    )


def calibrate_cost_model_from_csv(path: str | Path) -> CostCalibration:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"arrival_mid", "fill_price", "side", "participation"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows.extend(reader)

    quoted_spread = None
    if rows and "quoted_spread_bps" in rows[0]:
        quoted_spread = [float(row["quoted_spread_bps"]) for row in rows]

    return calibrate_cost_model(
        arrival_mid=[float(row["arrival_mid"]) for row in rows],
        fill_price=[float(row["fill_price"]) for row in rows],
        side=[int(row["side"]) for row in rows],
        participation=[float(row["participation"]) for row in rows],
        quoted_spread_bps=quoted_spread,
    )
