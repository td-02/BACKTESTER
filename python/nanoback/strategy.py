from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ._nanoback import (
    BacktestConfig,
    OrderType,
    cross_sectional_momentum_targets as _cross_sectional_momentum_targets,
    cross_sectional_rank as _cross_sectional_rank,
    mean_reversion_targets as _mean_reversion_targets,
    minimum_variance_weights as _minimum_variance_weights,
    momentum_targets as _momentum_targets,
    moving_average_crossover_targets as _moving_average_crossover_targets,
    rolling_volatility as _rolling_volatility,
    volatility_filtered_momentum_targets as _volatility_filtered_momentum_targets,
)
from .data import MarketData
from .wrapper import run_backtest_matrix


@dataclass(slots=True)
class MarketEvent:
    index: int
    timestamp: int
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    symbols: Sequence[str]


@dataclass(slots=True)
class OrderIntent:
    asset: int
    target_position: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float = math.nan


class Strategy:
    def on_start(self, data: MarketData) -> None:
        return None

    def on_event(self, event: MarketEvent) -> Iterable[OrderIntent]:
        return ()


def load_strategy(
    path: str,
    *,
    allow_untrusted: bool = False,
    allowed_prefixes: Sequence[str] = ("tests.", "nanoback.", "strategies."),
    **kwargs: object,
) -> Strategy:
    if ":" not in path:
        raise ValueError("strategy path must use 'module:ClassName'")
    module_name, class_name = path.split(":", maxsplit=1)
    if not allow_untrusted and not any(module_name.startswith(prefix) for prefix in allowed_prefixes):
        raise ValueError(
            "refusing to import untrusted strategy module. "
            "Pass allow_untrusted=True to bypass this safeguard."
        )
    module = importlib.import_module(module_name)
    strategy_cls = getattr(module, class_name)
    instance = strategy_cls(**kwargs)
    if not isinstance(instance, Strategy):
        raise TypeError(f"{path} does not resolve to a nanoback.strategy.Strategy subclass")
    return instance


def generate_target_matrices(
    data: MarketData,
    strategy: Strategy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_count, asset_count = data.shape
    targets = np.zeros((row_count, asset_count), dtype=np.int64)
    order_types = np.full((row_count, asset_count), int(OrderType.MARKET), dtype=np.int8)
    limit_prices = np.full((row_count, asset_count), np.nan, dtype=np.float64)

    strategy.on_start(data)
    current_targets = np.zeros(asset_count, dtype=np.int64)
    current_types = np.full(asset_count, int(OrderType.MARKET), dtype=np.int8)
    current_limits = np.full(asset_count, np.nan, dtype=np.float64)

    for idx in range(row_count):
        event = MarketEvent(
            index=idx,
            timestamp=int(data.timestamps[idx]),
            close=data.close[idx],
            high=data.high[idx],
            low=data.low[idx],
            volume=data.volume[idx],
            symbols=data.symbols,
        )
        for intent in strategy.on_event(event):
            current_targets[intent.asset] = int(intent.target_position)
            current_types[intent.asset] = int(intent.order_type)
            current_limits[intent.asset] = float(intent.limit_price)

        targets[idx] = current_targets
        order_types[idx] = current_types
        limit_prices[idx] = current_limits

    return targets, order_types, limit_prices


def run_strategy_backtest(
    data: MarketData,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    tradable_mask: np.ndarray | None = None,
):
    targets, order_types, limit_prices = generate_target_matrices(data, strategy)
    return run_backtest_matrix(
        timestamps=data.timestamps,
        close=data.close,
        high=data.high,
        low=data.low,
        volume=data.volume,
        bid=data.bid,
        ask=data.ask,
        target_positions=targets,
        order_types=order_types,
        limit_prices=limit_prices,
        tradable_mask=tradable_mask,
        asset_max_positions=data.asset_max_positions,
        asset_notional_limits=data.asset_notional_limits,
        config=config,
        symbols=data.symbols,
    )


def compiled_momentum_targets(
    close: np.ndarray,
    *,
    lookback: int,
    max_position: int,
) -> np.ndarray:
    return np.asarray(
        _momentum_targets(np.ascontiguousarray(close, dtype=np.float64), lookback, max_position),
        dtype=np.int64,
    )


def compiled_rolling_volatility(
    close: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    return np.asarray(
        _rolling_volatility(np.ascontiguousarray(close, dtype=np.float64), window),
        dtype=np.float64,
    )


def compiled_cross_sectional_rank(
    values: np.ndarray,
    *,
    descending: bool = True,
) -> np.ndarray:
    return np.asarray(
        _cross_sectional_rank(np.ascontiguousarray(values, dtype=np.float64), descending),
        dtype=np.int64,
    )


def compiled_minimum_variance_weights(
    close: np.ndarray,
    *,
    window: int,
    ridge: float = 1e-6,
    leverage: float = 1.0,
) -> np.ndarray:
    return np.asarray(
        _minimum_variance_weights(
            np.ascontiguousarray(close, dtype=np.float64),
            window,
            ridge,
            leverage,
        ),
        dtype=np.float64,
    )


def compiled_mean_reversion_targets(
    close: np.ndarray,
    *,
    lookback: int,
    max_position: int,
) -> np.ndarray:
    return np.asarray(
        _mean_reversion_targets(np.ascontiguousarray(close, dtype=np.float64), lookback, max_position),
        dtype=np.int64,
    )


def compiled_volatility_filtered_momentum_targets(
    close: np.ndarray,
    *,
    lookback: int,
    vol_window: int,
    volatility_ceiling: float,
    max_position: int,
) -> np.ndarray:
    return np.asarray(
        _volatility_filtered_momentum_targets(
            np.ascontiguousarray(close, dtype=np.float64),
            lookback,
            vol_window,
            volatility_ceiling,
            max_position,
        ),
        dtype=np.int64,
    )


def compiled_moving_average_crossover_targets(
    close: np.ndarray,
    *,
    fast_window: int,
    slow_window: int,
    max_position: int,
) -> np.ndarray:
    return np.asarray(
        _moving_average_crossover_targets(
            np.ascontiguousarray(close, dtype=np.float64),
            fast_window,
            slow_window,
            max_position,
        ),
        dtype=np.int64,
    )


def compiled_cross_sectional_momentum_targets(
    close: np.ndarray,
    *,
    lookback: int,
    winners: int,
    losers: int,
    max_position: int,
) -> np.ndarray:
    return np.asarray(
        _cross_sectional_momentum_targets(
            np.ascontiguousarray(close, dtype=np.float64),
            lookback,
            winners,
            losers,
            max_position,
        ),
        dtype=np.int64,
    )


def weights_to_positions(
    weights: np.ndarray,
    *,
    gross_target: int,
) -> np.ndarray:
    scaled = np.rint(np.asarray(weights, dtype=np.float64) * gross_target)
    return scaled.astype(np.int64, copy=False)


def run_compiled_policy_backtest(
    data: MarketData,
    *,
    policy: str,
    config: BacktestConfig | None = None,
    tradable_mask: np.ndarray | None = None,
    order_types: np.ndarray | None = None,
    limit_prices: np.ndarray | None = None,
    **policy_kwargs: int,
):
    if policy == "momentum":
        targets = compiled_momentum_targets(data.close, **policy_kwargs)
    elif policy == "mean_reversion":
        targets = compiled_mean_reversion_targets(data.close, **policy_kwargs)
    elif policy == "moving_average_crossover":
        targets = compiled_moving_average_crossover_targets(data.close, **policy_kwargs)
    elif policy == "volatility_filtered_momentum":
        targets = compiled_volatility_filtered_momentum_targets(data.close, **policy_kwargs)
    elif policy == "cross_sectional_momentum":
        targets = compiled_cross_sectional_momentum_targets(data.close, **policy_kwargs)
    elif policy == "minimum_variance":
        gross_target = int(policy_kwargs.pop("gross_target", (config or BacktestConfig()).max_position))
        weights = compiled_minimum_variance_weights(data.close, **policy_kwargs)
        targets = weights_to_positions(weights, gross_target=gross_target)
    else:
        raise ValueError(f"unknown compiled policy: {policy}")

    return run_backtest_matrix(
        timestamps=data.timestamps,
        close=data.close,
        high=data.high,
        low=data.low,
        volume=data.volume,
        bid=data.bid,
        ask=data.ask,
        target_positions=targets,
        order_types=order_types,
        limit_prices=limit_prices,
        tradable_mask=tradable_mask,
        asset_max_positions=data.asset_max_positions,
        asset_notional_limits=data.asset_notional_limits,
        config=config,
        symbols=data.symbols,
    )
