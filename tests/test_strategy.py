from __future__ import annotations

import numpy as np

import nanoback as nb


def test_session_calendar_masks_outside_hours() -> None:
    calendar = nb.SessionCalendar(start_minute=9 * 60 + 30, end_minute=9 * 60 + 31)
    timestamps = np.array([34_140, 34_200, 34_260, 34_320], dtype=np.int64)
    mask = calendar.tradable_mask(timestamps)
    assert mask.tolist() == [0, 1, 1, 0]


def test_strategy_plugin_loader_and_runner() -> None:
    strategy = nb.load_strategy("tests.sample_strategy:FlipStrategy")
    data = nb.MarketData(
        timestamps=np.array([34_200, 34_260, 34_320, 34_380], dtype=np.int64),
        close=np.array([100.0, 101.0, 102.0, 99.0], dtype=np.float64),
        high=np.array([100.2, 101.2, 102.2, 99.2], dtype=np.float64),
        low=np.array([99.8, 100.8, 101.8, 98.8], dtype=np.float64),
        volume=np.full(4, 1_000.0, dtype=np.float64),
        symbols=["AAA"],
    )
    calendar = nb.SessionCalendar(start_minute=9 * 60 + 30, end_minute=9 * 60 + 33)
    result = nb.run_strategy_backtest(
        data=data,
        strategy=strategy,
        config=nb.BacktestConfig(max_position=2, slippage_bps=0.0, volume_share_impact=0.0),
        tradable_mask=calendar.tradable_mask(data.timestamps),
    )

    assert result.positions.shape == (4, 1)
    assert len(result.fills) >= 1


def test_strategy_loader_blocks_untrusted_modules_by_default() -> None:
    try:
        nb.load_strategy("json:JSONDecoder")
    except ValueError as exc:
        assert "refusing to import untrusted strategy module" in str(exc)
    else:
        raise AssertionError("expected untrusted module import to be blocked")


def test_compiled_policy_runner() -> None:
    data = nb.MarketData(
        timestamps=np.array([1, 2, 3, 4, 5], dtype=np.int64),
        close=np.array(
            [
                [10.0, 20.0],
                [11.0, 19.5],
                [12.0, 19.0],
                [11.5, 19.4],
                [12.5, 19.8],
            ],
            dtype=np.float64,
        ),
        high=np.array(
            [
                [10.1, 20.1],
                [11.1, 19.6],
                [12.1, 19.1],
                [11.6, 19.5],
                [12.6, 19.9],
            ],
            dtype=np.float64,
        ),
        low=np.array(
            [
                [9.9, 19.9],
                [10.9, 19.4],
                [11.9, 18.9],
                [11.4, 19.3],
                [12.4, 19.7],
            ],
            dtype=np.float64,
        ),
        volume=np.full((5, 2), 10_000.0, dtype=np.float64),
        symbols=["AAA", "BBB"],
    )

    targets = nb.compiled_momentum_targets(data.close, lookback=1, max_position=2)
    result = nb.run_compiled_policy_backtest(
        data=data,
        policy="momentum",
        lookback=1,
        max_position=2,
        config=nb.BacktestConfig(slippage_bps=0.0, volume_share_impact=0.0),
    )

    assert targets.shape == (5, 2)
    assert result.positions.shape == (5, 2)
    assert len(result.fills) >= 1


def test_compiled_research_primitives() -> None:
    close = np.array(
        [
            [10.0, 20.0, 30.0],
            [10.5, 19.0, 30.5],
            [11.0, 18.5, 31.0],
            [10.8, 19.2, 30.8],
            [11.4, 18.8, 31.4],
        ],
        dtype=np.float64,
    )

    vol = nb.compiled_rolling_volatility(close, window=2)
    ranks = nb.compiled_cross_sectional_rank(close, descending=True)
    weights = nb.compiled_minimum_variance_weights(close, window=2, leverage=1.0)
    cs_targets = nb.compiled_cross_sectional_momentum_targets(
        close,
        lookback=1,
        winners=1,
        losers=1,
        max_position=2,
    )
    filtered = nb.compiled_volatility_filtered_momentum_targets(
        close,
        lookback=1,
        vol_window=2,
        volatility_ceiling=1.0,
        max_position=2,
    )

    assert vol.shape == close.shape
    assert ranks.shape == close.shape
    assert weights.shape == close.shape
    assert cs_targets.shape == close.shape
    assert filtered.shape == close.shape
    assert ranks[0].tolist() == [3, 2, 1]
    assert np.allclose(weights[-1].sum(), 1.0)


def test_minimum_variance_policy_runner() -> None:
    data = nb.MarketData(
        timestamps=np.array([1, 2, 3, 4, 5, 6], dtype=np.int64),
        close=np.array(
            [
                [100.0, 50.0, 25.0],
                [100.2, 49.8, 25.2],
                [100.1, 49.7, 25.1],
                [100.3, 49.9, 25.4],
                [100.4, 50.1, 25.5],
                [100.5, 50.0, 25.7],
            ],
            dtype=np.float64,
        ),
        high=np.array(
            [
                [100.1, 50.1, 25.1],
                [100.3, 49.9, 25.3],
                [100.2, 49.8, 25.2],
                [100.4, 50.0, 25.5],
                [100.5, 50.2, 25.6],
                [100.6, 50.1, 25.8],
            ],
            dtype=np.float64,
        ),
        low=np.array(
            [
                [99.9, 49.9, 24.9],
                [100.1, 49.7, 25.1],
                [100.0, 49.6, 25.0],
                [100.2, 49.8, 25.3],
                [100.3, 50.0, 25.4],
                [100.4, 49.9, 25.6],
            ],
            dtype=np.float64,
        ),
        volume=np.full((6, 3), 20_000.0, dtype=np.float64),
        symbols=["A", "B", "C"],
    )

    result = nb.run_compiled_policy_backtest(
        data=data,
        policy="minimum_variance",
        window=2,
        leverage=1.0,
        gross_target=6,
        config=nb.BacktestConfig(max_position=6, slippage_bps=0.0, volume_share_impact=0.0),
    )

    assert result.positions.shape == (6, 3)
    assert len(result.fills) >= 1
