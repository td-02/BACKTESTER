from __future__ import annotations

import numpy as np

import nanoback as nb
from nanoback import _nanoback as core


def test_buffer_matrix_view_shares_memory() -> None:
    matrix = np.arange(20, dtype=np.float64).reshape(5, 4)
    view = core.buffer_matrix_view(matrix)
    engine_view = np.frombuffer(view, dtype=np.float64).reshape(matrix.shape)

    assert np.shares_memory(matrix, engine_view)
    engine_view[0, 0] = 123.0
    assert matrix[0, 0] == 123.0


def test_run_backtest_matrix_rejects_wrong_dtype_buffers() -> None:
    timestamps = np.array([1, 2], dtype=np.int64)
    close = np.array([[100.0], [101.0]], dtype=np.float64)
    targets = np.array([[1], [1]], dtype=np.int64)

    with np.testing.assert_raises(Exception):
        core.run_backtest_matrix(
            timestamps=timestamps.astype(np.float32),
            close=close,
            high=close,
            low=close,
            volume=np.array([[1000.0], [1000.0]], dtype=np.float64),
            bid=close,
            ask=close,
            target_positions=targets,
            order_types=np.full((2, 1), int(nb.OrderType.MARKET), dtype=np.int8),
            limit_prices=np.full((2, 1), np.nan, dtype=np.float64),
            tradable_mask=np.array([1, 1], dtype=np.uint8),
            asset_max_positions=np.array([10], dtype=np.int64),
            asset_notional_limits=np.array([0.0], dtype=np.float64),
            config=nb.BacktestConfig(),
            snapshot=None,
            start_row=0,
            end_row=2,
        )
