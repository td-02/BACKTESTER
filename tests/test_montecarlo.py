from __future__ import annotations

import numpy as np

import nanoback as nb


def test_montecarlo_shuffle_returns_percentiles() -> None:
    returns = np.array([0.01, -0.005, 0.002, 0.004, -0.003] * 120, dtype=np.float64)
    mc = nb.MonteCarlo(returns, seed=7)
    result = mc.run(n_sims=200, method="shuffle")

    assert result.sharpe.shape[0] == 200
    assert "p50" in result.sharpe_percentiles
    assert 0.0 <= result.p_value_proxy <= 1.0


def test_montecarlo_block_bootstrap_runs() -> None:
    returns = np.array([0.01, -0.005, 0.002, 0.004, -0.003] * 120, dtype=np.float64)
    mc = nb.MonteCarlo(returns, seed=11)
    result = mc.run(n_sims=150, method="block_bootstrap", block_size=20)

    assert result.max_drawdown.shape[0] == 150
    assert "p95" in result.cagr_percentiles
