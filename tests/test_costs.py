from __future__ import annotations

from pathlib import Path

import nanoback as nb


def test_calibrate_cost_model_returns_positive_params() -> None:
    calibration = nb.calibrate_cost_model(
        arrival_mid=[100.0, 100.0, 100.0, 100.0],
        fill_price=[100.02, 100.03, 99.98, 99.96],
        side=[1, 1, -1, -1],
        participation=[0.05, 0.10, 0.05, 0.10],
    )

    assert calibration.samples == 4
    assert calibration.fixed_slippage_bps >= 0.0
    assert calibration.volume_share_impact >= 0.0
    assert calibration.spread_bps > 0.0


def test_calibrate_cost_model_from_csv(tmp_path: Path) -> None:
    source = tmp_path / "fills.csv"
    source.write_text(
        "arrival_mid,fill_price,side,participation,quoted_spread_bps\n"
        "100,100.02,1,0.05,2.0\n"
        "100,100.03,1,0.10,2.0\n"
        "100,99.98,-1,0.05,2.0\n",
        encoding="utf-8",
    )

    calibration = nb.calibrate_cost_model_from_csv(source)
    config = calibration.to_config()

    assert calibration.samples == 3
    assert calibration.spread_bps == 2.0
    assert config.slippage_bps == calibration.fixed_slippage_bps
