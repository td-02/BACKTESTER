from __future__ import annotations

import pytest

import nanoback as nb


def test_load_corporate_actions_csv(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    path = tmp_path / "actions.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "ex_date": ["2024-01-02", "2024-01-03"],
            "action_type": ["SPLIT", "DIVIDEND"],
            "value": [2.0, 0.5],
        }
    ).to_csv(path, index=False)
    actions = nb.load_corporate_actions_csv(path, {"AAA": 0})
    assert len(actions) == 2
    assert int(actions[0].asset) == 0
