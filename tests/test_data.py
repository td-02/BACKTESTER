from __future__ import annotations

from pathlib import Path

import numpy as np

import nanoback as nb


def test_load_csv_directory(tmp_path: Path) -> None:
    (tmp_path / "AAA.csv").write_text(
        "timestamp,close,high,low,volume\n1,10,11,9,100\n2,11,12,10,100\n",
        encoding="utf-8",
    )
    (tmp_path / "BBB.csv").write_text(
        "timestamp,close,high,low,volume\n1,20,21,19,200\n2,19,20,18,200\n",
        encoding="utf-8",
    )

    data = nb.load_csv(tmp_path)

    assert data.symbols == ["AAA", "BBB"]
    assert data.close.shape == (2, 2)
    assert np.allclose(data.volume[:, 1], np.array([200.0, 200.0]))
