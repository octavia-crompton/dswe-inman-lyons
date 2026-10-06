"""cache.py – tiny CSV cache for expensive (Earth Engine) computations.

    df = load_or_compute(FIG_DIR / "et_monthly.csv", compute_fn, run=RUN_EE)

* ``run=True``  → call ``compute_fn()``, write the CSV, return the frame.
* ``run=False`` → read the CSV if it exists, otherwise fall back to computing
  (so a missing cache never raises ``FileNotFoundError``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd


def load_or_compute(
    path: str | Path,
    compute: Callable[[], pd.DataFrame],
    run: bool = True,
    parse_dates: Sequence[str] = ("date",),
) -> pd.DataFrame:
    path = Path(path)
    if not run:
        if path.exists():
            df = pd.read_csv(path, parse_dates=[c for c in parse_dates if c])
            print(f"Loaded cached: {path}  ({len(df)} rows)")
            return df
        print(f"No cache at {path} — computing.")
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(df)} rows)")
    return df
