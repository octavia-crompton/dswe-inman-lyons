"""time_utils.py – water-year and other temporal helpers for the Okavango project."""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd


def water_year(dt: _dt.datetime, start_month: int = 10) -> int:
    """Return the water year for a datetime.

    By default uses the **austral** convention (Oct 1 – Sep 30),
    so ``water_year(datetime(2015, 11, 1))`` → 2015 and
    ``water_year(datetime(2016, 3, 1))`` → 2015.

    Parameters
    ----------
    dt : datetime-like
        Must expose ``.year`` and ``.month`` attributes.
    start_month : int
        First month of the water year (default 10 = October).
    """
    return dt.year if dt.month >= start_month else dt.year - 1


def assign_water_year(
    df: pd.DataFrame,
    start_month: int = 7,
    label: str = "ending",
) -> pd.Series:
    """Assign a water-year integer to each row of *df*.

    Parameters
    ----------
    df : DataFrame
        Must have a ``DatetimeIndex`` **or** columns ``year`` and ``month_num``.
    start_month : int
        First month of the water year (e.g. 7 for Jul–Jun).
    label : {"ending", "starting"}
        ``"ending"`` labels Jul 2015 – Jun 2016 as WY 2016;
        ``"starting"`` labels it as WY 2015.

    Returns
    -------
    pd.Series[int]
        Water-year value for each row, same index as *df*.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        year = df.index.year.astype(int)
        month = df.index.month.astype(int)
    else:
        year = df["year"].astype(int)
        month = df["month_num"].astype(int)
        if month.min() == 0:
            month = month + 1

    if label == "ending":
        if start_month == 1:
            return pd.Series(year, index=df.index, name="water_year")
        return pd.Series(
            np.where(month >= start_month, year + 1, year),
            index=df.index,
            name="water_year",
        )
    if label == "starting":
        return pd.Series(
            np.where(month < start_month, year - 1, year),
            index=df.index,
            name="water_year",
        )
    raise ValueError("label must be 'ending' or 'starting'")
