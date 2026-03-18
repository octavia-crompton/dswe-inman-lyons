"""time_utils.py – water-year and other temporal helpers for the Okavango project."""
from __future__ import annotations

import datetime as _dt


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
